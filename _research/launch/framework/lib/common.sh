# shellcheck shell=bash
#
# common.sh — the invariant training scaffolding for every run.
#
# This file is SOURCED by train.sbatch AFTER the chosen size/, recipe/ and
# clusters/ fragments have run. Those fragments set a handful of variables and
# arrays (the "contract" below); this file fills in every default they didn't
# set, assembles the final MEGATRON_ARGS, wires up wandb, and launches via srun.
#
# You almost never edit this file. Everything experiment-specific lives in
# sizes/<size>.sh (model + data scale) and recipes/<name>.sh (optimizer/arch);
# everything machine-specific lives in clusters/<cluster>.sh (container, mpi
# flavor, precision, MoE dispatcher, sbatch flags — contract in clusters/alps3.sh).
#
# ── contract ────────────────────────────────────────────────────────────────
# A size file MUST set:
#   NUM_LAYERS HIDDEN FFN_HIDDEN NUM_HEADS NUM_KV_HEADS SEQ_LEN
#   MBS GBS TRAIN_SAMPLES SAVE_INTERVAL APERTUS_TRACK
#   MOE_ARGS=(...)        # array; empty () for a dense model
# and MAY set (else defaulted here):
#   TP PP CP EP  INIT_STD  EXIT_DURATION_MINS
#   DEFAULT_NODES DEFAULT_TIME         # read by submit.sh, not used here
#
# A recipe file MUST set:
#   OPTIMIZER            # --optimizer value (adam|sgd|muon|lion|md_decoupling)
#   EXP_TAG              # short slug for the run name, e.g. "md"
#   RECIPE_ARGS=(...)    # optimizer/arch-specific flags (the actual idea)
# and MAY set (else defaulted here):
#   LR MIN_LR  KNOB_STR                # KNOB_STR is appended to EXP_NAME
#                                      # (RUN_TAG, from the env, is appended too)
#   WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 CLIP_GRAD
#   LR_DECAY_STYLE LR_WARMUP_SAMPLES LR_WSD_DECAY_STYLE LR_WSD_DECAY_SAMPLES
#   DIST_OPT_ARGS=(...)                # optimizer-state sharding flags; default
#                                      # (--use-distributed-optimizer
#                                      #  --overlap-grad-reduce). md_decoupling
#                                      # overrides to --use-layer-wise-distributed-optimizer
#   EXTRA_REG_ARGS=(...)               # escape hatch for odd reg flags
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

: "${SIZE:?set SIZE (see _research/launch/framework/sizes/)}"
: "${RECIPE:?set RECIPE (see _research/launch/framework/recipes/)}"

[ -z "${DRY_RUN:-}" ] && echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

# ── paths & environment (identical across every run) ─────────────────────────
REPO_DIR=/iopsstor/scratch/cscs/$USER/megatron-apertus-moe
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd "$WORKDIR"

# ARCH_TAG (from clusters/<cluster>.sh) keeps compiled-kernel caches and
# pip-installed packages separate per GPU arch; empty on alps3.
PACKAGE_DIR=$REPO_DIR/_research/packages${ARCH_TAG:-}
DATASET_CACHE_DIR=$REPO_DIR/cache/dataset
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton${ARCH_TAG:-}
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor${ARCH_TAG:-}
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export TMPDIR=$TMP_CACHE_DIR
# Data is pre-tokenized .bin/.idx; the HF tokenizer is only loaded for vocab
# metadata, so its thread pool is useless — silence the fork warning.
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATASET_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TMP_CACHE_DIR" "$PACKAGE_DIR" \
    _research/results/runs _research/results/performance _research/results/ckpts

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=${MASTER_PORT:-25678}

# ── defaults for anything the size/recipe fragment didn't set ────────────────
TP=${TP:-1}; PP=${PP:-1}; CP=${CP:-1}; EP=${EP:-1}
# Init std defaults to 1/sqrt(HIDDEN) (e.g. 1024 -> 0.03125).
INIT_STD=${INIT_STD:-$(awk "BEGIN{printf \"%.6g\", 1.0/sqrt($HIDDEN)}")}
EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-595}

LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
# Linear decay to MIN_LR over all of TRAIN_SAMPLES is the default: it auto-scales
# with run length and has no cooldown to tune. The WSD knobs below only take
# effect when a recipe or the env sets LR_DECAY_STYLE=WSD.
LR_DECAY_STYLE=${LR_DECAY_STYLE:-linear}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-0}
LR_WSD_DECAY_STYLE=${LR_WSD_DECAY_STYLE:-minus_sqrt}
LR_WSD_DECAY_SAMPLES=${LR_WSD_DECAY_SAMPLES:-732422}
KNOB_STR=${KNOB_STR:-lr${LR}}
# A size file may export ARCH_KNOB_STR for architecture knobs the recipe-owned
# KNOB_STR can't see. Appended here, after the recipe has set KNOB_STR, so it
# lands in EXP_NAME (run/wandb name).
[ -n "${ARCH_KNOB_STR:-}" ] && KNOB_STR="${KNOB_STR}-${ARCH_KNOB_STR}"

# Default to an empty array ONLY when the fragment didn't declare one. (The
# `("${X[@]:-}")` idiom would inject a phantom empty-string element instead.)
declare -p RECIPE_ARGS    >/dev/null 2>&1 || RECIPE_ARGS=()
declare -p MOE_ARGS       >/dev/null 2>&1 || MOE_ARGS=()
declare -p MLA_ARGS       >/dev/null 2>&1 || MLA_ARGS=()
declare -p EXTRA_REG_ARGS >/dev/null 2>&1 || EXTRA_REG_ARGS=()
# Escape hatch for extra architecture/network flags from a size file (appended
# right after NETWORK_SIZE_ARGS), e.g. --post-norm for arch ablations.
declare -p EXTRA_NETWORK_ARGS >/dev/null 2>&1 || EXTRA_NETWORK_ARGS=()

# Optimizer-state sharding. Default = the standard distributed optimizer (adam,
# muon, ...). md_decoupling forbids it and overrides DIST_OPT_ARGS in its recipe
# (--use-layer-wise-distributed-optimizer).
declare -p DIST_OPT_ARGS >/dev/null 2>&1 || DIST_OPT_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
)

# Cluster defaults (= alps3) for legacy invocations that source no cluster file.
declare -p SRUN_EXTRA_ARGS      >/dev/null 2>&1 || SRUN_EXTRA_ARGS=(--network=disable_rdzv_get)
declare -p CLUSTER_TRAIN_ARGS   >/dev/null 2>&1 || CLUSTER_TRAIN_ARGS=()
declare -p MIXED_PRECISION_ARGS >/dev/null 2>&1 || MIXED_PRECISION_ARGS=(--bf16)
# Container: absolute path used as-is; bare filename resolved under
# _research/launch/. Default = gfu's deep_gemm image (see clusters/alps3.sh).
CONTAINER=${CONTAINER:-/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-a139.toml}
case "$CONTAINER" in
    /*) CONTAINER_PATH=$CONTAINER ;;
    *)  CONTAINER_PATH=$WORKDIR/_research/launch/$CONTAINER ;;
esac

# ── MoE routing policy: DeepSeek-V3-style, invariant across the whole ladder ──
# The small MoE rungs are iso-sparsity proxies (~5% active/total non-embedding)
# for a 670B-A40B target, so they all share one routing recipe; only the expert
# *geometry* (count, expert+shared width, dense-layer fraction) scales per size
# and lives in sizes/<size>.sh's MOE_ARGS. A size declares itself MoE by leaving
# MOE_ARGS non-empty (dense sizes set MOE_ARGS=() and skip all of this).
#
# DeepSeek-V3 routing = sigmoid gating + aux-loss-free per-expert bias (the bias
# steers selection only, not the gate weights; updated at moe-router-bias-update-
# rate) + a tiny complementary seq-wise aux loss + top-k renorm scaled by 2.5,
# with the router math in fp32 for stability at high expert counts. See
# https://arxiv.org/abs/2412.19437 (DeepSeek-V3) and arxiv 2408.15664 (loss-free
# balancing). MOE_AUX_LOSS_COEFF is the one knob worth sweeping.
if [ "${#MOE_ARGS[@]}" -gt 0 ]; then
    MOE_ARGS+=(
        --moe-router-score-function sigmoid
        --moe-router-enable-expert-bias
        --moe-router-bias-update-rate 1e-3
        --moe-router-topk-scaling-factor 2.5
        --moe-router-dtype fp32
        --moe-router-load-balancing-type seq_aux_loss
        --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-1e-3}"
        --moe-grouped-gemm
    )
    [ "${MOE_PERMUTE_FUSION:-1}" = 1 ] && MOE_ARGS+=(--moe-permute-fusion)
    MOE_ARGS+=(
        --moe-token-dispatcher-type "${MOE_DISPATCHER:-allgather}"
        --moe-per-layer-logging
    )
fi

# ── run identity ─────────────────────────────────────────────────────────────
# wandb project: WANDB_PROJECT overrides outright; otherwise WANDB_PROJECT_PREFIX
# (debug.sh sets "debug-") is prepended to the base name so debug runs land in a
# separate board.
PROJECT=${WANDB_PROJECT:-${WANDB_PROJECT_PREFIX:-}apertus-moe-optim-baseline}
# Run name = size + recipe tag + swept knobs (+ CLUSTER_TAG on non-default
# clusters) (+ optional RUN_TAG to disambiguate any axis KNOB_STR doesn't cover).
EXP_NAME=${EXP_NAME:-${SIZE}-${EXP_TAG}-${KNOB_STR}${CLUSTER_TAG:-}${RUN_TAG:+-${RUN_TAG}}}
CKPT_DIR=${CKPT_DIR:-$WORKDIR/_research/results/ckpts/$EXP_NAME}
mkdir -p "$CKPT_DIR"

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=${APERTUS_FEATURE:-transformer-pp}
export APERTUS_TRACK=${APERTUS_TRACK:-$SIZE}
export APERTUS_RUN_NAME=$EXP_NAME-${SLURM_JOB_ID:-interactive}
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

# ── invariant arg groups ─────────────────────────────────────────────────────
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN"
    --ffn-hidden-size "$FFN_HIDDEN"
    --num-attention-heads "$NUM_HEADS"
)
# Attention type: a size file sets MLA_ARGS (non-empty) to use Multi-Latent
# Attention (DeepSeek-V3 style; the args carry the lora ranks + per-head dims);
# otherwise the default is GQA via NUM_KV_HEADS.
if [ "${#MLA_ARGS[@]}" -gt 0 ]; then
    NETWORK_SIZE_ARGS+=(--multi-latent-attention "${MLA_ARGS[@]}")
else
    NETWORK_SIZE_ARGS+=(--group-query-attention --num-query-groups "$NUM_KV_HEADS")
fi
NETWORK_SIZE_ARGS+=(
    --max-position-embeddings "$SEQ_LEN"
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length "$SEQ_LEN"
)

TRAINING_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --train-samples "$TRAIN_SAMPLES"
    --exit-duration-in-mins "$EXIT_DURATION_MINS"
    --log-interval 1
    --eval-interval "$TRAIN_SAMPLES"
    --eval-iters 0
    --optimizer "$OPTIMIZER"
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
)
[ "${#CLUSTER_TRAIN_ARGS[@]}" -gt 0 ] && TRAINING_ARGS+=("${CLUSTER_TRAIN_ARGS[@]}")

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay "$WEIGHT_DECAY"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
)
[ -n "${CLIP_GRAD:-}" ] && REGULARIZATION_ARGS+=(--clip-grad "$CLIP_GRAD")
[ "${#EXTRA_REG_ARGS[@]}" -gt 0 ] && REGULARIZATION_ARGS+=("${EXTRA_REG_ARGS[@]}")

LEARNING_RATE_ARGS=(
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-warmup-samples "$LR_WARMUP_SAMPLES"
)
if [ "$LR_DECAY_STYLE" = "WSD" ]; then
    LEARNING_RATE_ARGS+=(
        --lr-wsd-decay-style "$LR_WSD_DECAY_STYLE"
        --lr-wsd-decay-samples "$LR_WSD_DECAY_SAMPLES"
    )
fi
# LR_DECAY_SAMPLES: decouple the LR decay length from TRAIN_SAMPLES. Defaults
# (unset) to TRAIN_SAMPLES in Megatron.
[ -n "${LR_DECAY_SAMPLES:-}" ] && LEARNING_RATE_ARGS+=(--lr-decay-samples "$LR_DECAY_SAMPLES")
# OVERRIDE_OPT_SCHED=1: rebuild the LR schedule from the args above instead of
# the one stored in the checkpoint (e.g. branching a cooldown off a constant-LR
# run). Optimizer state and iteration are still loaded; only the schedule is replaced.
[ -n "${OVERRIDE_OPT_SCHED:-}" ] && LEARNING_RATE_ARGS+=(--override-opt-param-scheduler)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std "$INIT_STD"
)

# MIXED_PRECISION_ARGS comes from clusters/<cluster>.sh (alps3: bf16) — defaulted
# above with the other cluster knobs.

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --expert-model-parallel-size "$EP"
    "${DIST_OPT_ARGS[@]}"
)

LOGGING_ARGS=(
    --save "$CKPT_DIR"
    --load "$CKPT_DIR"
    --save-interval "$SAVE_INTERVAL"
    --ckpt-format torch
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir "$WORKDIR/_research/results/runs/tb-${SLURM_JOB_ID:-interactive}"
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)
# PRETRAINED_CKPT: seed a fresh run from another run's weights WITHOUT its
# optimizer state (Megatron finetunes: counters/optimizer/RNG/dataloader reset)
# on the FIRST load only; once this run writes its own checkpoint, normal resume
# takes over.
[ -n "${PRETRAINED_CKPT:-}" ] && LOGGING_ARGS+=(--pretrained-checkpoint "$PRETRAINED_CKPT")
# NO_LOAD_OPTIM=1: load model (+ iteration / consumed-samples / RNG so the data
# loader CONTINUES) from CKPT_DIR but start a fresh optimizer + LR schedule. Pair
# with LR_DECAY_SAMPLES to anneal over just the continuation. Not crash-resume-safe.
[ -n "${NO_LOAD_OPTIM:-}" ] && LOGGING_ARGS+=(--no-load-optim)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL:-alehc/swissai-tokenizer}"
)

# Dataset: a blend of swissai sources under DATA_ROOT. Each source dir holds many
# dump-N-merged.{bin,idx} shards; we glob every shard prefix and hand the flat
# list to --train-data-path, letting Megatron infer blend weights from the shard
# lengths (so the mixture is token-proportional across sources). Override the
# mixture with DATA_ROOT / DATA_SOURCES, or set MEGATRON_DATA_PATH to fall back
# to a single --data-path prefix (e.g. the GPT2BPE climbmix debug set).
DATA_ROOT=${DATA_ROOT:-/iopsstor/scratch/cscs/jpcoles/a06}
DATA_SOURCES=${DATA_SOURCES:-"
    swissai-dclm-edu-filterrobots_fine-merge
    swissai-fineweb-2-quality_10-filterrobots-merge/euro-high
    swissai-fineweb-2-quality_10-filterrobots-merge/euro-mid
    swissai-fineweb-2-quality_10-filterrobots-merge/other-high
"}

if [ -n "${MEGATRON_DATA_PATH:-}" ]; then
    # Single-prefix escape hatch (quick overrides, dry-diff tooling). NOTE: the
    # climbmix set is GPT2BPE-tokenized — pair it with TOKENIZER_MODEL/-type
    # overrides if the vocab must match the data.
    DATA_ARGS=(--data-path "$MEGATRON_DATA_PATH" --split 99,1,0)
else
    TRAIN_DATA_PATHS=()
    for src in $DATA_SOURCES; do
        d="$DATA_ROOT/$src"
        [ -d "$d" ] || { echo "data source not found: $d" >&2; exit 1; }
        while IFS= read -r p; do
            TRAIN_DATA_PATHS+=("$p")
        done < <(find "$d" -type f \( -name '*.bin' -o -name '*.idx' \) \
                     | sed -E 's/\.[^.]+$//' | sort -u)
    done
    [ "${#TRAIN_DATA_PATHS[@]}" -gt 0 ] || { echo "no .bin/.idx shards found under $DATA_ROOT" >&2; exit 1; }
    DATA_ARGS=(--train-data-path "${TRAIN_DATA_PATHS[@]}")
fi
DATA_ARGS+=(
    --data-cache-path "$DATASET_CACHE_DIR"
    --num-workers ${NUM_WORKERS:-4}
)

# Flat array. RECIPE_ARGS sits where each legacy file put its optimizer block.
MEGATRON_ARGS=(
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
)
[ "${#EXTRA_NETWORK_ARGS[@]}" -gt 0 ] && MEGATRON_ARGS+=("${EXTRA_NETWORK_ARGS[@]}")
[ "${#MOE_ARGS[@]}" -gt 0 ] && MEGATRON_ARGS+=("${MOE_ARGS[@]}")
MEGATRON_ARGS+=(
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
)
[ "${#RECIPE_ARGS[@]}" -gt 0 ] && MEGATRON_ARGS+=("${RECIPE_ARGS[@]}")
MEGATRON_ARGS+=(
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-${SLURM_JOB_ID:-interactive}
    MEGATRON_ARGS+=(
        --wandb-project "$PROJECT"
        --wandb-exp-name "$EXP_NAME-${SLURM_JOB_ID:-interactive}"
        --wandb-save-dir "$WORKDIR/_research/results/runs"
    )
else
    export WANDB_MODE=disabled
fi

# When sourced for arg inspection (DRY_RUN=1) or by an interactive runner
# (COMMON_NO_LAUNCH=1), stop here: the caller just wants MEGATRON_ARGS populated.
if [ -n "${DRY_RUN:-}" ] || [ -n "${COMMON_NO_LAUNCH:-}" ]; then
    echo "SIZE=$SIZE  RECIPE=$RECIPE  OPTIMIZER=$OPTIMIZER  EXP_NAME=$EXP_NAME" >&2
    return 0 2>/dev/null || exit 0
fi

# Shell-quote every token: the args are flattened into one string and re-parsed
# by `bash -c` below, so values with shell metacharacters (e.g.
# --moe-layer-freq '([0]*1+[1]*13)') must stay quoted or bash chokes on the '('.
MEGATRON_ARGS_STR="${MEGATRON_ARGS[*]@Q}"

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((${SLURM_NTASKS:-4}/${SLURM_NNODES:-1})); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"

echo "SIZE=$SIZE  RECIPE=$RECIPE  OPTIMIZER=$OPTIMIZER"
echo "EXP_NAME:  $EXP_NAME"
echo "CKPT_DIR:  $CKPT_DIR"
echo "CONTAINER: $CONTAINER_PATH"
echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
SRUN_RC=0
srun -lu --mpi="${SRUN_MPI:-pmix}" "${SRUN_EXTRA_ARGS[@]}" \
    --environment="$CONTAINER_PATH" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD" \
    || SRUN_RC=$?

echo "END TIME: $(date)"

# ── auto-requeue: chain the next allocation if training isn't done yet ────────
# Enabled by `submit.sh --auto-requeue` (sets AUTO_REQUEUE=1 in the job env).
# We chain ONLY on a clean exit — rc 0 means Megatron either finished or hit
# --exit-duration-in-mins and saved a checkpoint. A crash (rc != 0) stops the
# chain so a human investigates. The next job shares EXP_NAME -> same CKPT_DIR
# -> resumes via --load.
maybe_auto_requeue() {
    [ -n "${AUTO_REQUEUE:-}" ] || return 0

    if [ "$SRUN_RC" -ne 0 ]; then
        echo ">>> auto-requeue: srun exited $SRUN_RC (not clean) — NOT chaining."
        return 0
    fi

    local rq_count=${REQUEUE_COUNT:-0} rq_max=${MAX_REQUEUES:-50}
    if [ "$rq_count" -ge "$rq_max" ]; then
        echo ">>> auto-requeue: hit MAX_REQUEUES=$rq_max — NOT chaining."
        return 0
    fi

    local marker="$CKPT_DIR/latest_checkpointed_iteration.txt"
    if [ ! -f "$marker" ]; then
        echo ">>> auto-requeue: no checkpoint written yet — NOT chaining."
        return 0
    fi
    local iter target_iters
    iter=$(tr -dc '0-9' < "$marker")
    if [ -z "$iter" ]; then
        echo ">>> auto-requeue: marker not numeric ('$(cat "$marker")') — NOT chaining."
        return 0
    fi
    # Compare ITERATIONS with Megatron's floor semantics (train_iters =
    # train_samples // GBS): comparing raw samples chains forever when
    # TRAIN_SAMPLES isn't a multiple of GBS.
    target_iters=$(( TRAIN_SAMPLES / GBS ))
    if [ "$iter" -ge "$target_iters" ]; then
        echo ">>> auto-requeue: reached iter $iter/$target_iters — DONE."
        return 0
    fi

    echo ">>> auto-requeue: iter $iter/$target_iters (chain #$((rq_count+1))/$rq_max) — submitting dependent job."
    sbatch \
        --dependency=afterany:"${SLURM_JOB_ID}",singleton \
        --nodes="${SLURM_NNODES:-1}" \
        --time="${REQUEUE_TIME:-${DEFAULT_TIME:-10:00:00}}" \
        --job-name="${SLURM_JOB_NAME:-$SIZE-$RECIPE}" \
        ${CLUSTER_SBATCH_FLAGS[@]+"${CLUSTER_SBATCH_FLAGS[@]}"} \
        ${RESERVATION:+--reservation="$RESERVATION"} \
        --export=ALL,SIZE="$SIZE",RECIPE="$RECIPE",CLUSTER="${CLUSTER:-alps3}",FRAMEWORK_DIR="$FRAMEWORK_DIR",AUTO_REQUEUE=1,REQUEUE_COUNT=$((rq_count+1)),MAX_REQUEUES="$rq_max",REQUEUE_TIME="${REQUEUE_TIME:-${DEFAULT_TIME:-10:00:00}}" \
        "$FRAMEWORK_DIR/train.sbatch"
}
maybe_auto_requeue

exit "$SRUN_RC"
