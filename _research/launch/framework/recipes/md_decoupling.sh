# shellcheck shell=bash
#
# md_decoupling — magnitude-direction decoupling (--optimizer md_decoupling).
#
# Hypersphere normalization (direction) on 2D matrices + router, learnable
# per-axis gains (magnitude), and Muon orthogonalized updates. Embedding + LM
# head ALWAYS use the Adam branch. This is the recipe from submit_test.sh,
# ported into the framework.
#
# md_decoupling is INCOMPATIBLE with the standard --use-distributed-optimizer;
# this recipe shards optimizer state with --use-layer-wise-distributed-optimizer
# instead (DIST_OPT_ARGS, consumed by lib/common.sh). Learnable gains require
# --ckpt-format torch (already set in common.sh).
#
# Knobs (override via env, e.g. `MLR=8e-3 bash submit.sh ...`):
#   LR        scalar-group / base LR (LM head, biases, norm gains, per-axis gains
#             unless GAINS_LR set); drives everything not on the matrix group
#   MLR       matrix LR — the 2D / Muon param group; tune independently of LR
#   GAINS_LR  absolute LR for the per-axis gains AdamW (unset → falls back to LR)
#   MIN_LR    LR floor; --min-lr-mode absolute floors EVERY group at MIN_LR
#   RADIUS_FROM_INIT  1 → put each flat matrix sphere at its init Frobenius norm
#             (sqrt(min/hidden) rescale) instead of sqrt(max); keeps narrow
#             matrices (MLA lora, MoE fc2, GQA K/V) on their init sphere

OPTIMIZER=md_decoupling
EXP_TAG=md

LR=${LR:-1e-3}
MLR=${MLR:-1e-2}
GAINS_LR=${GAINS_LR:-1e-3}
MIN_LR=${MIN_LR:-1e-4}
KNOB_STR=lr${LR}-mlr${MLR}
[ "$GAINS_LR" != "$LR" ] && KNOB_STR=${KNOB_STR}-glr${GAINS_LR}

# Put narrow flat matrices on their init sphere instead of sqrt(max).
RADIUS_FROM_INIT=${RADIUS_FROM_INIT:-1}
RADIUS_FROM_INIT_ARGS=()
if [ "$RADIUS_FROM_INIT" = 1 ]; then
    RADIUS_FROM_INIT_ARGS=(--hypersphere-radius-from-init)
fi

# md_decoupling's canonical regularization (no weight decay; Adam betas reused
# by the Adam branch + the gains AdamW).
WEIGHT_DECAY=0.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# Linear decay to MIN_LR over all of TRAIN_SAMPLES (auto-scales with run length;
# no cooldown to tune). Overridable from the env, e.g. LR_DECAY_STYLE=WSD.
LR_DECAY_STYLE=${LR_DECAY_STYLE:-linear}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-0}

# Shard optimizer state over DP with the layer-wise wrapper (md_decoupling
# forbids the standard distributed optimizer). Overrides common.sh's default
# DIST_OPT_ARGS=(--use-distributed-optimizer --overlap-grad-reduce).
DIST_OPT_ARGS=(
    --use-layer-wise-distributed-optimizer
    --overlap-grad-reduce
    # --overlap-param-gather omitted: corrupts the Newton-Schulz orthogonalization.
)

RECIPE_ARGS=(
    # LR routing: matrix (2D / Muon) params at MLR; --lr drives everything else
    # incl. gains; all groups floor at --min-lr via absolute mode.
    --matrix-lr "$MLR"
    --gains-lr "$GAINS_LR"
    --min-lr-mode absolute
    # Muon orthogonalized updates on matrix params.
    --use-orthogonal-updates
    --muon-scale-mode shape_up
    --muon-momentum 0.95
    --muon-use-nesterov
    # Hypersphere (direction): flat on matrices, off on embedding/LM head,
    # row-wise on the router.
    --hypersphere-mode flat
    --hypersphere-embedding-mode none
    --hypersphere-router-mode row
    # NOTE: --hypersphere-scale-out-proj-init (1/sqrt(2*num_layers) on the out-proj
    # weight-norm/gain) is intentionally NOT set: common.sh enables
    # --residual-output-scaling, whose forward multiplier alpha=1/sqrt(2*num_layers)
    # is the single depth-scaling mechanism. Enabling both double-counts (the two are
    # mutually exclusive and asserted against in arguments.py validation).
    "${RADIUS_FROM_INIT_ARGS[@]}"
    # Router uses the Muon branch.
    --md-router-use-orthogonal-updates True
    # Learnable per-axis gains (magnitude), softplus-parametrized (positive).
    --hypersphere-gains-mode rowcol
    --gain-parametrization softplus
    # Grad-accum fusion is incompatible with the custom optimizer's grad path.
    --no-gradient-accumulation-fusion
)
