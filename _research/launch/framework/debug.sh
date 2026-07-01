#!/bin/bash
#
# debug.sh — run a (SIZE, RECIPE) composition INTERACTIVELY for a few iters.
# The interactive twin of submit.sh: same --size/--recipe, but runs inline via
# ../interactive-run.sh (torchrun, no SLURM srun wrapper) with short-run defaults.
#
# Pre-req: you're already inside an interactive allocation inside the container.
# Grab one on a GH200 node with the deep_gemm image the framework defaults to:
#
#   CONTAINER=/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-a139.toml
#   srun --account=infra01 --time=01:00:00 --nodes=1 --gpus-per-node=4 \
#        --cpus-per-task=72 --mem=460000 --mpi=pmix \
#        --network=disable_rdzv_get --environment=$CONTAINER --pty bash
#   # ... now inside the container, from the repo root ...
#
#   bash _research/launch/framework/debug.sh --size 270m-moe --recipe md_decoupling
#   bash _research/launch/framework/debug.sh --size 1.5b-moe-128e --recipe md_decoupling --nproc 4 --iters 10
#   bash _research/launch/framework/debug.sh --size 420m-moe --recipe md_decoupling -- --lr 1e-4  # extra flags override
#   bash _research/launch/framework/debug.sh --size 420m-moe --recipe md_decoupling --dry-run     # print args, don't run
#
# Defaults: 4 ranks, 20 iters, 10-min exit cap, deps installed once (SKIP_DEPS=1
# to skip on later runs). MEGATRON_DATA_PATH=<prefix> overrides the data blend.
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
INTERACTIVE_RUN="$FRAMEWORK_DIR/../interactive-run.sh"

NPROC=${NPROC_PER_NODE:-4}; ITERS=20; DRY=""; EXTRA=(); CLUSTER=${CLUSTER:-alps3}
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --nproc)   NPROC="$2"; shift 2 ;;
        --iters)   ITERS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --)        shift; EXTRA=("$@"); break ;;
        *) echo "unknown arg: $1 (use -- to pass extra megatron flags)" >&2; exit 1 ;;
    esac
done
: "${SIZE:?usage: debug.sh --size <size> --recipe <recipe> [--cluster C] [--nproc N] [--iters N] [--dry-run] [-- extra flags]}"
: "${RECIPE:?usage: debug.sh --size <size> --recipe <recipe> [--cluster C] [--nproc N] [--iters N] [--dry-run] [-- extra flags]}"
[ -f "$FRAMEWORK_DIR/clusters/$CLUSTER.sh" ] || { echo "no such cluster: $CLUSTER (see $FRAMEWORK_DIR/clusters/)" >&2; exit 1; }

# The framework trains in SAMPLES (--train-samples), and Megatron forbids mixing
# that with --train-iters. So a short debug run is just a small TRAIN_SAMPLES
# override (iters * GBS), keeping the run sample-based and identical in shape to
# a real launch — only shorter (LR also decays over this short budget).
SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }
GBS=$(grep -E '^GBS=' "$SIZE_FILE" | head -1 | sed -E 's/^GBS=\$\{GBS:-([0-9]+)\}.*/\1/; s/^GBS=([0-9]+).*/\1/')
: "${GBS:?could not read GBS from $SIZE_FILE}"
TRAIN_SAMPLES=$(( ITERS * GBS ))

export SIZE RECIPE CLUSTER TRAIN_SAMPLES
# Debug runs land in a separate wandb board ("debug-<base>") so they don't
# clutter the real one. Override: WANDB_PROJECT_PREFIX= (empty) for the base
# board, or WANDB_PROJECT=... for an explicit name.
export WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-debug-}"

if [ -n "$DRY" ]; then
    echo ">>> DRY RUN: SIZE=$SIZE RECIPE=$RECIPE CLUSTER=$CLUSTER  (composed args below; not launching)"
    SIZE="$SIZE" RECIPE="$RECIPE" CLUSTER="$CLUSTER" bash "$FRAMEWORK_DIR/lib/dump-args.sh" | sed 's/^/  /'
    echo ">>> would run: NPROC_PER_NODE=$NPROC TRAIN_SAMPLES=$TRAIN_SAMPLES ($ITERS iters * GBS $GBS) interactive-run.sh train.sbatch --exit-duration-in-mins 10 ${EXTRA[*]}"
    exit 0
fi

echo ">>> debug: $ITERS iters -> TRAIN_SAMPLES=$TRAIN_SAMPLES (GBS=$GBS)"
# Short-run debug overrides go LAST so they win (argparse last-value-wins).
exec env NPROC_PER_NODE="$NPROC" SIZE="$SIZE" RECIPE="$RECIPE" CLUSTER="$CLUSTER" TRAIN_SAMPLES="$TRAIN_SAMPLES" \
    bash "$INTERACTIVE_RUN" "$FRAMEWORK_DIR/train.sbatch" \
        --exit-duration-in-mins 10 \
        "${EXTRA[@]}"
