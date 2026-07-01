#!/bin/bash
#
# interactive-run.sh — run the framework's training command INLINE in your
# current interactive shell (skips the wrapping srun in lib/common.sh).
#
# Pre-req: you're already inside an interactive allocation AND the container.
# Grab one on a GH200 node (alps3) with the deep_gemm image the framework
# defaults to:
#
#   CONTAINER=/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-a139.toml
#   srun --account=infra01 --time=01:00:00 --nodes=1 --gpus-per-node=4 \
#        --cpus-per-task=72 --mem=460000 --mpi=pmix \
#        --network=disable_rdzv_get --environment=$CONTAINER --pty bash
#   # ... now inside the container ...
#
# Then either drive it via framework/debug.sh (recommended — short-run defaults),
# or call this wrapper directly with a SIZE/RECIPE:
#
#   SIZE=270m-moe RECIPE=md_decoupling bash _research/launch/interactive-run.sh \
#       _research/launch/framework/train.sbatch --train-samples 2560 --exit-duration-in-mins 10
#
# Extra flags after the sbatch are appended AFTER MEGATRON_ARGS, so they override
# (last value wins for argparse).
#
# Env overrides:
#   NPROC_PER_NODE=2 bash ...   → fewer ranks (default 4)
#   SKIP_DEPS=1     bash ...    → skip install_python_deps.sh
#   MEGATRON_DATA_PATH=<prefix> → single --data-path instead of the blend
#
# Implementation: stubs `srun` + `scontrol`, sources the sbatch with
# COMMON_NO_LAUNCH=1 (so common.sh populates MEGATRON_ARGS and returns instead of
# launching srun), then runs torchrun with those args.

set -euo pipefail

# Repo root = parent of _research/. Computed from this script's own path so the
# wrapper works regardless of where the user invoked `srun --pty bash` from.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export SLURM_SUBMIT_DIR=$REPO_ROOT
cd "$REPO_ROOT"

# Default entrypoint is the framework train.sbatch; it needs SIZE + RECIPE in the
# env (bare use without them errors clearly — drive via framework/debug.sh).
DEFAULT_SBATCH=_research/launch/framework/train.sbatch
SBATCH=${1:-$DEFAULT_SBATCH}
[ $# -gt 0 ] && shift

if [ ! -f "$SBATCH" ]; then
    echo "sbatch not found: $SBATCH" >&2
    exit 1
fi

# Stubs so sourcing the sbatch doesn't try to launch / call SLURM commands.
srun() { :; }
scontrol() { hostname; }
export -f srun scontrol

# Tell common.sh to populate MEGATRON_ARGS/WORKDIR/PACKAGE_DIR and then `return`
# instead of running srun + the auto-requeue tail (which ends in `exit` and would
# otherwise kill this wrapper before torchrun launches).
export COMMON_NO_LAUNCH=1

# Fallback defaults for SLURM vars the sbatch/common reference directly. Inside
# an interactive allocation these are normally set already.
: "${SLURM_CPUS_PER_TASK:=72}"
: "${SLURM_NTASKS:=4}"
: "${SLURM_NNODES:=1}"
: "${SLURM_JOB_NODELIST:=$(hostname)}"
: "${SLURM_JOB_ID:=interactive}"
export SLURM_CPUS_PER_TASK SLURM_NTASKS SLURM_NNODES SLURM_JOB_NODELIST SLURM_JOB_ID

echo ">>> sourcing $SBATCH (srun stubbed, COMMON_NO_LAUNCH=1)"
source "$SBATCH"
echo

# Install deps (common.sh's TRAINING_CMD does this; we replicate it here).
if [ -z "${SKIP_DEPS:-}" ]; then
    echo ">>> installing python deps into $PACKAGE_DIR"
    bash "$WORKDIR/_research/launch/install_python_deps.sh" "$PACKAGE_DIR"
    echo
fi

# Single-node torchrun rendezvous (overrides the sbatch's SLURM-based addr).
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-25678}

NPROC=${NPROC_PER_NODE:-4}
cd "$WORKDIR"

echo ">>> WORKDIR=$WORKDIR"
echo ">>> torchrun --nproc-per-node=$NPROC --nnodes=$SLURM_NNODES pretrain_gpt.py [MEGATRON_ARGS] $*"
echo
exec torchrun --standalone --nproc-per-node="$NPROC" --nnodes="$SLURM_NNODES" \
    pretrain_gpt.py "${MEGATRON_ARGS[@]}" "$@"
