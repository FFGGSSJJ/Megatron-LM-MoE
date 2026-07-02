#!/bin/bash
#
# train.sbatch — the single entrypoint for every run.
#
# It does almost nothing itself: pick a SIZE, a RECIPE (and optionally a
# CLUSTER, default alps3), source them, then hand off to lib/common.sh which
# assembles the args and launches.
#
#   SIZE=420m-moe RECIPE=md_decoupling sbatch \
#       --account=infra01 --nodes=2 --time=10:00:00 \
#       _research/launch/framework/train.sbatch
#
# Prefer submit.sh, which fills --nodes/--time from the size file AND the
# per-cluster sbatch flags (account/cpus/mem/reservation) from
# clusters/<cluster>.sh for you:
#   bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling
#
# The #SBATCH lines below are cluster-independent DEFAULTS; submit.sh (or your
# own `sbatch` flags) supplies --nodes/--time/--job-name and the per-cluster
# flags. Raw `sbatch train.sbatch` users must add those themselves (e.g. the
# alps3 --account / --cpus-per-task / --mem, see clusters/alps3.sh).
#
#SBATCH --job-name=train
#SBATCH --time=10:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-apertus-moe/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-apertus-moe/_research/results/runs/%x-%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --no-requeue

set -euo pipefail

: "${SIZE:?set SIZE, e.g. SIZE=420m-moe (see framework/sizes/)}"
: "${RECIPE:?set RECIPE, e.g. RECIPE=md_decoupling (see framework/recipes/)}"
CLUSTER=${CLUSTER:-alps3}

# Locate the framework dir. Under `sbatch`, SLURM copies this script to a
# private spool dir (/var/spool/slurmd/...), so BASH_SOURCE no longer points at
# the framework. submit.sh exports FRAMEWORK_DIR; otherwise fall back to
# BASH_SOURCE (direct `bash`/interactive sourcing), then to the canonical repo
# layout on shared storage.
FRAMEWORK_DIR=${FRAMEWORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}
if [ ! -d "$FRAMEWORK_DIR/sizes" ]; then
    FRAMEWORK_DIR=/iopsstor/scratch/cscs/$USER/megatron-apertus-moe/_research/launch/framework
fi

[ -f "$FRAMEWORK_DIR/sizes/$SIZE.sh" ]     || { echo "no such size:    $SIZE   (see $FRAMEWORK_DIR/sizes/)"   >&2; exit 1; }
[ -f "$FRAMEWORK_DIR/recipes/$RECIPE.sh" ] || { echo "no such recipe:  $RECIPE (see $FRAMEWORK_DIR/recipes/)" >&2; exit 1; }
[ -f "$FRAMEWORK_DIR/clusters/$CLUSTER.sh" ] || { echo "no such cluster: $CLUSTER (see $FRAMEWORK_DIR/clusters/)" >&2; exit 1; }

# Order matters: size sets model+data scale, recipe sets the optimizer/idea
# (and may override regularization/LR/sharding), cluster sets machine knobs
# (guarded, so env/size/recipe win), common fills defaults and launches.
source "$FRAMEWORK_DIR/sizes/$SIZE.sh"
source "$FRAMEWORK_DIR/recipes/$RECIPE.sh"
source "$FRAMEWORK_DIR/clusters/$CLUSTER.sh"
source "$FRAMEWORK_DIR/lib/common.sh"
