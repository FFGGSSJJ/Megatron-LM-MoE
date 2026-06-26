#!/bin/bash
#
# submit.sh — convenience wrapper around `sbatch train.sbatch`.
#
# Reads DEFAULT_NODES / DEFAULT_TIME from the size file and the per-cluster
# sbatch flags (partition/cpus/mem/reservation) from clusters/<cluster>.sh so
# you don't have to remember them, then submits. Any knob the recipe reads from
# the environment (LR, MLR, ...) is passed straight through.
#
#   bash submit.sh --size 420m-moe --recipe master
#   bash submit.sh --size 420m-moe --recipe master --nodes 4
#   bash submit.sh --cluster mi300 --size 420m-moe --recipe master   # AMD MI300A
#   MLR=1e-2 bash submit.sh --size 420m-moe --recipe master       # sweep a knob
#   bash submit.sh ... --reservation SD-69241-apertus-1-5-0       # slurm reservation
#   RESERVATION=SD-69241-... bash sweep-scaling-laws.sh ...       # env form: all sweep
#                                                # scripts inherit it via submit.sh
#   bash submit.sh --size 1.3b --recipe master --auto-requeue     # chain jobs until done
#   bash submit.sh --size 420m-moe --recipe master --dry-run      # print args, no submit
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

NODES=""; TIME=""; DRY=""; AUTO_RQ=""; CLUSTER=${CLUSTER:-alps3}
RESERVATION=${RESERVATION:-}
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --nodes)   NODES="$2"; shift 2 ;;
        --time)    TIME="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        --auto-requeue) AUTO_RQ=1; shift ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

: "${SIZE:?usage: submit.sh --size <size> --recipe <recipe> [--cluster C] [--nodes N] [--time T] [--reservation R] [--auto-requeue] [--dry-run]}"
: "${RECIPE:?usage: submit.sh --size <size> --recipe <recipe> [--cluster C] [--nodes N] [--time T] [--reservation R] [--auto-requeue] [--dry-run]}"

SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }
CLUSTER_FILE="$FRAMEWORK_DIR/clusters/$CLUSTER.sh"
[ -f "$CLUSTER_FILE" ] || { echo "no such cluster: $CLUSTER (see $FRAMEWORK_DIR/clusters/)" >&2; exit 1; }

# Pull CLUSTER_SBATCH_FLAGS (+ CLUSTER_TAG for the job name) — the fragment is
# pure guarded assignments, safe to source on a login node.
source "$CLUSTER_FILE"

# Pull DEFAULT_NODES / DEFAULT_TIME out of the size file without running the
# whole thing.
DEFAULT_NODES=$(grep -E '^DEFAULT_NODES=' "$SIZE_FILE" | head -1 | cut -d= -f2)
DEFAULT_TIME=$(grep -E '^DEFAULT_TIME=' "$SIZE_FILE" | head -1 | cut -d= -f2)
NODES=${NODES:-${DEFAULT_NODES:-2}}
TIME=${TIME:-${DEFAULT_TIME:-10:00:00}}

export SIZE RECIPE CLUSTER

if [ -n "$DRY" ]; then
    echo ">>> DRY RUN: SIZE=$SIZE RECIPE=$RECIPE CLUSTER=$CLUSTER NODES=$NODES TIME=$TIME"
    echo ">>> composed MEGATRON_ARGS:"
    SIZE="$SIZE" RECIPE="$RECIPE" CLUSTER="$CLUSTER" bash "$FRAMEWORK_DIR/lib/dump-args.sh" | sed 's/^/  /'
    exit 0
fi

# Job name mirrors the run name (size-tag-knobs[-cluster][-runtag]) so that
# (a) sweep points are distinguishable in squeue, and (b) --dependency=singleton
# below serializes resubmits of the SAME point without blocking other points.
# EXP_TAG/KNOB_STR live in the recipe; source it in a subshell to read them
# (recipes are pure guarded assignments). Falls back to the recipe name if the
# recipe can't be sourced standalone.
NAME_PART=$( { source "$SIZE_FILE"; source "$FRAMEWORK_DIR/recipes/$RECIPE.sh"; } >/dev/null 2>&1; echo "${EXP_TAG:-${RECIPE//\//-}}${KNOB_STR:+-${KNOB_STR}}${ARCH_KNOB_STR:+-${ARCH_KNOB_STR}}" ) \
    || NAME_PART=${RECIPE//\//-}
JOB_NAME=${JOB_NAME:-${SIZE}-${NAME_PART}${CLUSTER_TAG:-}${RUN_TAG:+-${RUN_TAG}}}

# --auto-requeue: chain dependent jobs (same CKPT_DIR -> resume) until the run
# reaches TRAIN_SAMPLES. REQUEUE_TIME carries this run's --time so each chained
# allocation gets the same wall-clock; the chaining logic lives in common.sh.
EXPORT="ALL,SIZE=$SIZE,RECIPE=$RECIPE,CLUSTER=$CLUSTER,FRAMEWORK_DIR=$FRAMEWORK_DIR"
if [ -n "$AUTO_RQ" ]; then
    EXPORT="$EXPORT,AUTO_REQUEUE=1,REQUEUE_TIME=$TIME"
    echo ">>> auto-requeue ON: will chain dependent jobs until TRAIN_SAMPLES is reached (clean exits only)"
fi
# RESERVATION travels in the export list so auto-requeue re-applies it (the
# sbatch flag itself doesn't survive into chained jobs).
[ -n "$RESERVATION" ] && EXPORT="$EXPORT,RESERVATION=$RESERVATION"

echo ">>> sbatch --nodes=$NODES --time=$TIME --job-name=$JOB_NAME ${CLUSTER_SBATCH_FLAGS[@]+${CLUSTER_SBATCH_FLAGS[*]}}${RESERVATION:+ --reservation=$RESERVATION} (SIZE=$SIZE RECIPE=$RECIPE CLUSTER=$CLUSTER)"
# --dependency=singleton: at most one job with this (knob-unique) name runs at
# a time, so rerunning a sweep/submit while a same-point job is still queued or
# running QUEUES the new one instead of double-writing the checkpoint dir.
exec sbatch \
    --nodes="$NODES" \
    --time="$TIME" \
    --job-name="$JOB_NAME" \
    --dependency=singleton \
    ${CLUSTER_SBATCH_FLAGS[@]+"${CLUSTER_SBATCH_FLAGS[@]}"} \
    ${RESERVATION:+--reservation="$RESERVATION"} \
    --export="$EXPORT" \
    "$FRAMEWORK_DIR/train.sbatch"
