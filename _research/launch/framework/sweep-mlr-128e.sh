#!/bin/bash
#
# sweep-mlr-128e.sh — matrix-LR sweep for the md_decoupling (Muon) recipe on the
# smallest 128e rung, in steps of powers of sqrt(2) around a center (probable
# optimum 1e-2). MLR is the md_decoupling knob (--matrix-lr, the 2D/Muon group).
#
# Sweeps three architecture variants of the SAME 1.5b geometry by default:
#   1.5b-moe-128e            GQA
#   1.5b-moe-128e-mla        MLA
#   1.5b-moe-128e-sandwich   GQA + sandwich norm
# Each (size, MLR) is a distinct run — md_decoupling's KNOB_STR carries
# `-mlr<val>` so wandb groups them and submit.sh's --dependency=singleton keeps
# re-runs from double-writing a checkpoint dir.
#
# Grid: CENTER * sqrt(2)^k for k = -STEPS..STEPS  (default 1e-2, +/-3 -> 7 points,
# spanning ~3.5e-3 .. ~2.8e-2, an 8x range).
#
# A second axis can be swept alongside MLR with `--vary NAME=v1,v2,...`: each value
# is passed to submit.sh as the env var NAME. Combined with the parametrized
# width/depth size files (which read WIDTH / DEPTH and encode it in the run name),
# this gives the muP matrix-LR transfer sweeps:
#   width:  --sizes 1.5b-moe-128e-width --vary WIDTH=512,768,1024,1536
#   depth:  --sizes 1.5b-moe-128e-depth --vary DEPTH=6,10,14,20
#
# Usage:
#   bash sweep-mlr-128e.sh                          # 1 size x 7 MLR, submit
#   bash sweep-mlr-128e.sh --dry-run                # print the submits, launch nothing
#   bash sweep-mlr-128e.sh --sizes 1.5b-moe-128e    # a specific variant
#   CENTER=1e-2 STEPS=2 bash sweep-mlr-128e.sh      # narrower 5-point grid
#   bash sweep-mlr-128e.sh --sizes 1.5b-moe-128e-width --vary WIDTH=512,768,1024,1536  # width transfer
#   bash sweep-mlr-128e.sh --sizes 1.5b-moe-128e-depth --vary DEPTH=6,10,14,20         # depth transfer
#   bash sweep-mlr-128e.sh --reservation            # run on the apertus 1-5 reservation (no name needed)
#   RESERVATION=SD-OTHER-... bash sweep-mlr-128e.sh  # or a different reservation, by name
#   TRAIN_SAMPLES=1150000 bash sweep-mlr-128e.sh    # cheaper probe budget (env passes through)
#
set -euo pipefail
FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# --reservation (no arg) resolves to this; change here if the reservation rotates.
APERTUS_RESERVATION=SD-69241-apertus-1-5-0

CENTER=${CENTER:-1e-2}     # probable optimum; grid is centered here
STEPS=${STEPS:-3}          # +/- this many sqrt(2) steps -> 2*STEPS+1 points
RECIPE=${RECIPE:-md_decoupling}
SIZES=${SIZES:-"1.5b-moe-128e"}
RESERVATION=${RESERVATION:-}   # empty = none; --reservation sets the apertus 1-5 one, or set a name here

VARY_NAME=""; VARY_VALS=("")   # optional second axis (--vary NAME=v1,v2,...); default: none
PASS=()   # flags forwarded verbatim to submit.sh
while [ $# -gt 0 ]; do
    case "$1" in
        --center)  CENTER="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --sizes)   SIZES="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --vary)    VARY_NAME="${2%%=*}"; IFS=',' read -r -a VARY_VALS <<<"${2#*=}"; shift 2 ;;
        --reservation) RESERVATION=$APERTUS_RESERVATION; shift ;;   # bare flag: the apertus 1-5 reservation
        --dry-run|--auto-requeue) PASS+=("$1"); shift ;;
        --nodes|--time|--cluster) PASS+=("$1" "$2"); shift 2 ;;
        *) echo "unknown arg: $1 (sweep flags: --center --steps --sizes --recipe --vary --reservation; rest forwarded to submit.sh)" >&2; exit 1 ;;
    esac
done
# Forward the reservation (from the --reservation flag or a RESERVATION=... env override).
[ -n "$RESERVATION" ] && PASS+=(--reservation "$RESERVATION")

# Build the MLR grid: CENTER * 2^(k/2) for k = -STEPS..STEPS.
MLRS=()
for ((k=-STEPS; k<=STEPS; k++)); do
    MLRS+=("$(awk -v c="$CENTER" -v k="$k" 'BEGIN{printf "%.4g", c*exp((k/2.0)*log(2))}')")
done

echo ">>> MLR sweep (md_decoupling matrix-lr)"
echo ">>> center=$CENTER  steps=+/-$STEPS  grid: ${MLRS[*]}"
echo ">>> sizes: $SIZES"
[ -n "$VARY_NAME" ] && echo ">>> vary: $VARY_NAME = ${VARY_VALS[*]}"
echo ">>> recipe: $RECIPE${PASS[*]+  forwarded: ${PASS[*]}}"
echo

for size in $SIZES; do
    for vv in "${VARY_VALS[@]}"; do
        for mlr in "${MLRS[@]}"; do
            env_prefix=(MLR="$mlr")
            [ -n "$VARY_NAME" ] && env_prefix+=("$VARY_NAME=$vv")
            echo ">>> size=$size  ${VARY_NAME:+$VARY_NAME=$vv  }MLR=$mlr"
            env "${env_prefix[@]}" bash "$FRAMEWORK_DIR/submit.sh" \
                --size "$size" --recipe "$RECIPE" \
                ${PASS[@]+"${PASS[@]}"}
        done
    done
done
