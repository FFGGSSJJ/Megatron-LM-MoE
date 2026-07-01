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
# Usage:
#   bash sweep-mlr-128e.sh                          # full sweep (3 sizes x 7 MLR), submit
#   bash sweep-mlr-128e.sh --dry-run                # print the submits, launch nothing
#   bash sweep-mlr-128e.sh --sizes 1.5b-moe-128e    # one variant only
#   CENTER=1e-2 STEPS=2 bash sweep-mlr-128e.sh      # narrower 5-point grid
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

PASS=()   # flags forwarded verbatim to submit.sh
while [ $# -gt 0 ]; do
    case "$1" in
        --center)  CENTER="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --sizes)   SIZES="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --reservation) RESERVATION=$APERTUS_RESERVATION; shift ;;   # bare flag: the apertus 1-5 reservation
        --dry-run|--auto-requeue) PASS+=("$1"); shift ;;
        --nodes|--time|--cluster) PASS+=("$1" "$2"); shift 2 ;;
        *) echo "unknown arg: $1 (sweep flags: --center --steps --sizes --recipe --reservation; rest forwarded to submit.sh)" >&2; exit 1 ;;
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
echo ">>> recipe: $RECIPE${PASS[*]+  forwarded: ${PASS[*]}}"
echo

for size in $SIZES; do
    for mlr in "${MLRS[@]}"; do
        echo ">>> size=$size  MLR=$mlr"
        MLR="$mlr" bash "$FRAMEWORK_DIR/submit.sh" \
            --size "$size" --recipe "$RECIPE" \
            ${PASS[@]+"${PASS[@]}"}
    done
done
