# shellcheck shell=bash
#
# 1.5b-moe-128e-width — WIDTH-scaling family for a muP / matrix-LR transfer study.
# Fixed 10 layers (1 dense + 9 MoE) and the 1.5b base's batch + token budget; the
# ONLY thing that scales is hidden. Every width-derived shape follows the ladder's
# rules exactly, so the aspect ratio and MoE/attention proportions are held constant
# (that is what makes the optimal matrix LR expected to transfer):
#
#   head_dim = 128                -> NUM_HEADS   = hidden/128
#   GQA kv   = heads/2            -> NUM_KV_HEADS
#   dense MLP= hidden*2.5         -> FFN_HIDDEN
#   moe_ffn  = shared = round(hidden/1.75, x64)
#
# Set the width with WIDTH=... (default 768 = the 1.5b base). WIDTH must be a
# multiple of 256 so the head count is even (kv = heads/2 stays integer).
#
#   WIDTH=1024 MLR=1e-2 bash submit.sh --size 1.5b-moe-128e-width --recipe md_decoupling
#   bash sweep-mlr-128e.sh --sizes 1.5b-moe-128e-width --vary WIDTH=512,768,1024,1536

HIDDEN=${WIDTH:-${HIDDEN:-768}}
(( HIDDEN % 256 == 0 )) || { echo "WIDTH must be a multiple of 256 (even head count); got $HIDDEN" >&2; exit 1; }

NUM_LAYERS=10
NUM_HEADS=$(( HIDDEN / 128 ))          # head_dim = 128
NUM_KV_HEADS=$(( NUM_HEADS / 2 ))      # constant 2:1 GQA ratio
FFN_HIDDEN=$(( HIDDEN * 5 / 2 ))       # dense-layer MLP = hidden*2.5
MOE_FFN=$(awk -v h="$HIDDEN" 'BEGIN{printf "%d", int((h/1.75)/64 + 0.5)*64}')  # round(hidden/1.75, x64)
SEQ_LEN=8192

# Batch + length held at the 1.5b base across the whole family (muP: only width moves).
MBS=${MBS:-2}                 # measured good on 8 nodes w/ EP=4/alltoall/fp8 (~8h); max 4 (GBS>=MBS*DP)
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-4598912}   # ~37.7B tokens (base 1.5b budget; 35,929 iters)
SAVE_INTERVAL=3600            # ~10 saves over the run

APERTUS_TRACK=1.5a-moe-128e-width

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size "$MOE_FFN"
    --moe-shared-expert-intermediate-size "$MOE_FFN"
    --moe-layer-freq "([0]*1+[1]*9)"
)
EP=${EP:-4}                    # shard 128 experts 4-ways over intra-node NVLink (throughput);
                               # requires DP % 4 == 0; try MOE_DISPATCHER=alltoall if it doesn't pay off.

# Optional sandwich (post-)norm via SANDWICH=1: adds --sandwich-norm (its alpha comes
# from the ladder's --residual-output-scaling, already on) and tags the run so it stays
# distinct from the plain-norm width twin. See 1.5b-moe-128e-sandwich.sh for the details.
EXTRA_NETWORK_ARGS=()
_SW_TAG=""
if [ "${SANDWICH:-0}" = 1 ]; then
    EXTRA_NETWORK_ARGS=(--sandwich-norm)
    _SW_TAG="-sandwich"
fi
# Surface width (+ sandwich) in the run/wandb name so each point is a distinct run.
ARCH_KNOB_STR="w${HIDDEN}${_SW_TAG}"

# Node count fixed for the family; wider points run longer at the same node count
# (node-hours are constant — pass --nodes N / --auto-requeue to trade wall-clock).
DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
