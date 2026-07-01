# shellcheck shell=bash
#
# 1.5b-moe-128e-depth — DEPTH-scaling family for a muP / matrix-LR transfer study.
# Fixed 768 hidden (and all its width-derived shapes) and the 1.5b base's batch +
# token budget; the ONLY thing that scales is the number of layers. Layer 0 stays
# dense, the rest are MoE, so depth = 1 dense + (DEPTH-1) MoE.
#
# The 1/sqrt(2L) residual multiplier (--residual-output-scaling, on in common.sh)
# is computed from num_layers, so it re-scales automatically per depth — that is the
# mechanism the optimal matrix LR is expected to transfer across.
#
# Set the depth with DEPTH=... (default 10 = the 1.5b base); DEPTH >= 2.
#
#   DEPTH=20 MLR=1e-2 bash submit.sh --size 1.5b-moe-128e-depth --recipe md_decoupling
#   bash sweep-mlr-128e.sh --sizes 1.5b-moe-128e-depth --vary DEPTH=6,10,14,20

NUM_LAYERS=${DEPTH:-${NUM_LAYERS:-10}}
(( NUM_LAYERS >= 2 )) || { echo "DEPTH must be >= 2 (1 dense + >=1 MoE); got $NUM_LAYERS" >&2; exit 1; }

HIDDEN=768
NUM_HEADS=6                    # head_dim = hidden/heads = 128
NUM_KV_HEADS=3                 # heads/2
FFN_HIDDEN=1920               # dense-layer MLP = hidden*2.5
SEQ_LEN=8192

# Batch + length held at the 1.5b base across the whole family (muP: only depth moves).
MBS=${MBS:-2}                 # measured good on 8 nodes w/ EP=4/alltoall/fp8 (~8h); max 4 (GBS>=MBS*DP)
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-4598912}   # ~37.7B tokens (base 1.5b budget; 35,929 iters)
SAVE_INTERVAL=3600            # ~10 saves over the run

APERTUS_TRACK=1.5a-moe-128e-depth

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh. 1 dense layer then MoE.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 448      # round(768/1.75, x64), base 1.5b value
    --moe-shared-expert-intermediate-size 448
    --moe-layer-freq "([0]*1+[1]*$((NUM_LAYERS - 1)))"
)
EP=${EP:-4}                    # shard 128 experts 4-ways over intra-node NVLink (throughput);
                               # requires DP % 4 == 0; try MOE_DISPATCHER=alltoall if it doesn't pay off.

# Surface the depth in the run/wandb name so each depth point is a distinct run.
ARCH_KNOB_STR="L${NUM_LAYERS}"

# Node count fixed for the family; deeper points run longer at the same node count
# (node-hours are constant — pass --nodes N / --auto-requeue to trade wall-clock).
DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
