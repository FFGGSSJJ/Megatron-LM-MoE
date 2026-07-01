# shellcheck shell=bash
#
# 1.4b-moe-128e — smallest rung of the fine-grained (128e) MoE ladder (GQA);
# the cheap LR/architecture-probe point.
# Iso-architecture proxy for the in-house 670B-A40B target (128e top-4 + 1
# shared, moe_ffn = hidden/1.75). head_dim 128 (kernel-friendly, = target).
# 10L / 768H / 6h / 3kv, first layer dense then 9 MoE. ~0.27B active / ~1.42B
# total; 5.69% non-embed sparsity. Token budget = 100 tokens / active param.
# Routing policy is invariant and lives in lib/common.sh.

NUM_LAYERS=10
HIDDEN=768
FFN_HIDDEN=1920               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=6                    # head_dim = hidden/heads = 128
NUM_KV_HEADS=3                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3305344}   # ~27.1B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=2600            # ~10 saves over the run

APERTUS_TRACK=1.4a-moe-128e

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 448
    --moe-shared-expert-intermediate-size 448
    --moe-layer-freq "([0]*1+[1]*9)"
)
EP=${EP:-1}                    # default pure DP (128 experts replicated per GPU)

DEFAULT_NODES=2
DEFAULT_TIME=12:00:00
