# shellcheck shell=bash
#
# 14b-moe-128e — fifth rung of the fine-grained (128e) MoE ladder (GQA).
# Iso-architecture proxy for the in-house 670B-A40B target (128e top-4 + 1
# shared, moe_ffn = hidden/1.75). head_dim 128 (kernel-friendly, = target).
# 20L / 1792H / 14h / 7kv, first layer dense then 19 MoE. ~1.21B active / ~14.18B
# total; 5.43% non-embed sparsity. At the ~1B-active eval point. Token budget =
# 100 tokens / active param.

NUM_LAYERS=20
HIDDEN=1792
FFN_HIDDEN=4480               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=14                   # head_dim = hidden/heads = 128
NUM_KV_HEADS=7                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-14817536}  # ~121.4B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=11600           # ~10 saves over the run

APERTUS_TRACK=14a-moe-128e

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 1024
    --moe-shared-expert-intermediate-size 1024
    --moe-layer-freq "([0]*1+[1]*19)"
)
EP=${EP:-1}                    # default pure DP; env EP>1 shards experts (likely needed)

DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
