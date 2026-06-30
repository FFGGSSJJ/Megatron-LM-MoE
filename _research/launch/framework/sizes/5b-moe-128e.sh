# shellcheck shell=bash
#
# 5b-moe-128e — third rung of the fine-grained (128e) MoE ladder (GQA).
# Iso-architecture proxy for the in-house 670B-A40B target (128e top-4 + 1
# shared, moe_ffn = hidden/1.75). head_dim 128 (kernel-friendly, = target).
# 14L / 1280H / 10h / 5kv, first layer dense then 13 MoE. ~0.59B active / ~4.95B
# total; 5.61% non-embed sparsity. Token budget = 100 tokens / active param.

NUM_LAYERS=14
HIDDEN=1280
FFN_HIDDEN=3200               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=10                   # head_dim = hidden/heads = 128
NUM_KV_HEADS=5                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-7256960}  # ~59.5B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=5700           # ~10 saves over the run

APERTUS_TRACK=5a-moe-128e

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 704
    --moe-shared-expert-intermediate-size 704
    --moe-layer-freq "([0]*1+[1]*13)"
)
EP=${EP:-1}                    # default pure DP

DEFAULT_NODES=4
DEFAULT_TIME=12:00:00
