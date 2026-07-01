# shellcheck shell=bash
#
# 9.3b-moe-128e — fourth rung of the fine-grained (128e) MoE ladder (GQA).
# Iso-architecture proxy for the in-house 670B-A40B target (128e top-4 + 1
# shared, moe_ffn = hidden/1.75). head_dim 128 (kernel-friendly, = target).
# 17L / 1536H / 12h / 6kv, first layer dense then 16 MoE. ~1.09B active / ~9.28B
# total; 5.44% non-embed sparsity. Near the ~1B-active eval point. Token budget
# = 100 tokens / active param.

NUM_LAYERS=17
HIDDEN=1536
FFN_HIDDEN=3840               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=12                   # head_dim = hidden/heads = 128
NUM_KV_HEADS=6                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-13257472}  # ~108.6B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=10400           # ~10 saves over the run

APERTUS_TRACK=9.3a-moe-128e

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 896
    --moe-shared-expert-intermediate-size 896
    --moe-layer-freq "([0]*1+[1]*16)"
)
EP=${EP:-1}                    # default pure DP; env EP>1 shards experts if memory tight

DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
