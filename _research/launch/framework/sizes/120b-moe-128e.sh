# shellcheck shell=bash
#
# 120b-moe-128e — gpt-oss-sized top anchor of the fine-grained (128e) MoE ladder
# (GQA); the largest in-family stand-in for the in-house 670B-A40B target.
# Same ladder rules (128e top-4 + 1 shared, moe_ffn = hidden/1.75, head_dim 128,
# 1 dense layer). 42L / 3584H / 28h / 14kv, first layer dense then 41 MoE.
# ~7.19B active / ~119.1B total; 5.29% non-embed sparsity. Token budget = 100
# tok/active param (~719B tokens — a large run; this rung is mainly a config
# reference / extrapolation anchor). Routing policy invariant in lib/common.sh.
#
# Sized like gpt-oss-120b (~117B total / 5.1B active) but in THIS ladder's family
# (gpt-oss differs: moe_ffn=hidden, no shared expert, head_dim 64).

NUM_LAYERS=42
HIDDEN=3584
FFN_HIDDEN=8960               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=28                   # head_dim = hidden/heads = 128
NUM_KV_HEADS=14                # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-87735168}  # ~719B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=68500           # ~10 saves over the run

APERTUS_TRACK=120a-moe-128e

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-layer-freq "([0]*1+[1]*41)"
)
EP=${EP:-1}                    # CANNOT run pure DP at this scale — set EP>1 (and
                               # likely TP/PP) to fit; 128 experts replicated won't fit a GPU

DEFAULT_NODES=64
DEFAULT_TIME=12:00:00
