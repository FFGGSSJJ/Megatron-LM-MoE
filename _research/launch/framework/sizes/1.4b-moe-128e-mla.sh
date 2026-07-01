# shellcheck shell=bash
#
# 1.4b-moe-128e-mla — MLA twin of 1.4b-moe-128e (Multi-Latent Attention).
# 10L / 768H / 6 heads, first layer dense then 9 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target (qk 128 / rope 64 / v 128);
# lora ranks scaled at target ratios (kv ≈ hidden/14, q = 3×kv). ~0.27B active /
# ~1.4B total. Token budget = 100 tok/active-param. Routing policy in common.sh.

NUM_LAYERS=10
HIDDEN=768
FFN_HIDDEN=1920
NUM_HEADS=6                    # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=3                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-4}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3305344}   # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=2600

APERTUS_TRACK=1.4a-moe-128e-mla

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

MLA_ARGS=(
    --q-lora-rank 192
    --kv-lora-rank 64
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 448
    --moe-shared-expert-intermediate-size 448
    --moe-layer-freq "([0]*1+[1]*9)"
)
EP=${EP:-1}

DEFAULT_NODES=2
DEFAULT_TIME=12:00:00
