# shellcheck shell=bash
#
# 14b-moe-128e-mla — MLA twin of 14b-moe-128e (Multi-Latent Attention).
# 20L / 1792H / 14 heads, first layer dense then 19 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~1.21B active / ~14.2B total. At the ~1B-active eval point. Token
# budget = 100 tok/active-param.

NUM_LAYERS=20
HIDDEN=1792
FFN_HIDDEN=4480
NUM_HEADS=14                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=7                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-14817536}  # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=11600

APERTUS_TRACK=14a-moe-128e-mla

MLA_ARGS=(
    --q-lora-rank 384
    --kv-lora-rank 128
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 1024
    --moe-shared-expert-intermediate-size 1024
    --moe-layer-freq "([0]*1+[1]*19)"
)
EP=${EP:-1}

DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
