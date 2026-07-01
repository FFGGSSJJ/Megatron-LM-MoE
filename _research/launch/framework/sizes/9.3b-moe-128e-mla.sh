# shellcheck shell=bash
#
# 9.3b-moe-128e-mla — MLA twin of 9.3b-moe-128e (Multi-Latent Attention).
# 17L / 1536H / 12 heads, first layer dense then 16 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~1.09B active / ~9.28B total. Near the ~1B-active eval point. Token
# budget = 100 tok/active-param.

NUM_LAYERS=17
HIDDEN=1536
FFN_HIDDEN=3840
NUM_HEADS=12                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=6                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-13257472}  # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=10400

APERTUS_TRACK=9.3a-moe-128e-mla

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

MLA_ARGS=(
    --q-lora-rank 288
    --kv-lora-rank 96
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 896
    --moe-shared-expert-intermediate-size 896
    --moe-layer-freq "([0]*1+[1]*16)"
)
EP=${EP:-1}

DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
