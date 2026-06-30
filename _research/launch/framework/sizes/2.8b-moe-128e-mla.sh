# shellcheck shell=bash
#
# 2.8b-moe-128e-mla — MLA twin of 2.8b-moe-128e (Multi-Latent Attention).
# 12L / 1024H / 8 heads, first layer dense then 11 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~0.41B active / ~2.8B total. Token budget = 100 tok/active-param.

NUM_LAYERS=12
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=8                    # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=4                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-4}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-5039232}  # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=3900

APERTUS_TRACK=2.8a-moe-128e-mla

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
    --moe-ffn-hidden-size 576
    --moe-shared-expert-intermediate-size 576
    --moe-layer-freq "([0]*1+[1]*11)"
)
EP=${EP:-1}

DEFAULT_NODES=4
DEFAULT_TIME=12:00:00
