# shellcheck shell=bash
#
# 120b-moe-128e-mla — MLA twin of 120b-moe-128e (Multi-Latent Attention); the
# gpt-oss-sized top anchor with target-matched attention.
# 42L / 3584H / 28 heads, first layer dense then 41 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target (qk 128 / rope 64 / v 128);
# lora ranks scaled at target ratios (kv-lora = hidden/14 = 256, q-lora = 3×kv).
# ~7.19B active / ~119B total. Token budget = 100 tok/active-param (~719B tokens).

NUM_LAYERS=42
HIDDEN=3584
FFN_HIDDEN=8960
NUM_HEADS=28                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=14                # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-87735168}  # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=68500

APERTUS_TRACK=120a-moe-128e-mla

MLA_ARGS=(
    --q-lora-rank 768
    --kv-lora-rank 256
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-layer-freq "([0]*1+[1]*41)"
)
EP=${EP:-1}                    # CANNOT run pure DP at this scale — set EP>1 (and likely TP/PP)

DEFAULT_NODES=64
DEFAULT_TIME=12:00:00
