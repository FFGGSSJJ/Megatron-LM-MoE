# shellcheck shell=bash
#
# 120b-moe-128e-mla — MLA twin of 120b-moe-128e (Multi-Latent Attention); the
# gpt-oss-sized top anchor with target-matched attention.
# 42L / 3584H / 28 heads, first layer dense then 41 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target (qk 128 / rope 64 / v 128);
# lora ranks scaled at target ratios (kv-lora = hidden/14 = 256, q-lora = 3×kv).
# ~7.68B active / ~119.6B total. Token budget = 100 tok/active-param (~768B tokens).

NUM_LAYERS=42
HIDDEN=3584
FFN_HIDDEN=8960
NUM_HEADS=28                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=14                # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-2048}             # ≥ MBS×DP (512 GPU) and caps the run at ≤60k steps
TRAIN_SAMPLES=${TRAIN_SAMPLES:-93771776}  # ~100 tok/active-param (active from GQA twin, ÷GBS=2048)
SAVE_INTERVAL=73000

APERTUS_TRACK=120a-moe-128e-mla

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

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
EP=${EP:-8}                    # CANNOT run pure DP; EP8 shards experts, may still need TP/PP

# 128 nodes is the stated upper bound for this anchor. ~9 days at ~9% MFU → many
# chained allocations (`submit.sh --auto-requeue`), or cut tokens / raise MFU.
DEFAULT_NODES=128
DEFAULT_TIME=12:00:00
