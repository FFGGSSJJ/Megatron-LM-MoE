# shellcheck shell=bash
#
# 5b-moe-128e-mla — MLA twin of 5b-moe-128e (Multi-Latent Attention).
# 14L / 1280H / 10 heads, first layer dense then 13 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~0.77B active / ~5.13B total. Token budget = 100 tok/active-param.

NUM_LAYERS=14
HIDDEN=1280
FFN_HIDDEN=3200
NUM_HEADS=10                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=5                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-256}               # ≥ MBS×DP (128 GPU) and caps the run at ≤60k steps
TRAIN_SAMPLES=${TRAIN_SAMPLES:-9413120}  # ~100 tok/active-param (active from GQA twin, ÷GBS=256)
SAVE_INTERVAL=7400

APERTUS_TRACK=5a-moe-128e-mla

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
    --moe-ffn-hidden-size 704
    --moe-shared-expert-intermediate-size 704
    --moe-layer-freq "([0]*1+[1]*13)"
)
EP=${EP:-4}                    # shard 128 experts 4-ways (alltoall auto) for ~9% MFU; EP1 ~2x slower

# Sized to finish in one ~9h allocation (<12h); see the GQA twin for the rationale.
DEFAULT_NODES=32
DEFAULT_TIME=12:00:00
