# shellcheck shell=bash
#
# 3b-moe-128e-mla — MLA twin of 3b-moe-128e (Multi-Latent Attention).
# 12L / 1024H / 8 heads, first layer dense then 11 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~0.55B active / ~2.97B total. Token budget = 100 tok/active-param.

NUM_LAYERS=12
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=8                    # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=4                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-6764032}  # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=5300

APERTUS_TRACK=3a-moe-128e-mla

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
    --moe-ffn-hidden-size 576
    --moe-shared-expert-intermediate-size 576
    --moe-layer-freq "([0]*1+[1]*11)"
)
EP=${EP:-4}                    # shard 128 experts 4-ways (alltoall auto) for ~9% MFU; EP1 ~2x slower

# Sized to finish in one ~9h allocation (<12h); see the GQA twin for the rationale.
DEFAULT_NODES=16
DEFAULT_TIME=12:00:00
