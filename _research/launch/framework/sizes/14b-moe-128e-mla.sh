# shellcheck shell=bash
#
# 14b-moe-128e-mla — MLA twin of 14b-moe-128e (Multi-Latent Attention).
# 20L / 1792H / 14 heads, first layer dense then 19 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~1.46B active / ~14.43B total. At the ~1B-active eval point. Token
# budget = 100 tok/active-param.

NUM_LAYERS=20
HIDDEN=1792
FFN_HIDDEN=4480
NUM_HEADS=14                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=7                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-512}               # ≥ MBS×DP (256 GPU) and caps the run at ≤60k steps
TRAIN_SAMPLES=${TRAIN_SAMPLES:-17836032}  # ~100 tok/active-param (active from GQA twin, ÷GBS=512)
SAVE_INTERVAL=14000

APERTUS_TRACK=14a-moe-128e-mla

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

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
EP=${EP:-4}                    # shard experts across 4 ranks (matches calibration anchor)

# No hard 12h cap (12h is a must only up to 9b). ~16h at ~9% MFU → chain with
# `submit.sh --auto-requeue`. See the GQA twin for the rationale.
DEFAULT_NODES=64
DEFAULT_TIME=12:00:00
