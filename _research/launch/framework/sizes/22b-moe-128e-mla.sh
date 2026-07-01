# shellcheck shell=bash
#
# 22b-moe-128e-mla — MLA twin of 22b-moe-128e (Multi-Latent Attention); largest
# in-ladder stand-in for the 670B-A40B target with target-matched attention.
# 24L / 2048H / 16 heads, first layer dense then 23 MoE. 128e top-4 + 1 shared,
# moe_ffn = hidden/1.75. MLA per-head dims = target; lora ranks scaled at target
# ratios. ~1.97B active / ~22.16B total. Token budget = 100 tok/active-param.

NUM_LAYERS=24
HIDDEN=2048
FFN_HIDDEN=5120
NUM_HEADS=16                   # attention-out = heads*v_head_dim = hidden
NUM_KV_HEADS=8                 # unused under MLA
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-24083200}  # ~100 tok/active-param (active from GQA twin, ÷GBS)
SAVE_INTERVAL=18800

APERTUS_TRACK=22a-moe-128e-mla

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

MLA_ARGS=(
    --q-lora-rank 480
    --kv-lora-rank 160
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 1152
    --moe-shared-expert-intermediate-size 1152
    --moe-layer-freq "([0]*1+[1]*23)"
)
EP=${EP:-1}

DEFAULT_NODES=16
DEFAULT_TIME=12:00:00
