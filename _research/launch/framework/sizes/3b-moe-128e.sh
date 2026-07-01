# shellcheck shell=bash
#
# 3b-moe-128e — second rung of the fine-grained (128e) MoE ladder (GQA).
# Iso-architecture proxy for the in-house 670B-A40B target (128e top-4 + 1
# shared, moe_ffn = hidden/1.75). head_dim 128 (kernel-friendly, = target).
# 12L / 1024H / 8h / 4kv, first layer dense then 11 MoE. ~0.55B active / ~2.97B
# total; 5.64% non-embed sparsity. Token budget = 100 tokens / active param.

NUM_LAYERS=12
HIDDEN=1024
FFN_HIDDEN=2560               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=8                    # head_dim = hidden/heads = 128
NUM_KV_HEADS=4                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-6764032}  # ~55.4B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=5300            # ~10 saves over the run

APERTUS_TRACK=3a-moe-128e

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 576
    --moe-shared-expert-intermediate-size 576
    --moe-layer-freq "([0]*1+[1]*11)"
)
EP=${EP:-1}                    # default pure DP

# Sized to finish TRAIN_SAMPLES in one ~9h allocation (<12h). ~9% MFU calibration
# (7b/810m-active/EP4 anchor: 32 nodes for 100B tok in 12h → GPU-hours ∝ active²).
DEFAULT_NODES=16
DEFAULT_TIME=12:00:00
