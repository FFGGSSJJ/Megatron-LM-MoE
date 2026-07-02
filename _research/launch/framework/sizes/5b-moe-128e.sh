# shellcheck shell=bash
#
# 5b-moe-128e — third rung of the fine-grained (128e) MoE ladder (GQA).
# Iso-architecture proxy for the in-house 670B-A40B target (128e top-4 + 1
# shared, moe_ffn = hidden/1.75). head_dim 128 (kernel-friendly, = target).
# 14L / 1280H / 10h / 5kv, first layer dense then 13 MoE. ~0.77B active / ~5.13B
# total; 5.61% non-embed sparsity. Token budget = 100 tokens / active param.

NUM_LAYERS=14
HIDDEN=1280
FFN_HIDDEN=3200               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=10                   # head_dim = hidden/heads = 128
NUM_KV_HEADS=5                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-2}
GBS=${GBS:-256}                # ≥ MBS×DP (128 GPU) and caps the run at ≤60k steps
TRAIN_SAMPLES=${TRAIN_SAMPLES:-9413120}  # ~77.1B tokens = 100 tok/active-param (÷GBS=256 → 36,770 iters)
SAVE_INTERVAL=7400           # ~10 saves over the run

APERTUS_TRACK=5a-moe-128e

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 704
    --moe-shared-expert-intermediate-size 704
    --moe-layer-freq "([0]*1+[1]*13)"
)
EP=${EP:-4}                    # shard 128 experts 4-ways (alltoall auto) for ~9% MFU / on-target
                               # wall-clock; EP1 runs this fine-grained MoE at ~4% (~2x slower)

# Sized to finish TRAIN_SAMPLES in one ~9h allocation (<12h). ~9% MFU calibration
# (7b/810m-active/EP4 anchor: 32 nodes for 100B tok in 12h → GPU-hours ∝ active²).
DEFAULT_NODES=32
DEFAULT_TIME=12:00:00
