# shellcheck shell=bash
#
# 22b-moe-128e — top rung of the fine-grained (128e) MoE ladder (GQA); the
# largest in-ladder stand-in for the in-house 670B-A40B target.
# Iso-architecture proxy (128e top-4 + 1 shared, moe_ffn = hidden/1.75). head_dim
# 128 (kernel-friendly, = target). 24L / 2048H / 16h / 8kv, first layer dense
# then 23 MoE. ~1.97B active / ~22.16B total; 5.41% non-embed sparsity. Token
# budget = 100 tokens / active param.

NUM_LAYERS=24
HIDDEN=2048
FFN_HIDDEN=5120               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=16                   # head_dim = hidden/heads = 128
NUM_KV_HEADS=8                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-1}
GBS=${GBS:-768}               # ≥ MBS×DP (384 GPU) and caps the run at ≤60k steps
TRAIN_SAMPLES=${TRAIN_SAMPLES:-24082944}  # ~197.3B tokens = 100 tok/active-param (÷GBS=768 → 31,358 iters)
SAVE_INTERVAL=18800           # ~10 saves over the run

APERTUS_TRACK=22a-moe-128e

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 1152
    --moe-shared-expert-intermediate-size 1152
    --moe-layer-freq "([0]*1+[1]*23)"
)
EP=${EP:-4}                    # shard experts across 4 ranks; EP1 (~44GB params+grads)
                               # will NOT fit 96GB — EP is mandatory here

# No hard 12h cap here (12h is a must only up to 9b). At ~9% MFU this needs ~19h →
# runs across ≥2 allocations; launch with `submit.sh --auto-requeue` to chain
# until TRAIN_SAMPLES. Fewer/more nodes trade wall-clock, not GPU-hours.
DEFAULT_NODES=96
DEFAULT_TIME=12:00:00
