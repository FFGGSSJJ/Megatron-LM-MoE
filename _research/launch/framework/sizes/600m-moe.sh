# shellcheck shell=bash
#
# 600m-moe — in-between rung of the MoE ladder (between 420m and 810m).
# 22L / 1280H / 20h / 4kv. MoE: 64 routed experts top-2 + 1 shared (half a routed
# expert's width), first layer dense then 21 MoE (~5% dense). moe_ffn = hidden/
# 1.82. ~0.58B active / ~4.10B total; ~6.4% active/total (non-embedding) — iso-
# sparsity proxy for 670B-A40B. hidden=1280=256*5=64*20 keeps head_dim at the
# ladder-invariant 64; active params sit at the geometric midpoint of 420m/810m.
# DeepSeek-V3-style routing is invariant and lives in lib/common.sh. 30B tokens,
# 4 nodes (matching the 810m scale-up budget).

NUM_LAYERS=22
HIDDEN=1280
FFN_HIDDEN=3200
NUM_HEADS=20
NUM_KV_HEADS=4
SEQ_LEN=4096

MBS=${MBS:-4}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-7324219}   # ~30B tokens (override for other budgets)
SAVE_INTERVAL=5722

APERTUS_TRACK=600a-moe

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 704
    --moe-shared-expert-intermediate-size 352
    --moe-layer-freq "([0]*1+[1]*21)"
)
EP=${EP:-1}                    # default pure DP (all 64 experts replicated per GPU)

DEFAULT_NODES=4
DEFAULT_TIME=10:00:00
