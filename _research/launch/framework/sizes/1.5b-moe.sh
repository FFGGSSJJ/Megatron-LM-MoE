# shellcheck shell=bash
#
# 1.5b-moe — top scale-up rung of the MoE ladder.
# 32L / 2048H / 32h / 8kv. MoE: 64 routed experts top-2 + 1 shared (half a routed
# expert's width), first 2 layers dense then 30 MoE (~5% dense). moe_ffn =
# hidden/1.75. ~1.47B active / ~14.6B total; ~6.6% active/total (non-embedding) —
# iso-sparsity proxy for 670B-A40B. The confirmation rung for promoting a 420m
# winner. DeepSeek-V3-style routing is invariant and lives in lib/common.sh.

NUM_LAYERS=32
HIDDEN=2048
FFN_HIDDEN=5120
NUM_HEADS=32
NUM_KV_HEADS=8
SEQ_LEN=4096

MBS=${MBS:-4}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-7324219}   # ~30B tokens (override for other budgets)
SAVE_INTERVAL=5722

APERTUS_TRACK=transformer-pp-1.5b

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 1152
    --moe-shared-expert-intermediate-size 576
    --moe-layer-freq "([0]*2+[1]*30)"
)
EP=1                           # pure DP: all 64 experts replicated on each GPU

DEFAULT_NODES=4
DEFAULT_TIME=12:00:00
