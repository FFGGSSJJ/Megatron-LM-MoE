# shellcheck shell=bash
#
# 810m-moe — second scale-up rung of the MoE ladder.
# 24L / 1536H / 24h / 8kv. MoE: 64 routed experts top-2 + 1 shared (half a routed
# expert's width), first layer dense then 23 MoE (~5% dense). moe_ffn = hidden/
# 1.75. ~0.81B active / ~6.70B total; ~6.5% active/total (non-embedding) — iso-
# sparsity proxy for 670B-A40B. DeepSeek-V3-style routing is invariant and lives
# in lib/common.sh. Leaderboard ladder budget: 30B tokens, 4 nodes.

NUM_LAYERS=24
HIDDEN=1536
FFN_HIDDEN=4096
NUM_HEADS=24
NUM_KV_HEADS=8
SEQ_LEN=4096

MBS=${MBS:-4}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-7324219}   # ~30B tokens (leaderboard 810m budget)
SAVE_INTERVAL=5722

APERTUS_TRACK=810a-moe

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 896
    --moe-shared-expert-intermediate-size 448
    --moe-layer-freq "([0]*1+[1]*23)"
)
EP=${EP:-1}                    # default pure DP (all 64 experts replicated per GPU);
                               # env EP>1 shards experts (see sweep-scaling-laws-810m-ep4.sh)

DEFAULT_NODES=4
DEFAULT_TIME=10:00:00
