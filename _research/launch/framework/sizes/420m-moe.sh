# shellcheck shell=bash
#
# 420m-moe — THE MAIN BASELINE.
#
# Transformer++ MoE, ~0.41B active / ~2.50B total. 20L / 1024H / 16h / 4kv (GQA).
# Sparse MoE: 64 routed experts top-2 + 1 shared (half a routed expert's width),
# first layer dense then 19 MoE (~5% dense). moe_ffn = hidden/1.75. ~6.5%
# active/total (non-embedding) — an iso-sparsity proxy for the 670B-A40B target.
# DeepSeek-V3-style routing (sigmoid + aux-loss-free bias + ...) is invariant and
# lives in lib/common.sh; this file sets only the per-scale expert geometry below.
# 15B tokens: 3,662,109 samples * 4096 seq = 15.0B, ~29k steps. FP8 blockwise.
#
# Node count is just a data-parallel layout knob — GBS and TRAIN_SAMPLES are
# fixed, so 2 vs 4 nodes only changes wall-clock, not the run. submit.sh
# defaults to DEFAULT_NODES; override with --nodes (pure DP: any node count).

NUM_LAYERS=20
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=16
NUM_KV_HEADS=4
SEQ_LEN=4096

MBS=${MBS:-8}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3662109}   # 15.0B tokens (overridable for length sweeps)
SAVE_INTERVAL=5722             # ~10 saves over the run

EMBEDDING_MULTIPLIER=32.0      # sqrt(1024)

APERTUS_TRACK=420a-moe

# Expert geometry only — routing policy is in common.sh. 64 experts top-2 + 1
# shared at half a routed expert's width; first layer dense, the other 19 MoE.
MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 576
    --moe-shared-expert-intermediate-size 288
    --moe-layer-freq "([0]*1+[1]*19)"
)
EP=1                           # pure DP: all 64 experts replicated on each GPU

# Defaults read by submit.sh (overridable on the CLI).
DEFAULT_NODES=2
DEFAULT_TIME=10:00:00
