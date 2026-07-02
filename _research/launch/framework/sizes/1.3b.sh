# shellcheck shell=bash
#
# 1.3b — DENSE baseline (no MoE). 24L / 2048H / 32h / 8kv.
# Leaderboard ladder budget: 100B tokens, 8 nodes. Pure data parallel
# (TP=PP=1) with the distributed optimizer.

NUM_LAYERS=24
HIDDEN=2048
FFN_HIDDEN=5632
NUM_HEADS=32
NUM_KV_HEADS=8
SEQ_LEN=4096

MBS=${MBS:-2}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-24414063}  # ~100B tokens (leaderboard 1.3b budget)
SAVE_INTERVAL=19073

APERTUS_TRACK=transformer-pp-1.3b

MOE_ARGS=()                                # dense

DEFAULT_NODES=8
DEFAULT_TIME=24:00:00
