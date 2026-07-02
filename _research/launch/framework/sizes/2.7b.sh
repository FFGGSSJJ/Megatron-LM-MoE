# shellcheck shell=bash
#
# 2.7b — DENSE baseline (no MoE). 32L / 2560H / 32h / 8kv.
# Leaderboard ladder budget: 300B tokens, 16 nodes. Pure data parallel
# (TP=PP=1) with the distributed optimizer.

NUM_LAYERS=32
HIDDEN=2560
FFN_HIDDEN=7680
NUM_HEADS=32
NUM_KV_HEADS=8
SEQ_LEN=4096

MBS=${MBS:-1}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-73242188}  # ~300B tokens (leaderboard 2.7b budget)
SAVE_INTERVAL=57220

APERTUS_TRACK=transformer-pp-2.7b

MOE_ARGS=()                                # dense

DEFAULT_NODES=16
DEFAULT_TIME=72:00:00
