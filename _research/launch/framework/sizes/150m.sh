# shellcheck shell=bash
#
# 150m — TINY dense debug rung for very quick runs. 8L / 512H / 8h / 2kv (GQA).
# Same conventions as the MoE ladder (untied embeddings, ffn = 2.5*hidden,
# head_dim 64, 4:1 GQA) but dense and as small as the 131k vocab allows: the
# untied embedding slab alone is 134M, so total is ~0.15B while the non-embed
# backbone is only ~16M. Don't read loss off this rung — it's for smoke tests,
# 1 node, MBS 32 x DP 4 = GBS 128 with no gradient accumulation.

NUM_LAYERS=8
HIDDEN=512
FFN_HIDDEN=1280
NUM_HEADS=8
NUM_KV_HEADS=2
SEQ_LEN=4096

MBS=${MBS:-16}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1280000}    # ~5.2B tokens, 10k steps (override for even quicker runs)
SAVE_INTERVAL=2288                        # ~10 saves over the run

APERTUS_TRACK=dense-150m

# 150m runs are length-ablation probes — keep them off the main optimizer
# board. debug.sh's "debug-" and any env override still win (`:-` default).
WANDB_PROJECT_PREFIX=${WANDB_PROJECT_PREFIX:-len-ablation-}

MOE_ARGS=()                                # dense

DEFAULT_NODES=1
DEFAULT_TIME=12:00:00
