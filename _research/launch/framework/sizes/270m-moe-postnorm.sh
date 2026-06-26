# shellcheck shell=bash
#
# 270m-moe-postnorm — 270m-moe rung (14L / 768H / 12h / 4kv, 64e top-2 + 1 shared)
# with the sandwich/post-norm residual variant. Same geometry as 270m-moe.sh; the
# only change is the residual block:
#
#     default :  x + sublayer(RMSNorm(x))
#     here    :  x + alpha * RMSNorm(sublayer(RMSNorm(x)))
#
# i.e. an extra RMSNorm on each attention/MLP sublayer output before the residual
# add (--post-norm), scaled by a fixed alpha (--fixed-layer-scale). The post-norm
# forces each sublayer's contribution to unit RMS, so the residual stream would
# grow like alpha*sqrt(2L); we set alpha = 1/sqrt(2L) to keep it O(1) — this is the
# value that takes over the role of the output-init depth downscaling, which we
# disable (--no-output-layer-init-depth-scaling) since it is a no-op under the
# scale-invariant RMSNorm anyway (muP width scaling, if any, is preserved).

NUM_LAYERS=14
HIDDEN=768
FFN_HIDDEN=1920
NUM_HEADS=12
NUM_KV_HEADS=4
SEQ_LEN=4096

MBS=${MBS:-8}
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3584000}   # ~15B tokens (override for other budgets)
SAVE_INTERVAL=2289

APERTUS_TRACK=270a-moe-postnorm

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 256
    --moe-layer-freq "([0]*1+[1]*13)"
)
EP=1                           # pure DP: all 64 experts replicated on each GPU

# Post-norm residual variant. alpha = 1/sqrt(2*NUM_LAYERS) (override FIXED_LAYER_SCALE
# to sweep). Appended after NETWORK_SIZE_ARGS via the common.sh escape hatch.
FIXED_LAYER_SCALE=${FIXED_LAYER_SCALE:-$(awk "BEGIN{printf \"%.10g\", 1/sqrt(2*$NUM_LAYERS)}")}
EXTRA_NETWORK_ARGS=(
    --post-norm
    --fixed-layer-scale "$FIXED_LAYER_SCALE"
    # --no-output-layer-init-depth-scaling
)
# Surface the layer scale in the run/wandb name (short form; full precision still
# goes to the flag). common.sh appends ARCH_KNOB_STR to KNOB_STR -> EXP_NAME.
ARCH_KNOB_STR="fls$(awk "BEGIN{printf \"%.4g\", $FIXED_LAYER_SCALE}")"

DEFAULT_NODES=2
DEFAULT_TIME=12:00:00
