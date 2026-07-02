# shellcheck shell=bash
#
# 1.5b-moe-128e-sandwich — smallest 128e GQA rung with sandwich (post-)norm.
# Identical geometry/tokens/parallelism to 1.5b-moe-128e.sh; the ONLY change is
# an extra RMSNorm on each attention/MLP sublayer output before the residual add:
#
#     default  :  x + alpha * Sublayer(RMSNorm(x))
#     sandwich :  x + alpha * RMSNorm(Sublayer(RMSNorm(x)))     (--sandwich-norm)
#
# alpha = 1/sqrt(2*num_layers) is supplied by the ladder's --residual-output-scaling
# (already on in common.sh) and is applied AFTER the sandwich norm — so no separate
# --fixed-layer-scale is needed (that flag no longer exists). The residual-output
# XOR asserts (vs hypersphere-scale-out-proj-init / keel) are unaffected: sandwich
# norm is orthogonal to them.
#
# NOTE: --sandwich-norm wiring is freshly merged from moe/main — smoke-test one
# short run before launching the full sweep.

NUM_LAYERS=10
HIDDEN=768
FFN_HIDDEN=1920               # dense-layer MLP (~hidden*2.5)
NUM_HEADS=6                    # head_dim = hidden/heads = 128
NUM_KV_HEADS=3                 # heads/2 (constant 2:1 GQA ratio across the ladder)
SEQ_LEN=8192

MBS=${MBS:-2}                 # measured good on 8 nodes w/ EP=4/alltoall/fp8 (~8h); max 4 (GBS>=MBS*DP)
GBS=${GBS:-128}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-4598912}   # ~37.7B tokens = 100 tok/active-param (÷GBS)
SAVE_INTERVAL=3600            # ~10 saves over the run

APERTUS_TRACK=1.5a-moe-128e-sandwich

# 128e ladder trains on the fineweb-2-hq mul_200k blend (see lib/common.sh).
DATA_PRESET=${DATA_PRESET:-fineweb2hq-mul200k}

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 4
    --moe-ffn-hidden-size 448
    --moe-shared-expert-intermediate-size 448
    --moe-layer-freq "([0]*1+[1]*9)"
)
EP=${EP:-4}                    # shard 128 experts 4-ways over intra-node NVLink (throughput);
                               # requires DP % 4 == 0; try MOE_DISPATCHER=alltoall if it doesn't pay off.

# Sandwich/post-norm residual variant. Appended after NETWORK_SIZE_ARGS via the
# common.sh escape hatch; alpha comes from --residual-output-scaling (see above).
EXTRA_NETWORK_ARGS=(
    --sandwich-norm
)
# Surface the variant in the run/wandb name (common.sh appends ARCH_KNOB_STR to
# KNOB_STR -> EXP_NAME), so its runs never collide with the plain-GQA twin's.
ARCH_KNOB_STR="sandwich"

# Sized to finish TRAIN_SAMPLES in one ~9h allocation (<12h). ~9% MFU calibration
# (7b/810m-active/EP4 anchor: 32 nodes for 100B tok in 12h → GPU-hours ∝ active²).
DEFAULT_NODES=8
DEFAULT_TIME=12:00:00
