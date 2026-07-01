# shellcheck shell=bash
#
# muon — Muon (orthogonalized momentum) on 2D non-embedding/output matrices via
# emerging_optimizers, with a scalar optimizer (AdamW by default) on everything
# else (embeddings, LM head, norms, biases). Both groups run at the base --lr;
# --muon-scale-mode RMS-matches the Muon update to Adam so a single Adam-scale LR
# (~1e-3) works for both.
#
# Muon is INCOMPATIBLE with the standard --use-distributed-optimizer. Two flavors:
#   dist_muon (default, MUON_DIST=1) — shards optimizer state over DP (layer-wise)
#       and supports --overlap-grad-reduce. Use this for real / multi-node runs.
#   muon      (MUON_DIST=0)          — no optimizer-state sharding, no overlap
#       (every DP rank holds the full fp32 state). Simpler; heavier on memory.
# Muon needs --ckpt-format torch/torch_dist (common.sh sets torch).
#
# Knobs (override via env, e.g. `LR=2e-3 MUON_SCALE_MODE=unit_rms_norm submit.sh ...`):
#   LR               base LR (Muon matrices + scalar optimizer)
#   MIN_LR           LR floor
#   WEIGHT_DECAY     decoupled weight decay on matrices (1D params get none)
#   MUON_MOMENTUM    Muon momentum beta (default 0.95)
#   MUON_SCALE_MODE  spectral|unit_rms_norm|shape_scaling|shape_up|none (default shape_up)
#   MUON_SCALAR_OPT  scalar-param optimizer: adam|lion (default adam)
#   MUON_DIST        1 → dist_muon (sharded, default); 0 → plain muon

EXP_TAG=muon

LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-5}
KNOB_STR=lr${LR}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
[ "$WEIGHT_DECAY" != 0.1 ] && KNOB_STR=${KNOB_STR}-wd${WEIGHT_DECAY}
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# Linear decay to MIN_LR over all of TRAIN_SAMPLES (auto-scales with run length).
LR_DECAY_STYLE=${LR_DECAY_STYLE:-linear}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-0}

MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_SCALE_MODE=${MUON_SCALE_MODE:-shape_scaling}
MUON_SCALAR_OPT=${MUON_SCALAR_OPT:-adam}

# dist_muon shards optimizer state over DP (layer-wise, from the optimizer name)
# and allows grad-reduce overlap; plain muon does neither. Neither uses the
# standard --use-distributed-optimizer, so override common.sh's default DIST_OPT_ARGS.
MUON_DIST=${MUON_DIST:-1}
if [ "$MUON_DIST" = 1 ]; then
    OPTIMIZER=dist_muon
    DIST_OPT_ARGS=(--overlap-grad-reduce)
else
    OPTIMIZER=muon
    DIST_OPT_ARGS=()
    KNOB_STR=${KNOB_STR}-nodist
fi

RECIPE_ARGS=(
    --muon-momentum "$MUON_MOMENTUM"
    --muon-use-nesterov
    --muon-scale-mode "$MUON_SCALE_MODE"
    --muon-scalar-optimizer "$MUON_SCALAR_OPT"
)
