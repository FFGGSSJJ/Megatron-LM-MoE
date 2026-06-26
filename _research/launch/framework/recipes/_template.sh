# shellcheck shell=bash
#
# _template.sh — copy this to add a new optimizer/idea.
#
#   cp _template.sh my-idea.sh
#   # edit, then:
#   SIZE=420m-moe RECIPE=my-idea bash submit.sh
#
# A recipe is the ONLY file you write. Everything else (model size, data, SLURM,
# logging, launch) is supplied by sizes/ and lib/common.sh. See README.md for
# the full variable contract.
#
# ── two ways to start ────────────────────────────────────────────────────────
# (A) From scratch: set OPTIMIZER + RECIPE_ARGS yourself (see md_decoupling.sh).
# (B) Inherit an existing recipe and tweak — uncomment to base on md_decoupling:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/md_decoupling.sh"
#   EXP_TAG=md-myidea
#   RECIPE_ARGS+=(--my-new-flag 0.95)   # append your idea's flags
#   return 0   # nothing more needed
# ─────────────────────────────────────────────────────────────────────────────

OPTIMIZER=adam                 # --optimizer value (adam|sgd|muon|lion|md_decoupling)
EXP_TAG=my-idea                # short slug -> shows up in EXP_NAME

# Learning rate + schedule (omit any line to take common.sh's default).
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-1e-5}
KNOB_STR=lr${LR}               # appended to the run name; add the knobs you sweep

# Regularization (optional overrides).
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
# CLIP_GRAD=1.0

# An optimizer that forbids the standard distributed optimizer (e.g.
# md_decoupling) overrides the sharding flags here:
# DIST_OPT_ARGS=(--use-layer-wise-distributed-optimizer --overlap-grad-reduce)

# The actual idea: optimizer/arch-specific flags. This is usually all that
# differs between your baseline and the next person's.
RECIPE_ARGS=(
    # --your-flag value
)
