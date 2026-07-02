#!/bin/bash
#
# dump-args.sh — print the composed MEGATRON_ARGS for a (SIZE, RECIPE), one
# token per line, WITHOUT launching anything. Stubs the SLURM commands so it
# runs on a login node.
#
#   SIZE=420m-moe RECIPE=md_decoupling bash dump-args.sh
#   CLUSTER=alps3 SIZE=420m-moe RECIPE=md_decoupling bash dump-args.sh
#
set -euo pipefail

LIB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
FRAMEWORK_DIR=$(cd "$LIB_DIR/.." && pwd)

: "${SIZE:?set SIZE}"
: "${RECIPE:?set RECIPE}"
CLUSTER=${CLUSTER:-alps3}
# Data falls through to common.sh's default (the swissai blend) for dry
# inspection unless MEGATRON_DATA_PATH / DATA_ROOT override it.

# Stub SLURM + git so sourcing works anywhere.
srun() { :; }
scontrol() { hostname; }
export -f srun scontrol
: "${SLURM_SUBMIT_DIR:=$(cd "$FRAMEWORK_DIR/../../.." && pwd)}"
export SLURM_SUBMIT_DIR
export DRY_RUN=1

# Send the fragments' informational echoes to stderr; keep stdout clean so it
# carries only the arg list.
{
    source "$FRAMEWORK_DIR/sizes/$SIZE.sh"
    source "$FRAMEWORK_DIR/recipes/$RECIPE.sh"
    source "$FRAMEWORK_DIR/clusters/$CLUSTER.sh"
    source "$FRAMEWORK_DIR/lib/common.sh"
} 1>&2

printf '%s\n' "${MEGATRON_ARGS[@]}"
