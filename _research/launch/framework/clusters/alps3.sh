# shellcheck shell=bash
#
# alps3 — GH200 nodes, the default (and only) cluster for megatron-apertus-moe.
#
# ── cluster contract ─────────────────────────────────────────────────────────
# A cluster file MAY set (defaults in common.sh reproduce alps3):
#   CONTAINER                  # absolute path to the container EDF .toml, OR a
#                              # bare filename resolved under _research/launch/
#   SRUN_MPI                   # srun --mpi flavor (pmix | pmi2)
#   SRUN_EXTRA_ARGS=(...)      # extra srun flags
#   MIXED_PRECISION_ARGS=(...) # precision flags (bf16 / fp8 recipe)
#   MOE_DISPATCHER             # --moe-token-dispatcher-type value
#   MOE_PERMUTE_FUSION         # 1 (default) adds --moe-permute-fusion; 0 omits
#   CLUSTER_TRAIN_ARGS=(...)   # appended to TRAINING_ARGS
#   ARCH_TAG                   # suffix for triton/inductor caches + packages dir
#   CLUSTER_TAG                # suffix appended to EXP_NAME (and job name)
#   CLUSTER_SBATCH_FLAGS=(...) # sbatch flags submit.sh (and auto-requeue) adds
#                              # on top of train.sbatch's generic header
# Everything uses ${VAR:-} / declare-guards so the env and earlier-sourced
# size/recipe fragments win over the cluster default.
# ─────────────────────────────────────────────────────────────────────────────

# gfu's pytorch2512 image bundles deep_gemm 2.5.0 + emerging_optimizers 0.2.0 +
# grouped-GEMM (so FP8 works and no pip install of those is needed). This is the
# image submit_test.sh ran the md_decoupling recipe on. The skinny
# _research/launch/alps3.toml (NGC, no deep_gemm) remains a fallback for
# FP8-disabled runs — pass CONTAINER=alps3.toml to use it.
CONTAINER=${CONTAINER:-/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-a139.toml}
SRUN_MPI=${SRUN_MPI:-pmix}
declare -p SRUN_EXTRA_ARGS >/dev/null 2>&1 || SRUN_EXTRA_ARGS=(--network=disable_rdzv_get)
# bf16 only — the proven md_decoupling run (submit_test.sh) trained in bf16 with
# FP8 commented out. To enable the DeepSeek-V3 FP8 blockwise recipe (Hopper,
# matmuls only; params/master-weights stay bf16) on the deep_gemm image:
#   MIXED_PRECISION_ARGS=(--bf16 --fp8-format e4m3 --fp8-recipe blockwise) bash submit.sh ...
declare -p MIXED_PRECISION_ARGS >/dev/null 2>&1 || MIXED_PRECISION_ARGS=(
    --bf16
)
# Dispatcher default is EP-aware (EP is set by the size file, sourced before this):
# the allgather dispatcher materializes every EP-group rank's tokens on each rank
# (OOMs at EP>1, e.g. EP=4 = 4x the token buffer), so default to alltoall — which
# routes each token only to its expert's rank — whenever experts are sharded.
# EP=1 keeps allgather (cheapest with no expert sharding). Env/size still overrides.
if [ "${EP:-1}" -gt 1 ]; then
    MOE_DISPATCHER=${MOE_DISPATCHER:-alltoall}
else
    MOE_DISPATCHER=${MOE_DISPATCHER:-allgather}
fi
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-1}
declare -p CLUSTER_TRAIN_ARGS >/dev/null 2>&1 || CLUSTER_TRAIN_ARGS=()
ARCH_TAG=${ARCH_TAG:-}
CLUSTER_TAG=${CLUSTER_TAG:-}
declare -p CLUSTER_SBATCH_FLAGS >/dev/null 2>&1 || CLUSTER_SBATCH_FLAGS=(
    --account=infra01
    --cpus-per-task=72
    --mem=460000
)
