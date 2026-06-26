#!/bin/bash
# Skinny python-deps installer for megatron-apertus-moe (MoE GPT runs).
# Adapted from the research-baseline launcher: keeps wandb/omegaconf +
# emerging_optimizers; drops the hybrid/deltanet stack (fla-core, tilelang, mamba).
#
# NOTE: the default container (gfu's pytorch2511 image) already bundles
# emerging_optimizers AND deep_gemm, so this installer is only a portability
# safety net (e.g. for the bare alps3 fallback). deep_gemm is NOT pip-installed
# here — its build needs CUTLASS/CUDA and fails in the bare NGC container; the MoE
# stack imports without it (the deep_gemm import in fp8_jit.py is now guarded), so
# it is only required for FP8 runs, which should use the deep_gemm-bundled image.
set -euo pipefail
PKG_DIR=${1:?usage: install_python_deps.sh PKG_DIR}
MARKER=$PKG_DIR/.deps_installed_moe_v1
LOCK=$PKG_DIR/.install.lockdir

mkdir -p "$PKG_DIR"
if [ -f "$MARKER" ]; then exit 0; fi

while ! mkdir "$LOCK" 2>/dev/null; do
    sleep 5
    if [ -f "$MARKER" ]; then exit 0; fi
done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

if [ -f "$MARKER" ]; then exit 0; fi

if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install --quiet"
else
    INSTALL="pip install --quiet"
fi

# $PKG_DIR is prepended to PYTHONPATH, so anything installed here shadows the
# container's matched torch/transformers/hf_hub stack and breaks imports. Install
# only container-absent packages; --no-deps everything below so none drags torch.
$INSTALL --target="$PKG_DIR" wandb omegaconf
# emerging_optimizers: only installed if the container doesn't already provide it
# (the gfu image ships 0.2.0). Harmless to install the same version into $PKG_DIR.
if ! python3 -c "import emerging_optimizers" 2>/dev/null; then
    $INSTALL --no-deps --target="$PKG_DIR" \
        'git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'
fi

touch "$MARKER"
