"""Tests for the fused SSSGLU kernels (``fused_bias_sssglu.py``).

SSSGLU is SwiGLU with the sigmoid inside SiLU replaced by softsign rescaled to (0, 1). The
fusion is a copy of SwiGLU's own (``@jit_fuser`` forward plus hand-derived analytic backward
wrapped in torch.autograd.Function), so the tests mirror ``test_swiglu_fusion.py`` and add an
fp64 autograd-reference check of the analytic backward. CUDA-gated like the other fusion tests;
run on the cluster GPU.
"""
import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_sssglu import (
    bias_sssglu_impl,
    ssslu,
    weighted_bias_sssglu_impl,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ref(x, bias=None):
    """Pure-torch autograd reference: ssslu(y1) * y2 with ssslu(y) = y*(0.5+0.5*softsign(y))."""
    if bias is not None:
        x = x + bias
    y_1, y_2 = torch.chunk(x, 2, -1)
    return (y_1 * (0.5 + 0.5 * F.softsign(y_1))) * y_2


def test_ssslu_matches_gate_definition():
    x = torch.randn(1024, dtype=torch.float64, device="cuda")
    gate = 0.5 * (1 + F.softsign(x))
    assert torch.allclose(ssslu(x), x * gate)
    # gate is softsign rescaled to (0, 1)
    assert gate.min() > 0 and gate.max() < 1


@pytest.mark.parametrize("input_dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_bias_sssglu_against_autograd_reference(input_dtype, use_bias):
    if input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda", requires_grad=True)
    # Like the fused SwiGLU original, the fused backward returns the unreduced (per-row) grad in
    # the bias slot, so the bias must not itself require grad through the autograd.Function here.
    bias = torch.randn(64, dtype=input_dtype, device="cuda") if use_bias else None
    g = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y_ref = _ref(x, bias)
    y_ref.backward(g)

    x_2 = x.detach().requires_grad_(True)
    y = bias_sssglu_impl(x_2, bias)
    y.backward(g)

    assert y.dtype == y_ref.dtype
    assert torch.allclose(y, y_ref, **tols)
    assert torch.allclose(x_2.grad, x.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_sssglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_sssglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_sssglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)
