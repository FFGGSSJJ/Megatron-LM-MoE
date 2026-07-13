"""Tests for the Triton RLGLU kernels (``megatron/core/transformer/moe/rlglu_jit.py``).

These are the fp32-internal Triton kernels used on the FP8 offloading-experts path (the sibling
of ``sssglu_jit.py`` / ``swiglu_jit.py``). RLGLU is a GLU whose gate is
``gate(a) = relu(a) - 0.5 * ln(1 + |a|)`` and output ``gate(a) * b``. Forward and backward
(including the per-row ``probs`` scaling and its gradient) are checked against a pure-torch
autograd reference. CUDA-gated like the other kernel tests; run on the cluster GPU.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

if torch.cuda.is_available():
    from megatron.core.transformer.moe.rlglu_jit import rlglu_forward, rlglu_backward


def _ref_forward(x, probs=None):
    """Reference ``gate(a) * b [* probs]`` with the [a|b] halves split along the last dim."""
    a, b = torch.chunk(x, 2, -1)
    y = (torch.relu(a) - 0.5 * torch.log1p(a.abs())) * b
    if probs is not None:
        y = y * probs
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_probs", [False, True])
@pytest.mark.parametrize("shape", [(16, 64), (128, 512)])
def test_rlglu_jit_matches_autograd_reference(dtype, use_probs, shape):
    M, two_d = shape
    D = two_d // 2
    tols = dict(rtol=2.0e-2, atol=1.0e-3) if dtype == torch.bfloat16 else dict(rtol=1.0e-4, atol=1.0e-5)

    x = torch.randn(M, two_d, dtype=dtype, device="cuda")
    probs = torch.rand(M, 1, dtype=dtype, device="cuda") if use_probs else None
    g = torch.randn(M, D, dtype=dtype, device="cuda")

    # Reference forward + backward via autograd (fp32 math to match the kernel's internal fp32).
    x_ref = x.detach().float().requires_grad_(True)
    probs_ref = probs.detach().float().requires_grad_(True) if use_probs else None
    y_ref = _ref_forward(x_ref, probs_ref)
    y_ref.backward(g.float())

    # Kernel forward.
    y = rlglu_forward(x, probs)
    assert y.shape == (M, D)
    assert y.dtype == dtype
    assert torch.allclose(y.float(), y_ref.detach(), **tols)

    # Kernel backward.
    if use_probs:
        grad_x, grad_probs = rlglu_backward(g, x, probs)
        assert torch.allclose(
            grad_probs.float(), probs_ref.grad.view(-1), **tols
        )
    else:
        grad_x = rlglu_backward(g, x, probs)
    assert grad_x.shape == (M, two_d)
    assert torch.allclose(grad_x.float(), x_ref.grad, **tols)


def test_rlglu_jit_gate_derivative_is_sssglu_gate():
    """The RLGLU kernel's da-branch derivative equals the SSSGLU gate (b == 1, no probs)."""
    from megatron.core.transformer.moe.sssglu_jit import sssglu_forward

    M, D = 64, 64
    a = torch.randn(M, D, dtype=torch.float32, device="cuda")
    x = torch.cat([a, torch.ones_like(a)], -1)  # b == 1 -> grad_a == gate'(a)
    g = torch.ones(M, D, dtype=torch.float32, device="cuda")
    grad_x = rlglu_backward(g, x)
    grad_a = torch.chunk(grad_x, 2, -1)[0]
    # SSSGLU gate as ssslu(a)/a, i.e. the (0,1) softsign gate == RLGLU gate'(a).
    sss_gate = 0.5 + 0.5 * a / (1.0 + a.abs())
    assert torch.allclose(grad_a, sss_gate, rtol=1e-4, atol=1e-5)


def test_rlglu_jit_differs_from_swiglu():
    """Sanity check that the RLGLU kernel is not accidentally computing SwiGLU."""
    from megatron.core.transformer.moe.swiglu_jit import swiglu_forward

    x = torch.randn(32, 128, dtype=torch.float32, device="cuda")
    assert not torch.allclose(rlglu_forward(x), swiglu_forward(x), rtol=1e-3, atol=1e-3)
