"""Tests for the Triton SSGLU kernels (``megatron/core/transformer/moe/ssglu_jit.py``).

These are the fp32-internal Triton kernels used on the FP8 offloading-experts path (the sibling
of ``swiglu_jit.py``). SSGLU is SwiGLU with the SiLU gate replaced by softsign rescaled to
(0, 1): ``sslu(a) = a * (0.5 + 0.5 * a / (1 + |a|))``. Forward and backward (including the
per-row ``probs`` scaling and its gradient) are checked against a pure-torch autograd reference.
CUDA-gated like the other kernel tests; run on the cluster GPU.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

if torch.cuda.is_available():
    from megatron.core.transformer.moe.ssglu_jit import ssglu_forward, ssglu_backward


def _ref_forward(x, probs=None):
    """Reference ``sslu(a) * b [* probs]`` with the [a|b] halves split along the last dim."""
    a, b = torch.chunk(x, 2, -1)
    y = (a * (0.5 + 0.5 * a / (1.0 + a.abs()))) * b
    if probs is not None:
        y = y * probs
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_probs", [False, True])
@pytest.mark.parametrize("shape", [(16, 64), (128, 512)])
def test_ssglu_jit_matches_autograd_reference(dtype, use_probs, shape):
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
    y = ssglu_forward(x, probs)
    assert y.shape == (M, D)
    assert y.dtype == dtype
    assert torch.allclose(y.float(), y_ref.detach(), **tols)

    # Kernel backward.
    if use_probs:
        grad_x, grad_probs = ssglu_backward(g, x, probs)
        assert torch.allclose(
            grad_probs.float(), probs_ref.grad.view(-1), **tols
        )
    else:
        grad_x = ssglu_backward(g, x, probs)
    assert grad_x.shape == (M, two_d)
    assert torch.allclose(grad_x.float(), x_ref.grad, **tols)


def test_ssglu_jit_differs_from_swiglu():
    """Sanity check that the SSGLU kernel is not accidentally computing SwiGLU."""
    from megatron.core.transformer.moe.swiglu_jit import swiglu_forward

    x = torch.randn(32, 128, dtype=torch.float32, device="cuda")
    assert not torch.allclose(ssglu_forward(x), swiglu_forward(x), rtol=1e-3, atol=1e-3)
