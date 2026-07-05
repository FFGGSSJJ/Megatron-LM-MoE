# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Tests for the fused GXPR GLU kernel.

Built the same way as SwiGLU's own fusion (``fused_bias_swiglu.py``): ``@jit_fuser``
(torch.compile)-fused elementwise math plus a hand-derived analytic backward, not a hand Triton
kernel (unlike PolyNorm GLU, this gate has no cross-feature reduction so it doesn't need one).

Validates the fused forward + backward against an independent torch-autograd reference (dense and
grouped/per-expert), and independently checks the backward with finite differences. The module
``use_fused`` gate only engages on CUDA (mirroring how the codebase's other bias/activation
fusions are exercised), so these tests are skipped on CPU-only boxes (e.g. the local Windows dev
env); run on the cluster GPU.
"""
import pytest
import torch

from megatron.core.activations import GXPR
from megatron.core.fusions.fused_gxpr_glu import fused_gxpr_glu_impl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ref(x_glu, x_linear, ap1, ap2, an, beta, score=None):
    """Pure-torch reference: gate(x_glu) * x_linear * [score], fp32 math cast to input dtype."""

    def col(a):
        a = a.float()
        return a.reshape(-1, 1) if a.numel() > 1 else a

    xf = x_glu.float()
    pos = xf > 0
    poly = col(ap2) * xf * xf + col(ap1) * xf + col(beta)
    neg = col(an) * torch.nn.functional.softsign(xf) + col(beta)
    gate = torch.where(pos, poly, neg)
    out = gate * x_linear.float()
    if score is not None:
        out = out * score.float().reshape(-1, 1)
    return out.to(x_glu.dtype)


def _tols(dtype):
    return dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-3)


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_score", [False, True])
@pytest.mark.parametrize("shape", [(16, 64), (128, 1024), (37, 257)])
def test_fused_matches_reference(dtype, use_score, shape):
    """Dense case: fused fwd + grads (dX, dY, dcoeffs, dscore) match the torch reference."""
    M, D = shape
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    ap1 = torch.rand(1, device="cuda", requires_grad=True)
    ap2 = torch.rand(1, device="cuda", requires_grad=True)
    an = torch.rand(1, device="cuda", requires_grad=True)
    beta = torch.rand(1, device="cuda", requires_grad=True)
    score = torch.rand(M, 1, device="cuda", requires_grad=True) if use_score else None
    g = torch.randn(M, D, dtype=dtype, device="cuda")

    def clones(*ts):
        return [t.detach().clone().requires_grad_(t.requires_grad) if t is not None else None for t in ts]

    xf, yf, ap1f, ap2f, anf, betaf, scf = clones(x, y, ap1, ap2, an, beta, score)

    y_fused = fused_gxpr_glu_impl(x, y, ap1, ap2, an, beta, score)
    y_ref = _ref(xf, yf, ap1f, ap2f, anf, betaf, score=scf)

    tols = _tols(dtype)
    assert y_fused.dtype == y_ref.dtype
    assert torch.allclose(y_fused, y_ref, **tols), (y_fused - y_ref).abs().max()

    y_fused.backward(g)
    y_ref.backward(g.clone())

    assert torch.allclose(x.grad, xf.grad, **tols), ("dX", (x.grad - xf.grad).abs().max())
    assert torch.allclose(y.grad, yf.grad, **tols), ("dY", (y.grad - yf.grad).abs().max())
    # coefficient grads accumulate over all M*D terms; compare in fp32 with a looser bf16 tol.
    atol_c = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=3e-2, atol=3e-2)
    assert torch.allclose(ap1.grad, ap1f.grad, **atol_c), ("dAP1", ap1.grad, ap1f.grad)
    assert torch.allclose(ap2.grad, ap2f.grad, **atol_c), ("dAP2", ap2.grad, ap2f.grad)
    assert torch.allclose(an.grad, anf.grad, **atol_c), ("dAN", an.grad, anf.grad)
    assert torch.allclose(beta.grad, betaf.grad, **atol_c), ("dBETA", beta.grad, betaf.grad)
    if use_score:
        assert torch.allclose(score.grad, scf.grad, **tols), ("dS", (score.grad - scf.grad).abs().max())


@pytest.mark.internal
def test_fused_backward_finite_difference():
    """Independent fp32 finite-difference check of the kernel backward (dX, dY, dcoeffs, dscore)."""
    torch.manual_seed(1)
    M, D = 4, 16
    x = torch.randn(M, D, dtype=torch.float64, device="cuda")
    y = torch.randn(M, D, dtype=torch.float64, device="cuda")
    coeffs = [torch.rand(1, dtype=torch.float64, device="cuda") for _ in range(4)]
    score = torch.rand(M, 1, dtype=torch.float64, device="cuda")

    def fwd(x_, y_, c_, s_):
        return fused_gxpr_glu_impl(
            x_.float(), y_.float(), c_[0].float(), c_[1].float(), c_[2].float(), c_[3].float(),
            s_.float(),
        )

    xv = x.float().requires_grad_(True)
    yv = y.float().requires_grad_(True)
    cv = [c.float().requires_grad_(True) for c in coeffs]
    sv = score.float().requires_grad_(True)
    out = fused_gxpr_glu_impl(xv, yv, cv[0], cv[1], cv[2], cv[3], sv)
    gout = torch.randn_like(out)
    out.backward(gout)

    def loss(x_, y_, c_, s_):
        return (fwd(x_, y_, c_, s_) * gout.double()).sum()

    eps_fd = 1e-4

    def fd_tensor(t, setter):
        grad = torch.zeros_like(t)
        flat = t.reshape(-1)
        gflat = grad.reshape(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps_fd
            lp = setter()
            flat[i] = orig - eps_fd
            lm = setter()
            flat[i] = orig
            gflat[i] = (lp - lm) / (2 * eps_fd)
        return grad

    dX_fd = fd_tensor(x, lambda: loss(x, y, coeffs, score).item())
    dY_fd = fd_tensor(y, lambda: loss(x, y, coeffs, score).item())
    dS_fd = fd_tensor(score, lambda: loss(x, y, coeffs, score).item())
    dC_fd = [fd_tensor(coeffs[k], lambda: loss(x, y, coeffs, score).item()) for k in range(4)]

    fd_tol = dict(rtol=2e-2, atol=2e-3)
    assert torch.allclose(xv.grad.double(), dX_fd, **fd_tol), (xv.grad.double() - dX_fd).abs().max()
    assert torch.allclose(yv.grad.double(), dY_fd, **fd_tol), (yv.grad.double() - dY_fd).abs().max()
    assert torch.allclose(sv.grad.double(), dS_fd, **fd_tol), (sv.grad.double() - dS_fd).abs().max()
    for k in range(4):
        assert torch.allclose(cv[k].grad.double(), dC_fd[k], **fd_tol), (k, cv[k].grad, dC_fd[k])


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_module_dense_uses_fused(dtype):
    """GXPR module (dense) on CUDA matches the reference and routes to the fused path."""
    M, D = 64, 256
    torch.manual_seed(2)
    x = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    mod = GXPR(num_local_experts=1, config=None).cuda().to(dtype)
    out = mod(x, y)
    ref = _ref(
        x.detach(), y.detach(),
        mod.alpha_p1.detach().abs(), mod.alpha_p2.detach().abs(),
        mod.beta.detach().abs() + mod.alpha_n.detach().abs(), mod.beta.detach().abs(),
    )
    assert torch.allclose(out, ref, **_tols(dtype)), (out - ref).abs().max()
    out.sum().backward()
    assert x.grad is not None and mod.alpha_p1.grad is not None


@pytest.mark.internal
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_module_grouped_matches_reference(dtype):
    """GXPR module (grouped, per-expert coeffs + probs) matches the reference."""
    E = 3
    tpe = [5, 0, 11]  # include an empty expert
    Mt, D = sum(tpe), 128
    torch.manual_seed(3)
    x = torch.randn(Mt, D, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.randn(Mt, D, dtype=dtype, device="cuda", requires_grad=True)
    probs = torch.rand(Mt, 1, device="cuda", requires_grad=True)
    mod = GXPR(num_local_experts=E, config=None).cuda().to(dtype)
    with torch.no_grad():
        mod.alpha_p1.copy_(torch.tensor([0.1, 0.3, 0.5], device="cuda").to(dtype))
        mod.alpha_p2.copy_(torch.tensor([0.2, 0.4, 0.6], device="cuda").to(dtype))
        mod.alpha_n.copy_(torch.tensor([0.15, 0.25, 0.35], device="cuda").to(dtype))
        mod.beta.copy_(torch.tensor([0.5, 0.5, 0.5], device="cuda").to(dtype))
    out = mod(x, y, tokens_per_expert=tpe, scores=probs)

    tpe_t = torch.tensor(tpe, device="cuda")
    ap1 = torch.repeat_interleave(mod.alpha_p1.detach().abs(), tpe_t)
    ap2 = torch.repeat_interleave(mod.alpha_p2.detach().abs(), tpe_t)
    beta = torch.repeat_interleave(mod.beta.detach().abs(), tpe_t)
    an = beta + torch.repeat_interleave(mod.alpha_n.detach().abs(), tpe_t)
    ref = _ref(x.detach(), y.detach(), ap1, ap2, an, beta, score=probs.detach())
    assert torch.allclose(out, ref, **_tols(dtype)), (out - ref).abs().max()
    out.sum().backward()
    assert mod.alpha_p1.grad.shape == (E,)
    assert x.grad is not None and probs.grad is not None


@pytest.mark.internal
@pytest.mark.parametrize("D", [63, 1024, 4097, 8193])
def test_module_matches_reference_at_odd_and_wide_feature_dims(D):
    """No MAX_FUSED_FEATURE_DIM cap (unlike PolyNorm) -- non-power-of-2 and very wide D all use
    the fused path directly, with no block-per-row padding to worry about."""
    M = 4
    torch.manual_seed(4)
    x = torch.randn(M, D, device="cuda", requires_grad=True)
    y = torch.randn(M, D, device="cuda", requires_grad=True)
    mod = GXPR(num_local_experts=1, config=None).cuda()
    out = mod(x, y)
    ref = _ref(
        x.detach(), y.detach(),
        mod.alpha_p1.detach().abs(), mod.alpha_p2.detach().abs(),
        mod.beta.detach().abs() + mod.alpha_n.detach().abs(), mod.beta.detach().abs(),
    )
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    out.sum().backward()
    assert x.grad is not None
