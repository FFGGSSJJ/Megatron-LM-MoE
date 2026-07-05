# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# pylint: disable=missing-function-docstring, missing-class-docstring

"""Fused kernel for the GXPR GLU activation, built the same way as SwiGLU's own fusion
(``fused_bias_swiglu.py``): ``@jit_fuser`` (torch.compile)-fused elementwise math plus a
hand-derived analytic backward, wrapped in a ``torch.autograd.Function``, rather than a
hand-written Triton kernel.

For a token row ``x`` (the GLU gate half) and ``y`` (the GLU linear half ``x_linear``) this
computes::

    gate = ap2*x**2 + ap1*x + beta        if x > 0
         = an*softsign(x) + beta           if x <= 0
    out  = gate * y * score                # score optional (per token)

GXPR's gate has **no cross-element reduction** -- unlike PolyNorm GLU (which needs a full-row
RMSNorm reduction and therefore a hand Triton kernel, one block per row, capped at
``MAX_FUSED_FEATURE_DIM``) every output element here depends only on the matching elements of
``x``/``y``, exactly like SwiGLU's ``silu(x) * y``. That means torch.compile's ordinary pointwise
codegen (the same mechanism ``swiglu``/``weighted_swiglu`` use) already tiles this optimally
without any row-alignment padding, and there's no feature-dimension cap to enforce.

Two measured-motivated micro-optimizations vs. the first version of this file (real 3B-model A/B:
GXPR ran ~14% slower than SwiGLU even with the fusion mechanism unified):

1. **``ap1``/``ap2``/``an``/``beta`` are packed into one ``coeffs`` tensor** before entering the
   custom autograd Function (via a plain, differentiable ``torch.cat`` -- autograd splits the
   gradient back into the four original parameter grads automatically, no manual code needed).
   torch.compile re-validates its guards (dtype/device/shape/stride/etc.) against *every* tensor
   argument on *every* call; that cost scales with argument count and recurs once per MoE layer
   per micro-batch, not with the tensor sizes -- so it doesn't amortize away just because the
   model is bigger, only because there are fewer arguments to guard. Packing 4 small coefficient
   tensors into 1 cuts that specific overhead roughly 4x.
2. **The ``+ beta`` term (common to both branches) is hoisted out of the ``torch.where``**, so it
   is computed once per element instead of once per branch. ``torch.where``/predicated-select
   does not short-circuit on a GPU -- both branches are always evaluated for every element, so
   GXPR inherently does more arithmetic per element than SwiGLU's single-formula gate; this
   doesn't remove that structural cost (nothing implementation-level can, short of changing the
   math), but it does remove the one clearly redundant operation Inductor wasn't deduplicating on
   its own (confirmed by reading the generated Triton source: it computed ``+ beta`` twice, once
   per branch, before the ``where``).

``ap1``/``ap2``/``an``/``beta`` broadcast against ``x``/``y`` exactly like the eager
(``compiled_xpr_gate``) path: shape ``(1,)`` for a single shared (dense) coefficient, or
``(num_tokens, 1)`` for the grouped-expert case (the owning module, :class:`~megatron.core.
activations.GXPR`, is responsible for that broadcast shape via ``_broadcast_coeffs``). ``abs()``
(and the ``beta + |alpha_n|`` composition) is applied by the module *outside* this fusion, so the
gradients returned here are w.r.t. the already-positive ``ap1/ap2/an/beta`` tensors passed in --
ordinary broadcasting-aware reduction (``_sum_to_shape``) then produces the correctly-shaped
gradient for whatever shape was passed in, mirroring how ``weighted_swiglu_back`` reduces its
per-token ``weights`` gradient.
"""

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator


def _sum_to_shape(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Sum ``grad`` down to ``shape``, undoing whatever broadcasting the forward applied.

    Mirrors what autograd does automatically for a broadcasted-multiply operand; needed here
    because the backward below is hand-derived (not autograd-through-``@jit_fuser``).
    """
    while grad.dim() > len(shape):
        grad = grad.sum(0)
    for dim, size in enumerate(shape):
        if size == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad


def _unpack_coeffs(coeffs):
    """Split the packed ``(..., 4)`` coefficient tensor into ``(ap1, ap2, an, beta)``, each
    ``(..., 1)``. A cheap view/slice, not a copy -- negligible cost inside the compiled region.
    """
    return coeffs[..., 0:1], coeffs[..., 1:2], coeffs[..., 2:3], coeffs[..., 3:4]


@jit_fuser
def gxpr_glu(x, y, coeffs):
    """GXPR GLU gate * y, fused in one compiled pass (mirrors ``swiglu(y)``).

    ``coeffs`` packs ``(ap1, ap2, an, beta)`` along its last dim -- see module docstring for why.
    """
    ap1, ap2, an, beta = _unpack_coeffs(coeffs)
    pos = x > 0
    delta = torch.where(pos, ap2 * x * x + ap1 * x, an * F.softsign(x))
    return (delta + beta) * y


@jit_fuser
def weighted_gxpr_glu(x, y, coeffs, score):
    dtype = x.dtype
    res = gxpr_glu(x, y, coeffs) * score
    return res.to(dtype)


@jit_fuser
def gxpr_glu_back(g, x, y, coeffs):
    """Analytic backward for ``gxpr_glu``. Returns ``(dx, dy, dcoeffs)``, ``dcoeffs`` packed the
    same way as the input ``coeffs`` (already reduced to that shape)."""
    ap1, ap2, an, beta = _unpack_coeffs(coeffs)
    pos = x > 0
    inv_denom = 1.0 / (1.0 + torch.abs(x))  # 1/(1+|x|); softsign(x) = x * inv_denom
    softsign = x * inv_denom
    delta = torch.where(pos, ap2 * x * x + ap1 * x, an * softsign)
    gate = delta + beta

    dgate = g * y
    dy = g * gate
    gate_prime = torch.where(pos, 2.0 * ap2 * x + ap1, an * inv_denom * inv_denom)
    dx = dgate * gate_prime

    zeros = torch.zeros_like(x)
    dap1 = torch.where(pos, dgate * x, zeros)
    dap2 = torch.where(pos, dgate * x * x, zeros)
    dan = torch.where(pos, zeros, dgate * softsign)
    dbeta = dgate  # both branches add beta with coefficient 1

    dap1 = _sum_to_shape(dap1, ap1.shape)
    dap2 = _sum_to_shape(dap2, ap2.shape)
    dan = _sum_to_shape(dan, an.shape)
    dbeta = _sum_to_shape(dbeta, beta.shape)
    dcoeffs = torch.cat([dap1, dap2, dan, dbeta], dim=-1)
    return dx, dy, dcoeffs


@jit_fuser
def weighted_gxpr_glu_back(g, x, y, coeffs, score):
    input_dtype = x.dtype
    score_dtype = score.dtype
    dx, dy, dcoeffs = gxpr_glu_back(g * score, x, y, coeffs)
    # d(out)/d(score) = gate(x)*y == the *unweighted* gxpr_glu output.
    score_grad = gxpr_glu(x, y, coeffs) * g.to(score_dtype)
    score_grad = _sum_to_shape(score_grad, score.shape)
    return dx.to(input_dtype), dy.to(input_dtype), dcoeffs, score_grad.to(score_dtype)


class GXPRGLUFunction(torch.autograd.Function):
    """Custom autograd function for GXPR GLU (no score)."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, x, y, coeffs):
        ctx.save_for_backward(x, y, coeffs)
        return gxpr_glu(x, y, coeffs)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        x, y, coeffs = ctx.saved_tensors
        return gxpr_glu_back(grad_output, x, y, coeffs)


class WeightedGXPRGLUFunction(torch.autograd.Function):
    """Custom autograd function for GXPR GLU with a per-token ``score`` multiplier."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, x, y, coeffs, score):
        ctx.save_for_backward(x, y, coeffs, score)
        return weighted_gxpr_glu(x, y, coeffs, score)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        x, y, coeffs, score = ctx.saved_tensors
        return weighted_gxpr_glu_back(grad_output, x, y, coeffs, score)


def fused_gxpr_glu_impl(x_glu, x_linear, alpha_p1, alpha_p2, alpha_n, beta, score=None):
    """Fused GXPR GLU: ``gate(x_glu) * x_linear * [score]``.

    Args:
        x_glu, x_linear: ``(..., D)`` gate / linear halves of the fc1 output (same shape & dtype).
        alpha_p1, alpha_p2, alpha_n, beta: positive coefficients, broadcastable against the
            flattened ``(M, D)`` view -- shape ``(1,)`` (shared/dense) or ``(M, 1)`` (per-token,
            grouped experts). Packed into one tensor here (see module docstring) before entering
            the compiled/autograd boundary; ordinary ``torch.cat`` autograd splits the combined
            gradient back into the four original tensors' gradients on the way out.
        score: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).

    Returns:
        Tensor of ``x_glu``'s shape and dtype.
    """
    ori_shape = x_glu.shape
    assert len(ori_shape) in (2, 3)
    x = x_glu.reshape(-1, ori_shape[-1])
    y = x_linear.reshape(-1, ori_shape[-1])
    coeffs = torch.cat([alpha_p1, alpha_p2, alpha_n, beta], dim=-1)

    if score is not None:
        score = score.reshape(-1, 1)
        out = WeightedGXPRGLUFunction.apply(x, y, coeffs, score)
    else:
        out = GXPRGLUFunction.apply(x, y, coeffs)

    return out if len(ori_shape) == 2 else out.view(ori_shape[0], ori_shape[1], -1)
