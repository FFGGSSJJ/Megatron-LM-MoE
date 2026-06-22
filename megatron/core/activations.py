# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.fusions.fused_polynorm_glu import (
    MAX_FUSED_FEATURE_DIM,
    HAVE_TRITON as HAVE_FUSED_PNGLU,
    fused_polynorm_glu_impl,
)
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def compiled_polynorm(x, alpha_1, alpha_2, alpha_3, eps: float = 1e-6):
    """Core PolyNorm GLU gate: ``a1*RMSNorm(x) + a2*RMSNorm(x**2) + a3*RMSNorm(x**3)``.

    The RMS normalization is taken over the last (feature) dimension. The math is done in
    fp32 for numerical stability and cast back to the input dtype, mirroring how the
    RMSNorm/LayerNorm layers in this codebase behave under mixed precision.

    ``alpha_1``/``alpha_2``/``alpha_3`` broadcast against ``x``. They are either a single
    (broadcastable) coefficient of shape ``(1,)`` (dense / single-expert case) or per-token
    coefficients of shape ``(num_tokens, 1)`` (grouped-expert case, where each token already
    carries the coefficient of the expert it was routed to).

    This is the (torch.compile-fused) **gate-only** computation used by the non-Triton fallback
    paths; the CUDA fast path fuses the gate, the ``* x_linear`` and the ``* score`` multiplies in
    a single Triton kernel (see ``fused_polynorm_glu_impl``).
    """
    input_dtype = x.dtype
    x = x.float()

    def norm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    out = alpha_1 * norm(x) + alpha_2 * norm(x * x) + alpha_3 * norm(x * x * x)
    return out.to(input_dtype)


class _AllReduceSumSymmetric(torch.autograd.Function):
    """All-reduce(sum) over ``group`` in BOTH the forward and backward passes.

    Used to turn each rank's partial feature-sum into the full sum when the result is then
    consumed independently on every rank (each rank normalizes its own tokens with the shared
    statistic). Because the reduced value feeds rank-local downstream work, the gradient must
    be summed back across the group — unlike ``reduce_from_tensor_model_parallel_region``
    (forward all-reduce, backward identity), which is only correct when the reduced value feeds
    *replicated* downstream work.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        x = x.clone()
        torch.distributed.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class _SyncGradSum(torch.autograd.Function):
    """Identity in the forward pass; all-reduce(sum) the gradient over ``group`` in backward.

    Applied to the (TP-replicated) alpha coefficients so each rank's partial coefficient
    gradient — a sum over only that rank's feature shard — is completed into the full gradient,
    keeping the replicas in sync. (Same semantics as ``copy_to_tensor_model_parallel_region``,
    but over an arbitrary group so it also works for the expert-tensor-parallel group.)
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class PolyNorm(MegatronModule):
    """Learnable PolyNorm GLU activation — a drop-in replacement for the gate of a gated
    linear unit (e.g. SiLU in SwiGLU).

    In a GLU the first linear layer produces ``[x_glu, x_linear]`` and the block output is
    ``gate(x_glu) * x_linear``. Standard SwiGLU uses ``gate = SiLU``. Here the gate is the
    (3rd-order) PolyNorm::

        gate(x) = |alpha_1| * RMSNorm(x) + |alpha_2| * RMSNorm(x ** 2) + |alpha_3| * RMSNorm(x ** 3)

    where ``alpha_1``/``alpha_2``/``alpha_3`` are learnable (``abs`` keeps them positive).

    ``forward`` takes *both* GLU halves and returns the full ``gate(x_glu) * x_linear * [score]``
    product. On CUDA (and ``tp_size == 1``) the gate, the GLU multiply and the optional per-token
    ``score`` multiply (MoE router probs / per-token scale) are fused into a single Triton kernel
    (see ``megatron.core.fusions.fused_polynorm_glu``) so the op runs close to SwiGLU speed and is
    shape-agnostic over the (variable) MoE token count. Otherwise the gate is computed with the
    torch fallback (``compiled_polynorm`` or, when TP-sharded, ``_tp_forward``) and the
    multiplies are applied in eager torch.

    To support grouped MoE experts (where the activations of all local experts are
    concatenated along the token dimension and processed in a single call) this module holds
    one ``(alpha_1, alpha_2, alpha_3)`` triple *per local expert*: ``alpha_1``/``alpha_2``/
    ``alpha_3`` have shape ``(num_local_experts,)``. When ``tokens_per_expert`` is supplied the
    per-expert coefficients are expanded to per-token coefficients, so every token is gated by the
    coefficients of the expert it was routed to. For a dense MLP (or a ``SequentialMLP``
    expert) ``num_local_experts == 1`` and the single coefficient is broadcast to all tokens.

    Tensor parallelism: the RMSNorm reduces over the ffn feature dimension, which is sharded
    across ``tp_group`` (the main TP group for dense/shared MLPs, the expert-TP group for MoE
    experts). When ``tp_group`` has size > 1, the per-token sum-of-squares is all-reduced over
    the group (forward and backward) so every rank uses the *full-feature* RMS, and the
    replicated ``alpha`` gradients are all-reduced over the group so the replicas stay in sync.
    The result is therefore identical to (and bitwise-consistent across) any TP/ETP degree.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_init: float = 0.2,
        eps: float = 1e-6,
        tp_group: "torch.distributed.ProcessGroup | None" = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_1 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_2 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_3 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.eps = eps
        # The group over which the ffn feature dimension is sharded. tp_size==1 (no sharding,
        # e.g. local CPU runs or ETP=1 experts) takes the cheap fused path with no collectives.
        self.tp_group = tp_group
        if tp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.tp_size = torch.distributed.get_world_size(group=tp_group)
        else:
            self.tp_size = 1

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        # Keep the coefficients positive.
        alpha_1 = torch.abs(self.alpha_1)  # (num_local_experts,)
        alpha_2 = torch.abs(self.alpha_2)
        alpha_3 = torch.abs(self.alpha_3)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    "PolyNorm with num_local_experts > 1 requires `tokens_per_expert` so "
                    "the per-expert coefficients can be mapped onto the concatenated tokens."
                )
            # Single coefficient broadcast to every token: shape (1,).
            a1, a2, a3 = alpha_1, alpha_2, alpha_3
        else:
            # Expand per-expert coefficients to per-token coefficients: shape (num_tokens,).
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x_glu.device)
            a1 = torch.repeat_interleave(alpha_1, tpe_tensor)
            a2 = torch.repeat_interleave(alpha_2, tpe_tensor)
            a3 = torch.repeat_interleave(alpha_3, tpe_tensor)

        use_fused = (
            HAVE_FUSED_PNGLU
            and x_glu.is_cuda
            and self.tp_size == 1
            and x_glu.shape[-1] <= MAX_FUSED_FEATURE_DIM
            and (self.config is None or getattr(self.config, "pnglu_fusion", True))
        )
        if use_fused:
            # Single fused kernel: gate + (* x_linear) + (* scores), shape-agnostic over tokens.
            return fused_polynorm_glu_impl(x_glu, x_linear, a1, a2, a3, self.eps, scores)

        # Fallback: compute the gate in torch, then apply the multiplies in eager mode.
        a1b = a1.unsqueeze(-1) if a1.dim() == 1 and self.num_local_experts > 1 else a1
        a2b = a2.unsqueeze(-1) if a2.dim() == 1 and self.num_local_experts > 1 else a2
        a3b = a3.unsqueeze(-1) if a3.dim() == 1 and self.num_local_experts > 1 else a3
        if self.tp_size == 1:
            # ffn feature dim is whole on this rank: cheap fused per-token norm.
            gate = compiled_polynorm(x_glu, a1b, a2b, a3b, self.eps)
        else:
            # ffn feature dim is TP-sharded: reduce the feature statistics across the group.
            gate = self._tp_forward(x_glu, a1b, a2b, a3b)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out

    def _tp_forward(self, x, alpha_1, alpha_2, alpha_3):
        """TP-invariant path: recover the full-feature RMS from the local feature shards."""
        input_dtype = x.dtype
        xf = x.float()
        # Each ColumnParallel rank holds an equal 1/tp_size slice of the ffn features.
        n_global = xf.shape[-1] * self.tp_size
        # Per-token partial feature sums on this rank: sum(x^2), sum(x^4) (== sum((x^2)^2)) and
        # sum(x^6) (== sum((x^3)^2)) for RMSNorm(x), RMSNorm(x^2), RMSNorm(x^3). One symmetric
        # all-reduce completes all three.
        s1 = xf.pow(2).sum(-1, keepdim=True)
        s2 = xf.pow(2).pow(2).sum(-1, keepdim=True)
        s3 = xf.pow(3).pow(2).sum(-1, keepdim=True)
        s = _AllReduceSumSymmetric.apply(torch.cat([s1, s2, s3], dim=-1), self.tp_group)
        inv1 = torch.rsqrt(s[..., 0:1] / n_global + self.eps)
        inv2 = torch.rsqrt(s[..., 1:2] / n_global + self.eps)
        inv3 = torch.rsqrt(s[..., 2:3] / n_global + self.eps)
        # alpha is replicated across the group; all-reduce its gradient so the replicas stay
        # in sync (forward is identity, so the value is unchanged).
        alpha_1 = _SyncGradSum.apply(alpha_1.float(), self.tp_group)
        alpha_2 = _SyncGradSum.apply(alpha_2.float(), self.tp_group)
        alpha_3 = _SyncGradSum.apply(alpha_3.float(), self.tp_group)
        out = alpha_1 * (xf * inv1) + alpha_2 * (xf * xf * inv2) + alpha_3 * (xf * xf * xf * inv3)
        return out.to(input_dtype)


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
