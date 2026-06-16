# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def compiled_gated_polynorm(x, alpha_1, alpha_2, eps: float = 1e-6):
    """Core Gated PolyNorm computation: ``alpha_1 * RMSNorm(x) + alpha_2 * RMSNorm(x**2)``.

    The RMS normalization is taken over the last (feature) dimension. The math is done in
    fp32 for numerical stability and cast back to the input dtype, mirroring how the
    RMSNorm/LayerNorm layers in this codebase behave under mixed precision.

    ``alpha_1``/``alpha_2`` broadcast against ``x``. They are either a single (broadcastable)
    coefficient of shape ``(1,)`` (dense / single-expert case) or per-token coefficients of
    shape ``(num_tokens, 1)`` (grouped-expert case, where each token already carries the
    coefficient of the expert it was routed to).
    """
    input_dtype = x.dtype
    x = x.float()

    def norm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    out = alpha_1 * norm(x) + alpha_2 * norm(x * x)
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


class GatedPolyNorm(MegatronModule):
    """Learnable Gated PolyNorm activation — a drop-in replacement for the gate of a gated
    linear unit (e.g. SiLU in SwiGLU).

    In a GLU the first linear layer produces ``[x_glu, x_linear]`` and the block output is
    ``gate(x_glu) * x_linear``. Standard SwiGLU uses ``gate = SiLU``. Here the gate is::

        gate(x) = |alpha_1| * RMSNorm(x) + |alpha_2| * RMSNorm(x ** 2)

    where ``alpha_1``/``alpha_2`` are learnable (``abs`` keeps them positive).

    To support grouped MoE experts (where the activations of all local experts are
    concatenated along the token dimension and processed in a single call) this module holds
    one ``(alpha_1, alpha_2)`` pair *per local expert*: ``alpha_p1``/``alpha_p2`` have shape
    ``(num_local_experts,)``. When ``tokens_per_expert`` is supplied the per-expert
    coefficients are expanded to per-token coefficients, so every token is gated by the
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
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.eps = eps
        # The group over which the ffn feature dimension is sharded. tp_size==1 (no sharding,
        # e.g. local CPU runs or ETP=1 experts) takes the cheap fused path with no collectives.
        self.tp_group = tp_group
        if tp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.tp_size = torch.distributed.get_world_size(group=tp_group)
        else:
            self.tp_size = 1

    def forward(self, x, tokens_per_expert=None):
        # Keep the coefficients positive.
        alpha_p1 = torch.abs(self.alpha_p1)  # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    "GatedPolyNorm with num_local_experts > 1 requires `tokens_per_expert` so "
                    "the per-expert coefficients can be mapped onto the concatenated tokens."
                )
            # Single coefficient broadcast to every token.
            alpha_p1_t = alpha_p1
            alpha_p2_t = alpha_p2
        else:
            # Expand per-expert coefficients to per-token coefficients.
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            alpha_p1_t = torch.repeat_interleave(alpha_p1, tpe_tensor).unsqueeze(-1)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)

        if self.tp_size == 1:
            # ffn feature dim is whole on this rank: cheap fused per-token norm.
            return compiled_gated_polynorm(x, alpha_p1_t, alpha_p2_t, self.eps)
        # ffn feature dim is TP-sharded: reduce the feature statistics across the group.
        return self._tp_forward(x, alpha_p1_t, alpha_p2_t)

    def _tp_forward(self, x, alpha_1, alpha_2):
        """TP-invariant path: recover the full-feature RMS from the local feature shards."""
        input_dtype = x.dtype
        xf = x.float()
        # Each ColumnParallel rank holds an equal 1/tp_size slice of the ffn features.
        n_global = xf.shape[-1] * self.tp_size
        # Per-token partial feature sums on this rank: sum(x^2) for RMSNorm(x) and sum(x^4)
        # (== sum((x^2)^2)) for RMSNorm(x^2). One symmetric all-reduce completes both.
        s1 = xf.pow(2).sum(-1, keepdim=True)
        s2 = xf.pow(2).pow(2).sum(-1, keepdim=True)
        s = _AllReduceSumSymmetric.apply(torch.cat([s1, s2], dim=-1), self.tp_group)
        inv1 = torch.rsqrt(s[..., 0:1] / n_global + self.eps)
        inv2 = torch.rsqrt(s[..., 1:2] / n_global + self.eps)
        # alpha is replicated across the group; all-reduce its gradient so the replicas stay
        # in sync (forward is identity, so the value is unchanged).
        alpha_1 = _SyncGradSum.apply(alpha_1.float(), self.tp_group)
        alpha_2 = _SyncGradSum.apply(alpha_2.float(), self.tp_group)
        out = alpha_1 * (xf * inv1) + alpha_2 * (xf * xf * inv2)
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
