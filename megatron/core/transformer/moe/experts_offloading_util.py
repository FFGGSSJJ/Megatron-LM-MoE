# Copyright (c) 2026, Swiss AI Institute
"""
This module implements utility classes and functions for Mixture of Experts (MoE) in the Megatron-LM framework, including:
1) ExpertsWgradScheduler: a utility class to manage the scheduling of weight gradient computations for MoE experts, allowing for delayed computation of weight gradients to enable better interleaving of GPU computation and CPU-GPU communication.
2) MergedSwiGLU: a custom autograd function that implements the forward and backward pass of the SwiGLU activation function, with optional probability scaling for the forward pass and corresponding adjustments in the backward pass.
3) GroupedSwiMLP: a custom autograd function that implements the forward and backward pass of a grouped linear layer followed by a SwiGLU activation and another grouped linear layer, with support for delayed weight gradient computation and optional FP8 activation quantization for memory efficiency.
"""
from __future__ import annotations
import torch
import collections
import queue
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch import (
        Float8BlockQuantizer,
    )
    from transformer_engine.pytorch.constants import TE_DType
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

class ExpertsWgradScheduler:
    def __init__(self, delay_wgrad_compute: bool = False):
        self.delay_wgrad_compute = delay_wgrad_compute
        self.queue = queue.Queue()

    def register(self, grad_func, *grad_parms):
        if self.delay_wgrad_compute:
            self.queue.put((grad_func, grad_parms))

    def pop_callback(self):
        if self.queue.qsize() > 0 and self.delay_wgrad_compute:
            grad_func, grad_parms = self.queue.get()
            return grad_func(*grad_parms)
        else:
            # If there is no token assigned to the expert in this MoE layer,
            # then there will be case that the wgrad compute is not registered
            return

class MergedSwiGLU(torch.autograd.Function):
    """Re-implementation of Silu
    """

    @classmethod
    @torch.compile()
    def call_forward(
        cls,
        input_tensor: torch.Tensor,
        probs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """forward with optional probability scaling for SwiGLU activation. 
        If `probs` is provided, it will be used to scale the output of the SwiGLU activation, 
        otherwise it will compute the standard SwiGLU activation without scaling.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor | None, optional): Defaults to None.

        Returns:
            torch.Tensor: activation output
        """
        if probs is not None:
            return MergedSwiGLU.call_forward_silu_probs(input_tensor, probs)
        else:
            return MergedSwiGLU.call_forward_silu(input_tensor)

    @classmethod
    @torch.compile()
    def call_forward_silu(
        cls,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """forward pass for SwiGLU activation without probability scaling.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function

        Returns:
            torch.Tensor: activation output
        """
        a, b = input_tensor.chunk(2, dim=-1)
        return (torch.nn.functional.silu(a) * b).to(input_tensor.dtype)
    
    @classmethod
    @torch.compile()
    def call_forward_silu_probs(
        cls,
        input_tensor: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """actual forward function with probability.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor): probability derived from router

        Returns:
            torch.Tensor: activation output
        """
        a, b = input_tensor.chunk(2, dim=-1)
        return ((torch.nn.functional.silu(a) * b) * probs).to(input_tensor.dtype)
    
    @classmethod
    @torch.compile()
    def call_backward(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor,
        probs: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None]:
        """backward function for SwiGLU activation with optional probability scaling.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor | None, optional): Defaults to None.
        """
        if probs is not None:
            return MergedSwiGLU.call_backward_silu_probs(grad_output, input_tensor, probs)
        else:
            return MergedSwiGLU.call_backward_silu(grad_output, input_tensor)
    
    @classmethod
    @torch.compile()
    def call_backward_silu(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """actual backward function without probability.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function

        Returns:
            torch.Tensor: gradient of the input tensor
        """
        a, b = input_tensor.chunk(2, dim=-1)
        sigmoid_a = torch.sigmoid(a)
        ones = torch.ones(sigmoid_a.shape, device=sigmoid_a.device, dtype=sigmoid_a.dtype)
        grad_a = grad_output * (sigmoid_a + a * sigmoid_a * (ones - sigmoid_a)) * b
        grad_b = grad_output * torch.nn.functional.silu(a)
        return torch.cat([grad_a, grad_b], dim=-1)
    
    @classmethod
    @torch.compile()
    def call_backward_silu_probs(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor,
        probs: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """actual backward function with probability. 
        It computes the gradient of the input tensor and the probability.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor): probability derived from router
        """
        input_grad = MergedSwiGLU.call_backward_silu(
            grad_output * probs, 
            input_tensor
        )
        weights_grad = MergedSwiGLU.call_forward_silu(
            input_tensor
        ) * grad_output.to(probs.dtype)
        weights_grad = torch.sum(
            weights_grad, dim=-1
        )

        return input_grad.to(input_tensor.dtype) if input_grad is not None else None, \
        weights_grad.to(probs.dtype) if weights_grad is not None else None
    
    @staticmethod
    def forward(
        ctx,
        *args,
    ):
        args_q = collections.deque(args)
        input_tensor: torch.Tensor = args_q.popleft()
        probs: torch.Tensor | None = args_q.popleft()

        ctx.save_for_backward(input_tensor, probs)
        return MergedSwiGLU.call_forward(input_tensor, probs)
    
    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        grad_y: torch.Tensor = grad_outputs[0]
        (x, probs) = ctx.saved_tensors
        input_grad, prob_grad = MergedSwiGLU.call_backward(
            grad_y, x, probs
        )
        if prob_grad is not None:
            prob_grad = prob_grad.unsqueeze(-1)
        return input_grad, prob_grad
        

def release(t: torch.Tensor):
    """Helper function to release tensors that are no longer needed to save memory.
    """
    t.untyped_storage().resize_(0)

class StreamManager:
    """Manage CUDA streams and events shared by MoE offloading paths."""

    _instance = None

    def __init__(
        self,
        num_h2d_streams,
        num_compute_streams=4,
    ):
        self.num_compute_streams = num_compute_streams
        self.num_h2d_streams = num_h2d_streams
        self.h2d_streams = [torch.cuda.Stream() for _ in range(num_h2d_streams)]
        self.compute_streams = [torch.cuda.Stream() for _ in range(self.num_compute_streams)]
        self.compute_cuda_streams = [stream.cuda_stream for stream in self.compute_streams]

        # Dedicated copy streams for activation offload D2H/H2D.
        self.act_d2h_stream = torch.cuda.Stream()
        self.act_h2d_stream = torch.cuda.Stream()

    @classmethod
    def get_instance(
        cls,
        num_h2d_streams=2,
        num_compute_streams=4,
    ):
        if cls._instance is None:
            cls._instance = StreamManager(num_h2d_streams, num_compute_streams)
        return cls._instance

    def get_h2d_stream(self, idx) -> torch.cuda.Stream:
        return self.h2d_streams[idx]

    def get_compute_streams(self) -> list[int]:
        return self.compute_cuda_streams

    def get_launch_streams(self) -> list[torch.cuda.Stream]:
        # VPP can execute a model chunk on a non-default current stream.
        current_stream = torch.cuda.current_stream()
        default_stream = torch.cuda.default_stream()
        if current_stream.cuda_stream == default_stream.cuda_stream:
            return [current_stream]
        return [current_stream, default_stream]

    def launch_streams_wait_compute_streams(self):
        launch_streams = self.get_launch_streams()
        for i in range(self.num_compute_streams):
            for launch_stream in launch_streams:
                launch_stream.wait_stream(self.compute_streams[i])

    def default_stream_wait_h2d_stream(self, idx):
        torch.cuda.default_stream().wait_stream(self.get_h2d_stream(idx))

    def compute_streams_wait_launch_streams(self):
        launch_streams = self.get_launch_streams()
        for i in range(self.num_compute_streams):
            for launch_stream in launch_streams:
                self.compute_streams[i].wait_stream(launch_stream)

    def h2d_stream_wait_consumer_streams(self, idx):
        h2d_stream = self.get_h2d_stream(idx)
        for launch_stream in self.get_launch_streams():
            h2d_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            h2d_stream.wait_stream(self.compute_streams[i])

    def compute_streams_wait_h2d_stream(self, idx):
        h2d_stream = self.get_h2d_stream(idx)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_stream(h2d_stream)

    def consumer_streams_wait_event(self, event):
        for launch_stream in self.get_launch_streams():
            launch_stream.wait_event(event)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_event(event)

    def h2d_stream_wait_default_stream(self, idx):
        self.get_h2d_stream(idx).wait_stream(torch.cuda.default_stream())

    def act_d2h_stream_wait_producers(self):
        """Make the activation-offload D2H stream wait for activation producers."""
        for launch_stream in self.get_launch_streams():
            self.act_d2h_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            self.act_d2h_stream.wait_stream(self.compute_streams[i])

    def consumer_streams_wait_act_reload(self, h2d_done_event):
        """Make backward consumer streams wait until activation reload H2D completes."""
        self.consumer_streams_wait_event(h2d_done_event)


_dummy_wgrads = {}

def get_dummy_wgrad(
    shape: list,
    dtype: torch.dtype,
    device,
    zero=False
) -> torch.Tensor:
    """Returns a dummy tensor of given shape."""
    global _dummy_wgrads
    wgard_key = (*shape, dtype)
    if wgard_key not in _dummy_wgrads:
        _dummy_wgrads[wgard_key] = torch.empty(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[wgard_key].fill_(0)
    return _dummy_wgrads[wgard_key].detach()

def grouped_swiglu_mlp_torch_ref(
    w1,
    w2,
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: Optional[ExpertsWgradScheduler] = None,
    config: Optional["TransformerConfig"] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference path for the grouped SwiGLU MoE experts.

    Drop-in replacement for ``grouped_swiglu_mlp`` used only to verify
    correctness. Each expert is evaluated with plain ``torch.matmul`` and the
    backward pass is handled entirely by autograd -- no grouped GEMM, no custom
    CUDA streams / weight-prefetch buffers, and no manual ``main_grad`` writes.
    Expert-weight gradients reach ``main_grad`` through the standard DDP backward
    hook, exactly like an ordinary linear layer. This isolates whether a failure
    lives in the grouped-GEMM / offloading machinery or in the surrounding
    dispatcher / VPP wiring.

    Per expert ``i`` (``x_i: [t_i, in]``, ``w1[i]: [in, 2H]``, ``w2[i]: [H, in]``):

        fc1 = x_i @ w1[i]                 # [t_i, 2H]
        gate, lin = fc1.chunk(2, dim=-1)
        s = (silu(gate) * lin) * probs_i  # [t_i, H]
        y_i = s @ w2[i]                   # [t_i, in]

    ``expert_wgrad_scheduler`` and ``config`` are accepted only for signature
    compatibility with ``grouped_swiglu_mlp`` and are unused.
    """
    # Normalize weights to a list of per-expert 2D tensors (supports both the
    # per-expert parameter list and a stacked [E, in, 2H] / [E, H, in] tensor).
    w1_list = list(torch.unbind(w1, dim=0)) if isinstance(w1, torch.Tensor) else list(w1)
    w2_list = list(torch.unbind(w2, dim=0)) if isinstance(w2, torch.Tensor) else list(w2)

    # torch.split needs python ints; .tolist() syncs if tokens_per_expert is on GPU.
    tokens = (
        tokens_per_expert.tolist()
        if isinstance(tokens_per_expert, torch.Tensor)
        else list(tokens_per_expert)
    )

    x_chunks = torch.split(permuted_local_hidden_states, tokens, dim=0)
    probs_chunks = torch.split(permuted_probs.reshape(-1), tokens, dim=0)

    outputs = []
    for i in range(num_local_experts):
        x_i = x_chunks[i]
        fc1 = torch.matmul(x_i, w1_list[i])                       # [t_i, 2H]
        gate, lin = fc1.chunk(2, dim=-1)
        s = F.silu(gate) * lin                                    # [t_i, H]
        s = (s * probs_chunks[i].unsqueeze(-1)).to(x_i.dtype)
        outputs.append(torch.matmul(s, w2_list[i]))               # [t_i, in]

    return torch.cat(outputs, dim=0)