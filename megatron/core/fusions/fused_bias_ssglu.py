# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

###### BIAS SSGLU FUSION/ NO AUTOGRAD ################
# SSGLU is SwiGLU with the sigmoid inside SiLU replaced by softsign rescaled to (0, 1):
#   gate(x) = 0.5 * (1 + softsign(x)) = 0.5 + 0.5 * x / (1 + |x|)
#   sslu(x) = x * gate(x)
#   SSGLU(y_1, y_2) = sslu(y_1) * y_2
# Built exactly like the SwiGLU fusion in fused_bias_swiglu.py (@jit_fuser forward/backward
# pairs wrapped in torch.autograd.Function), since the gate has no cross-feature reduction.


def sslu(y):
    """Scaled-softsign counterpart of SiLU: ``y * (0.5 + 0.5 * softsign(y))``.

    Also serves as the ``config.activation_func`` identity that routes the MLP/TEGroupedMLP
    forward passes to the fused SSGLU kernels below (mirroring how ``F.silu`` routes to the
    fused SwiGLU kernels), and as the gate function for the unfused generic GLU fallback.
    """
    return y * (0.5 + 0.5 * y / (1 + torch.abs(y)))


@jit_fuser
def ssglu(y):
    """Performs SSGLU (Scaled-SoftSign-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SSGLU activation: sslu(y1) * y2, where y1, y2 are the
            split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (y_1 * (0.5 + 0.5 * y_1 / (1 + torch.abs(y_1)))) * y_2


@jit_fuser
def bias_ssglu(y, bias):
    """Performs SSGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SSGLU activation.
    """
    y = y + bias
    return ssglu(y)


@jit_fuser
def weighted_ssglu(y, weights):
    dtype = y.dtype
    res = ssglu(y) * weights
    return res.to(dtype)


@jit_fuser
def ssglu_back(g, y):
    """Computes the gradient for the SSGLU activation function.

    With gate(x) = 0.5 + 0.5 * x / (1 + |x|) and gate'(x) = 0.5 / (1 + |x|)**2:
    d/dx sslu(x) = gate(x) + x * gate'(x).

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    denom = 1 + torch.abs(y_1)
    gate = 0.5 + 0.5 * y_1 / denom
    d_sslu = gate + 0.5 * y_1 / (denom * denom)
    return torch.cat((g * d_sslu * y_2, g * (y_1 * gate)), -1)


@jit_fuser
def bias_ssglu_back(g, y, bias):
    """Computes the gradient for the biased SSGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return ssglu_back(g, y)


@jit_fuser
def weighted_ssglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = ssglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = ssglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasSSGLUFunction(torch.autograd.Function):
    """Custom autograd function for SSGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased SSGLU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_ssglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SSGLU activation."""
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_ssglu_back(grad_output, input, bias)
        return tmp, tmp, None, None


class SSGLUFunction(torch.autograd.Function):
    """Custom autograd function for SSGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of SSGLU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return ssglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SSGLU activation."""
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = ssglu_back(grad_output, input)
        return tmp, None, None


class WeightedSSGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_ssglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_ssglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_ssglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased SSGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SSGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SSGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SSGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SSGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSSGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SSGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_ssglu_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias ssglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted ssglu fusion")
    else:
        output = WeightedSSGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
