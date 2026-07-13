# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

###### BIAS RLGLU FUSION/ NO AUTOGRAD ################
# RLGLU is a gated linear unit whose gate is the "relu minus log" function
#   gate(x)  = relu(x) - 0.5 * ln(1 + |x|)                       (== rlglu_act in activations.py)
#   RLGLU(y_1, y_2) = gate(y_1) * y_2
# Unlike SwiGLU/SSSGLU the gate is NOT of the form ``x * squash(x)`` -- it is the gate itself, so
# there is no separate SiLU-style helper. The key property that makes the fused backward cheap:
#   gate'(x) = 0.5 + 0.5 * x / (1 + |x|)   (softsign rescaled to (0, 1))
# i.e. RLGLU's gate derivative is EXACTLY the SSSGLU gate. So the backward reuses that expression.
# Built exactly like the SwiGLU/SSSGLU fusions (fused_bias_swiglu.py / fused_bias_sssglu.py):
# @jit_fuser forward/backward pairs wrapped in torch.autograd.Function, since the gate has no
# cross-feature reduction. The gate math is inlined here (rather than importing rlglu_act) to keep
# the jit_fuser scripting self-contained; it must stay in sync with rlglu_act in activations.py.


@jit_fuser
def rlglu(y):
    """Performs RLGLU (Relu-Log-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of RLGLU activation: gate(y1) * y2, where y1, y2 are the split
            halves and gate(x) = relu(x) - 0.5 * ln(1 + |x|).
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (torch.relu(y_1) - 0.5 * torch.log1p(torch.abs(y_1))) * y_2


@jit_fuser
def bias_rlglu(y, bias):
    """Performs RLGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by RLGLU activation.
    """
    y = y + bias
    return rlglu(y)


@jit_fuser
def weighted_rlglu(y, weights):
    dtype = y.dtype
    res = rlglu(y) * weights
    return res.to(dtype)


@jit_fuser
def rlglu_back(g, y):
    """Computes the gradient for the RLGLU activation function.

    With gate(x) = relu(x) - 0.5 * ln(1 + |x|), the gate derivative is the SSSGLU gate:
        gate'(x) = 0.5 + 0.5 * x / (1 + |x|)   (softsign rescaled to (0, 1)).
    So d/dy1 [gate(y1) * y2] = gate'(y1) * y2 and d/dy2 [gate(y1) * y2] = gate(y1).

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    gate = torch.relu(y_1) - 0.5 * torch.log1p(torch.abs(y_1))
    gate_prime = 0.5 + 0.5 * y_1 / (1 + torch.abs(y_1))
    return torch.cat((g * gate_prime * y_2, g * gate), -1)


@jit_fuser
def bias_rlglu_back(g, y, bias):
    """Computes the gradient for the biased RLGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return rlglu_back(g, y)


@jit_fuser
def weighted_rlglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = rlglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = rlglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasRLGLUFunction(torch.autograd.Function):
    """Custom autograd function for RLGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased RLGLU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_rlglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased RLGLU activation."""
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_rlglu_back(grad_output, input, bias)
        return tmp, tmp, None, None


class RLGLUFunction(torch.autograd.Function):
    """Custom autograd function for RLGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of RLGLU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return rlglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of RLGLU activation."""
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = rlglu_back(grad_output, input)
        return tmp, None, None


class WeightedRLGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_rlglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_rlglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_rlglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased RLGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the RLGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply RLGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free RLGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased RLGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasRLGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = RLGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_rlglu_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias rlglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted rlglu fusion")
    else:
        output = WeightedRLGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
