# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core import tensor_parallel
from megatron.core.optimizer.md_decoupling import MDDecoupling


def _record_md_split_output(param, grad, **md_kwargs):
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        pg_collection=None,
        tp_mode="duplicated",
        **md_kwargs,
    )
    calls = []

    def record_split(split_grad, tp_group, partition_dim):
        del tp_group, partition_dim
        calls.append(split_grad.detach().clone())
        return torch.full_like(split_grad, float(len(calls)))

    optimizer._orthogonalize_tensor = record_split
    return optimizer._orthogonalize_param(
        param, grad, is_qkv=getattr(param, "is_qkv", False)
    ), calls


def test_md_decoupling_mla_kv_up_proj_split():
    param = torch.nn.Parameter(torch.empty(14, 4))
    param.is_kv_up_proj = True
    grad = torch.arange(56, dtype=torch.float32).view(14, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(8, 6),
    )

    assert [call.shape for call in calls] == [torch.Size([8, 4]), torch.Size([6, 4])]
    assert torch.equal(output[:8], torch.ones_like(output[:8]))
    assert torch.equal(output[8:], torch.full_like(output[8:], 2.0))


def test_md_decoupling_mla_kv_up_proj_split_per_head():
    param = torch.nn.Parameter(torch.empty(10, 4))
    param.is_kv_up_proj = True
    grad = torch.arange(40, dtype=torch.float32).view(10, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(3, 2),
        split_mla_per_head=True,
    )

    assert [call.shape for call in calls] == [torch.Size([6, 4]), torch.Size([4, 4])]
    expected = torch.tensor([1, 1, 1, 2, 2, 1, 1, 1, 2, 2]).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_md_decoupling_mla_q_up_proj_split_per_head():
    param = torch.nn.Parameter(torch.empty(12, 4))
    param.is_q_up_proj = True
    grad = torch.arange(48, dtype=torch.float32).view(12, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_q_up_proj_fn=lambda p: getattr(p, "is_q_up_proj", False),
        q_up_proj_head_dim=4,
        split_mla_per_head=True,
    )

    assert [call.shape for call in calls] == [
        torch.Size([4, 4]),
        torch.Size([4, 4]),
        torch.Size([4, 4]),
    ]
    expected = torch.tensor([1] * 4 + [2] * 4 + [3] * 4).view(12, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_md_decoupling_mla_qkv_down_proj_split_mechanics():
    param = torch.nn.Parameter(torch.empty(5, 4))
    param.is_qkv_down_proj = True
    grad = torch.arange(20, dtype=torch.float32).view(5, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_down_proj_fn=lambda p: getattr(p, "is_qkv_down_proj", False),
        qkv_down_proj_split_shapes=(2, 3),
    )

    assert [call.shape for call in calls] == [torch.Size([2, 4]), torch.Size([3, 4])]
    assert torch.equal(output[:2], torch.ones_like(output[:2]))
    assert torch.equal(output[2:], torch.full_like(output[2:], 2.0))


def test_md_decoupling_mla_param_tags_copy_to_main_param():
    param = torch.empty(2, 2)
    tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    param.is_kv_up_proj = True
    param.is_q_up_proj = True
    param.is_qkv_down_proj = True
    main_param = torch.empty_like(param)

    tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)

    assert main_param.is_kv_up_proj
    assert main_param.is_q_up_proj
    assert main_param.is_qkv_down_proj
