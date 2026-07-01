# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
import megatron.core.optimizer.layer_wise_optimizer as layer_wise_module
import megatron.core.optimizer.md_decoupling as md_module
from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


requires_cuda_and_emerging = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_EMERGING_OPTIMIZERS,
    reason="CUDA and emerging_optimizers are required for MDDecoupling orthogonal updates",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this MDDecoupling test"
)


class _NoProcessGroups:
    tp = None
    expt_tp = None


def _step_sum_loss(model, input_tensor):
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()


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


def _mla_kv_up_proj_optimizer(param, **kwargs):
    return MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(1, 1),
        split_mla_per_head=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
        **kwargs,
    )


def _gqa_qkv_optimizer(param, **kwargs):
    return MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
        **kwargs,
    )


def _assert_split_flat_norms(optimizer, param, tensor, expected_norm, is_qkv=False):
    parts, _ = optimizer._split_param_tensor(param, tensor, is_qkv=is_qkv)
    expected = torch.full((len(parts),), expected_norm, dtype=tensor.dtype, device=tensor.device)
    actual = torch.stack([torch.linalg.vector_norm(part) for part in parts])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _assert_split_tangent(optimizer, param, grad, is_qkv):
    p_parts, _ = optimizer._split_param_tensor(param, param, is_qkv=is_qkv)
    g_parts, _ = optimizer._split_param_tensor(param, grad, is_qkv=is_qkv)
    residuals = torch.stack(
        [
            (p_part * g_part).sum().abs()
            / (torch.linalg.vector_norm(p_part) * torch.linalg.vector_norm(g_part)).clamp_min(1e-12)
            for p_part, g_part in zip(p_parts, g_parts)
        ]
    )
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), rtol=1e-5, atol=1e-6)


def _bare_param_from_gains(optimizer, param):
    state = optimizer.state[param]
    bare_param = param.detach().clone()
    if "flat_gain" in state:
        bare_param.div_(optimizer._phi(state["flat_gain"]))
    if "row_gain" in state:
        bare_param.div_(optimizer._phi(state["row_gain"])[:, None])
    if "col_gain" in state:
        bare_param.div_(optimizer._phi(state["col_gain"])[None, :])
    return bare_param


@requires_cuda_and_emerging
def test_md_decoupling_qkv_split():
    qkv_size = 3 * 8 * 4
    hidden_size = 64
    qkv_split_shapes = (8, 8, 8)

    torch.manual_seed(42)
    input_tensor = torch.randn(8, hidden_size, dtype=torch.float32, device="cuda")

    model_split = torch.nn.Linear(
        hidden_size, qkv_size, bias=False, dtype=torch.float32, device="cuda"
    )
    model_no_split = torch.nn.Linear(
        hidden_size, qkv_size, bias=False, dtype=torch.float32, device="cuda"
    )
    model_split.weight.data.fill_(1.0)
    model_no_split.weight.data.copy_(model_split.weight.data)
    model_split.weight.is_qkv = True

    optimizer_split = MDDecoupling(
        params=[model_split.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )
    optimizer_no_split = MDDecoupling(
        params=[model_no_split.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    original_weight = model_split.weight.data.clone()
    _step_sum_loss(model_split, input_tensor)
    optimizer_split.step()
    weight_with_split = model_split.weight.data.clone()

    _step_sum_loss(model_no_split, input_tensor)
    optimizer_no_split.step()
    weight_without_split = model_no_split.weight.data.clone()

    assert not torch.equal(weight_with_split, original_weight)
    assert not torch.equal(weight_without_split, original_weight)
    assert not torch.equal(weight_with_split, weight_without_split)


@requires_cuda_and_emerging
@pytest.mark.parametrize("tp_mode", ["duplicated", "blockwise", "distributed"])
def test_md_decoupling_different_tp_modes_single_rank(tp_mode):
    torch.manual_seed(42)
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device="cuda")
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        weight_decay=0.0,
        use_orthogonal_updates=True,
        momentum_beta=0.95,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=tp_mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)


@requires_cuda_and_emerging
@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMDDecouplingMultiRankTP:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        world = int(os.getenv("WORLD_SIZE", "1"))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, tp_mode):
        rank = int(os.getenv("RANK", "0"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device="cuda")
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = MDDecoupling(
            params=[model.weight],
            lr=0.01,
            weight_decay=0.0,
            use_orthogonal_updates=True,
            momentum_beta=0.95,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("tp_mode", ["duplicated", "distributed"])
    def test_md_decoupling_modes_multirank_update(self, tp_mode):
        model, optimizer = self.create_tp_model_and_optimizer(tp_mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
        original_weight = model.weight.data.clone()
        _step_sum_loss(model, input_tensor)
        optimizer.step()

        assert not torch.equal(model.weight.data, original_weight)

    def test_md_decoupling_blockwise_mode_multirank_update(self):
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
        original_weight = model.weight.data.clone()
        _step_sum_loss(model, input_tensor)
        optimizer.step()

        assert not torch.equal(model.weight.data, original_weight)


def test_md_decoupling_mla_split_flat_normalization_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(4, 8))
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=False)

    _assert_split_flat_norms(optimizer, param, param, expected_norm=8**0.5)
    torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(4.0))


@requires_cuda
def test_md_decoupling_mla_split_gains_step_preserves_bare_split_norms():
    param = torch.nn.Parameter(
        torch.arange(1, 33, dtype=torch.float32, device="cuda").view(4, 8)
    )
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_gains_mode="row",
        gains_lr=0.05,
        use_orthogonal_updates=False,
    )

    grad_scale = torch.linspace(0.1, 3.2, param.numel(), device=param.device).view_as(param)
    loss = (param * grad_scale).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    row_gain = state["row_gain"]
    assert row_gain.shape == (param.size(0),)
    assert torch.isfinite(param).all()
    assert torch.isfinite(row_gain).all()
    assert not torch.allclose(row_gain, torch.ones_like(row_gain))

    bare_param = param.detach() / optimizer._phi(row_gain)[:, None]
    _assert_split_flat_norms(optimizer, param, bare_param, expected_norm=8**0.5)


def test_md_decoupling_mla_split_tangential_grad_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(4, 8))
    param.is_kv_up_proj = True
    grad = torch.linspace(-1.5, 2.5, param.numel()).view_as(param)
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_tangential_grad=True,
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._project_tangent_inplace(param, grad, is_qkv=False)

    _assert_split_tangent(optimizer, param, grad, is_qkv=False)


def test_md_decoupling_gqa_qkv_split_mechanics():
    param = torch.nn.Parameter(torch.empty(16, 4))
    param.is_qkv = True
    grad = torch.arange(64, dtype=torch.float32).view(16, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
    )

    assert [call.shape for call in calls] == [
        torch.Size([8, 4]),
        torch.Size([4, 4]),
        torch.Size([4, 4]),
    ]
    expected = torch.tensor([1, 1, 1, 1, 2, 2, 3, 3] * 2).view(16, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_md_decoupling_gqa_split_tangential_grad_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 129, dtype=torch.float32).view(16, 8))
    param.is_qkv = True
    grad = torch.linspace(-3.5, 4.5, param.numel()).view_as(param)
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_tangential_grad=True,
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._project_tangent_inplace(param, grad, is_qkv=True)

    _assert_split_tangent(optimizer, param, grad, is_qkv=True)


def test_md_decoupling_mla_split_row_normalization():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(4, 8))
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="row",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=False)

    parts, _ = optimizer._split_param_tensor(param, param, is_qkv=False)
    for part in parts:
        torch.testing.assert_close(
            torch.linalg.vector_norm(part, dim=1),
            torch.ones(part.size(0), dtype=part.dtype),
            rtol=1e-5,
            atol=1e-5,
        )


def test_md_decoupling_gqa_split_row_normalization():
    param = torch.nn.Parameter(torch.arange(1, 129, dtype=torch.float32).view(16, 8))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="row",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    parts, _ = optimizer._split_param_tensor(param, param, is_qkv=True)
    for part in parts:
        torch.testing.assert_close(
            torch.linalg.vector_norm(part, dim=1),
            torch.ones(part.size(0), dtype=part.dtype),
            rtol=1e-5,
            atol=1e-5,
        )


@requires_cuda
@pytest.mark.parametrize(
    ("gain_mode", "expected_state_keys"),
    [
        ("flat", ("flat_gain",)),
        ("rowcol", ("row_gain", "col_gain")),
    ],
)
def test_md_decoupling_mla_split_gain_modes_preserve_bare_split_norms(
    gain_mode, expected_state_keys
):
    param = torch.nn.Parameter(
        torch.arange(1, 33, dtype=torch.float32, device="cuda").view(4, 8)
    )
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_gains_mode=gain_mode,
        gains_lr=0.05,
        gain_parametrization="softplus",
        use_orthogonal_updates=False,
    )

    grad_scale = torch.linspace(0.1, 3.2, param.numel(), device=param.device).view_as(param)
    loss = (param * grad_scale).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    for key in expected_state_keys:
        assert key in state
        assert torch.isfinite(state[key]).all()
        assert not torch.allclose(optimizer._phi(state[key]), torch.ones_like(state[key]))
    _assert_split_flat_norms(
        optimizer,
        param,
        _bare_param_from_gains(optimizer, param),
        expected_norm=8**0.5,
    )


@requires_cuda
@pytest.mark.parametrize(
    ("gain_mode", "expected_state_keys"),
    [
        ("flat", ("flat_gain",)),
        ("rowcol", ("row_gain", "col_gain")),
    ],
)
def test_md_decoupling_gqa_split_gain_modes_preserve_bare_split_norms(
    gain_mode, expected_state_keys
):
    param = torch.nn.Parameter(
        torch.arange(1, 129, dtype=torch.float32, device="cuda").view(16, 8)
    )
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_gains_mode=gain_mode,
        gains_lr=0.05,
        gain_parametrization="softplus",
        use_orthogonal_updates=False,
    )

    grad_scale = torch.linspace(0.1, 12.8, param.numel(), device=param.device).view_as(param)
    loss = (param * grad_scale).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    for key in expected_state_keys:
        assert key in state
        assert torch.isfinite(state[key]).all()
        assert not torch.allclose(optimizer._phi(state[key]), torch.ones_like(state[key]))
    _assert_split_flat_norms(
        optimizer,
        param,
        _bare_param_from_gains(optimizer, param),
        expected_norm=8**0.5,
        is_qkv=True,
    )


@requires_cuda_and_emerging
@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_md_decoupling_num_ns_steps(num_ns_steps):
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device="cuda")
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)


@requires_cuda_and_emerging
@pytest.mark.parametrize("use_nesterov", [True, False])
def test_md_decoupling_nesterov(use_nesterov):
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device="cuda")
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        momentum_beta=0.9,
        use_nesterov=use_nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)


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


def test_md_decoupling_mla_kv_up_proj_split_uses_local_dim0_tp_shapes():
    param = torch.nn.Parameter(torch.empty(10, 4))
    param.is_kv_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(40, dtype=torch.float32).view(10, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(12, 8),
    )

    assert [call.shape for call in calls] == [torch.Size([6, 4]), torch.Size([4, 4])]
    assert torch.equal(output[:6], torch.ones_like(output[:6]))
    assert torch.equal(output[6:], torch.full_like(output[6:], 2.0))


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


def test_md_decoupling_mla_qkv_down_proj_split_uses_local_dim0_tp_shapes():
    param = torch.nn.Parameter(torch.empty(6, 4))
    param.is_qkv_down_proj = True
    param.partition_dim = 0
    grad = torch.arange(24, dtype=torch.float32).view(6, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_down_proj_fn=lambda p: getattr(p, "is_qkv_down_proj", False),
        qkv_down_proj_split_shapes=(4, 8),
    )

    assert [call.shape for call in calls] == [torch.Size([2, 4]), torch.Size([4, 4])]
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


def test_md_decoupling_builder_tags_mla_and_gqa_parameters(monkeypatch):
    class _FakeModelChunk:
        def __init__(self):
            self.config = SimpleNamespace(
                num_attention_heads=8,
                num_query_groups=2,
                kv_channels=4,
                multi_latent_attention=True,
                qk_head_dim=6,
                v_head_dim=5,
                qk_pos_emb_head_dim=2,
                q_lora_rank=3,
                kv_lora_rank=7,
                num_layers=4,
            )
            self.qkv = torch.nn.Parameter(torch.ones(48, 5))
            self.kv_up = torch.nn.Parameter(torch.ones(88, 5))
            self.q_up = torch.nn.Parameter(torch.ones(64, 5))
            self.qkv_down = torch.nn.Parameter(torch.ones(12, 5))
            self.named = [
                ("decoder.layers.0.self_attention.linear_qkv.weight", self.qkv),
                ("decoder.layers.0.self_attention.linear_kv_up_proj.weight", self.kv_up),
                ("decoder.layers.0.self_attention.linear_q_up_proj.weight", self.q_up),
                (
                    "decoder.layers.0.self_attention.linear_qkv_down_proj.weight",
                    self.qkv_down,
                ),
            ]

        def named_parameters(self):
            return iter(self.named)

    class _FakeOptimizerWrapper:
        def __init__(self, optimizer, config, init_state_fn=None):
            del init_state_fn
            self.optimizer = optimizer
            self.config = config
            self.param_groups = optimizer.param_groups
            self.state = optimizer.state
            self.is_stub_optimizer = False

        def get_parameters(self):
            return [p for group in self.param_groups for p in group["params"]]

    def fake_get_param_groups(model_chunks, config, config_overrides):
        del config, config_overrides
        params = [
            p
            for model_chunk in model_chunks
            for _, p in model_chunk.named_parameters()
            if p.requires_grad
        ]
        return [{"params": params, "is_expert_parallel": False}]

    def fake_get_megatron_optimizer(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(chained_optimizers=[])

    monkeypatch.setattr(md_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(md_module, "FP32Optimizer", _FakeOptimizerWrapper)
    monkeypatch.setattr(md_module, "get_megatron_optimizer", fake_get_megatron_optimizer)

    model_chunk = _FakeModelChunk()
    config = OptimizerConfig(optimizer="md_decoupling", lr=0.01, min_lr=0.0)
    config.use_orthogonal_updates = False
    config.hypersphere_mode = "flat"
    config.hypersphere_embedding_mode = None
    config.hypersphere_router_mode = None
    config.hypersphere_gains_mode = None
    config.muon_split_qkv = True
    config.muon_split_mla_per_head = True
    config.use_distributed_optimizer = False
    config.fp16 = False
    config.bf16 = False

    chained = md_module.get_megatron_mddecoupling_optimizer(
        config,
        [model_chunk],
        config_overrides={},
        pg_collection=_NoProcessGroups(),
    )

    optimizer = chained.chained_optimizers[0].optimizer
    assert optimizer.qkv_split_shapes == [16, 4, 4]
    assert optimizer.kv_up_proj_split_shapes == (6, 5)
    assert optimizer.q_up_proj_head_dim == 8
    assert optimizer.qkv_down_proj_split_shapes == (3, 9)
    assert model_chunk.qkv.is_qkv
    assert model_chunk.kv_up.is_kv_up_proj
    assert model_chunk.q_up.is_q_up_proj
    assert model_chunk.qkv_down.is_qkv_down_proj


def test_md_decoupling_layerwise_preserves_mla_and_gqa_parameter_tags(monkeypatch):
    class _FakeOptimizer:
        def __init__(self, params):
            self.config = SimpleNamespace()
            self.param_groups = [{"params": params, "is_expert_parallel": False}]
            self.state = {}
            self.is_stub_optimizer = False

        def get_parameters(self):
            return [p for group in self.param_groups for p in group["params"]]

    monkeypatch.setattr(layer_wise_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(layer_wise_module, "get_pg_rank", lambda group: 0)

    qkv = torch.nn.Parameter(torch.ones(16, 4))
    qkv.is_qkv = True
    kv_up = torch.nn.Parameter(torch.ones(8, 4))
    kv_up.is_kv_up_proj = True
    q_up = torch.nn.Parameter(torch.ones(12, 4))
    q_up.is_q_up_proj = True
    qkv_down = torch.nn.Parameter(torch.ones(5, 4))
    qkv_down.is_qkv_down_proj = True

    optimizer = _FakeOptimizer([qkv, kv_up, q_up, qkv_down])
    config = SimpleNamespace(bf16=False)
    pg_collection = SimpleNamespace(dp_cp=object(), expt_dp=object())

    layerwise = LayerWiseDistributedOptimizer([optimizer], config, pg_collection)

    sharded_params = [p for shard in layerwise.dp_cp_params_list for p in shard]
    assert any(p is qkv for p in sharded_params) and qkv.is_qkv
    assert any(p is kv_up for p in sharded_params) and kv_up.is_kv_up_proj
    assert any(p is q_up for p in sharded_params) and q_up.is_q_up_proj
    assert any(p is qkv_down for p in sharded_params) and qkv_down.is_qkv_down_proj
