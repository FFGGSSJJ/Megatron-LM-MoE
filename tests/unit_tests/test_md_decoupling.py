# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.optimizer.md_decoupling import _get_muon_scale_factor
from megatron.core.optimizer.md_decoupling import _split_qkv
from megatron.core.optimizer.md_decoupling import get_megatron_mddecoupling_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


requires_cuda_and_emerging = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_EMERGING_OPTIMIZERS,
    reason="CUDA and emerging_optimizers are required for MDDecoupling orthogonal updates",
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

    def record_split(split_grad, tp_group, partition_dim, flat_mode=False, is_router=False):
        del tp_group, partition_dim, flat_mode, is_router
        calls.append(split_grad.detach().clone())
        return torch.full_like(split_grad, float(len(calls)))

    optimizer._orthogonalize_tensor = record_split
    return optimizer._orthogonalize_param(
        param, grad, is_qkv=getattr(param, "is_qkv", False)
    ), calls


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


def _assert_qkv_split_flat_norms(optimizer, tensor, expected_norm):
    parts = _split_qkv(tensor, optimizer.qkv_split_shapes)
    expected = torch.full((len(parts),), expected_norm, dtype=tensor.dtype, device=tensor.device)
    actual = torch.stack([torch.linalg.vector_norm(part) for part in parts])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _assert_qkv_split_tangent(optimizer, param, grad):
    p_parts = _split_qkv(param, optimizer.qkv_split_shapes)
    g_parts = _split_qkv(grad, optimizer.qkv_split_shapes)
    residuals = torch.stack(
        [
            (p_part * g_part).sum().abs()
            / (torch.linalg.vector_norm(p_part) * torch.linalg.vector_norm(g_part)).clamp_min(1e-12)
            for p_part, g_part in zip(p_parts, g_parts)
        ]
    )
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), rtol=1e-5, atol=1e-6)


class _TinyMDDecouplingModel(torch.nn.Module):
    def __init__(self, shared_output=False, offloading_expert=False, device="cpu"):
        super().__init__()
        self.config = SimpleNamespace(
            context_parallel_size=1,
            hidden_size=8,
            kv_channels=4,
            moe_use_inplace_fp8_param=offloading_expert,
            moe_use_offloading_experts=offloading_expert,
            num_attention_heads=2,
            num_layers=1,
            num_query_groups=1,
        )
        self.ddp_config = SimpleNamespace(
            num_distributed_optimizer_instances=1,
            use_distributed_optimizer=False,
            use_megatron_fsdp=False,
        )
        self.embedding = torch.nn.Module()
        self.embedding.word_embeddings = torch.nn.Embedding(8, 8, device=device)
        self.output_layer = torch.nn.Linear(8, 8, bias=False, device=device)
        self.router = torch.nn.Linear(8, 4, bias=False, device=device)
        self.attn = torch.nn.Module()
        self.attn.linear_qkv = torch.nn.Linear(8, 24, bias=False, device=device)
        self.mlp = torch.nn.Module()
        self.mlp.linear_fc2 = torch.nn.Linear(8, 8, bias=False, device=device)
        self.norm = torch.nn.LayerNorm(8, device=device)
        if offloading_expert:
            self.experts = torch.nn.Module()
            self.experts.weight2 = torch.nn.Parameter(torch.ones(2, 8, 8, device=device))

        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        self.output_layer.weight.is_embedding_or_output_parameter = True
        if shared_output:
            self.output_layer.weight.shared_embedding = True


def test_md_decoupling_recipe_defaults():
    config = OptimizerConfig()

    assert config.hypersphere_mode == "flat"
    assert config.hypersphere_embedding_mode == "row"
    assert config.hypersphere_router_mode == "row"
    assert config.hypersphere_radius_from_init is False
    assert config.hypersphere_gains_mode == "rowcol"
    assert config.hypersphere_gains_mode_output == "inherit"
    assert config.hypersphere_gains_mode_embedding == "none"
    assert config.hypersphere_gains_mode_router == "rowcol"
    assert config.use_orthogonal_updates is True
    assert config.gain_parametrization == "softplus"
    assert config.muon_router_scale_mode == "none"


def test_md_decoupling_router_scale_mode_resolution():
    # Default: routers get "none" (constant 1.0) while matrices follow scale_mode. A router of
    # shape (num_experts, hidden) is non-square, so shape_up would give >1; "none" pins it to 1.
    optimizer = MDDecoupling(
        params=[torch.nn.Parameter(torch.ones(2, 2))],
        lr=0.01,
        scale_mode="shape_up",
        pg_collection=None,
    )
    assert optimizer.router_scale_mode == "none"
    assert optimizer._resolve_scale_mode(is_router=True) == "none"
    assert optimizer._resolve_scale_mode(is_router=False) == "shape_up"
    # The resolved router mode yields a constant 1.0 regardless of the router's aspect ratio,
    # so the update is width-invariant (num_experts=128 fixed while hidden scales).
    for hidden in (128, 768, 1536):
        assert _get_muon_scale_factor(
            128, hidden, mode=optimizer._resolve_scale_mode(is_router=True)
        ) == 1.0
    # shape_up on a router WOULD track width — this is the behavior being excluded.
    assert _get_muon_scale_factor(128, 768, mode="shape_up") > 1.0

    # Override: routers can be made to follow a mode explicitly.
    overridden = MDDecoupling(
        params=[torch.nn.Parameter(torch.ones(2, 2))],
        lr=0.01,
        scale_mode="shape_up",
        router_scale_mode="spectral",
        pg_collection=None,
    )
    assert overridden._resolve_scale_mode(is_router=True) == "spectral"
    assert overridden._resolve_scale_mode(is_router=False) == "shape_up"


def test_md_decoupling_router_gains_mode_override():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_router = True
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_router="none",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(param) == "none"


def test_md_decoupling_default_gains_mode_resolution():
    normal = torch.nn.Parameter(torch.ones(2, 2))
    embedding = torch.nn.Parameter(torch.ones(2, 2))
    output = torch.nn.Parameter(torch.ones(2, 2))
    router = torch.nn.Parameter(torch.ones(2, 2))
    embedding.is_md_embedding_parameter = True
    output.is_md_output_parameter = True
    router.is_router = True
    optimizer = MDDecoupling(
        params=[normal, embedding, output, router],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_output="inherit",
        hypersphere_gains_mode_embedding="none",
        hypersphere_gains_mode_router="rowcol",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(normal) == "rowcol"
    assert optimizer._resolve_gains_mode(embedding) == "none"
    assert optimizer._resolve_gains_mode(output) == "rowcol"
    assert optimizer._resolve_gains_mode(router) == "rowcol"


def test_md_decoupling_gains_mode_none_disables_gain_state():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_router = True
    param.grad = torch.ones_like(param)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="none",
        hypersphere_gains_mode_router="rowcol",
        use_orthogonal_updates=False,
        pg_collection=None,
    )

    optimizer.step()

    gain_state_keys = {
        "flat_gain",
        "flat_gain_m",
        "flat_gain_v",
        "row_gain",
        "row_gain_m",
        "row_gain_v",
        "col_gain",
        "col_gain_m",
        "col_gain_v",
    }
    assert optimizer.hypersphere_gains_mode is None
    assert optimizer._resolve_gains_mode(param) is None
    assert gain_state_keys.isdisjoint(optimizer.state[param])


def test_md_decoupling_embedding_gain_override_wins_for_tied_output():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_md_embedding_parameter = True
    param.is_md_output_parameter = True
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_embedding="none",
        hypersphere_gains_mode_output="flat",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(param) == "none"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optimizer wrapper creates CUDA scale")
def test_md_decoupling_builder_tags_embedding_output_and_shared_output():
    Utils.initialize_model_parallel()
    try:
        untied_model = _TinyMDDecouplingModel(shared_output=False, device="cuda")
        shared_model = _TinyMDDecouplingModel(shared_output=True, device="cuda")
        offload_model = _TinyMDDecouplingModel(
            offloading_expert=True, device="cuda"
        ).bfloat16()
        config = OptimizerConfig(
            optimizer="md_decoupling",
            lr=0.01,
            min_lr=0.0,
            use_orthogonal_updates=False,
        )
        offload_config = OptimizerConfig(
            optimizer="md_decoupling",
            lr=0.01,
            min_lr=0.0,
            bf16=True,
            use_orthogonal_updates=False,
        )

        optimizer = get_megatron_mddecoupling_optimizer(
            config,
            [untied_model],
            use_gloo_process_groups=False,
        )
        shared_optimizer = get_megatron_mddecoupling_optimizer(
            config,
            [shared_model],
            use_gloo_process_groups=False,
        )
        offload_optimizer = get_megatron_mddecoupling_optimizer(
            offload_config,
            [offload_model],
            use_gloo_process_groups=False,
        )
        md_optimizer = optimizer.chained_optimizers[0].optimizer
        shared_md_optimizer = shared_optimizer.chained_optimizers[0].optimizer
        offload_md_optimizer = offload_optimizer.chained_optimizers[0].optimizer

        assert untied_model.embedding.word_embeddings.weight.is_md_embedding_parameter is True
        assert not hasattr(untied_model.embedding.word_embeddings.weight, "is_md_output_parameter")
        assert untied_model.output_layer.weight.is_md_output_parameter is True
        assert not hasattr(untied_model.output_layer.weight, "is_md_embedding_parameter")
        assert untied_model.router.weight.is_router is True
        assert untied_model.attn.linear_qkv.weight.is_qkv is True
        assert untied_model.mlp.linear_fc2.weight.is_out_proj is True
        assert (
            md_optimizer._resolve_gains_mode(untied_model.embedding.word_embeddings.weight)
            == "none"
        )
        assert md_optimizer._resolve_gains_mode(untied_model.output_layer.weight) == "rowcol"
        assert md_optimizer._resolve_gains_mode(untied_model.router.weight) == "rowcol"

        assert shared_model.output_layer.weight.is_md_embedding_parameter is True
        assert not hasattr(shared_model.output_layer.weight, "is_md_output_parameter")
        assert shared_md_optimizer._resolve_gains_mode(shared_model.output_layer.weight) == "none"

        assert offload_model.experts.weight2.expert_tp is True
        assert offload_model.experts.weight2.merged_offload_expert is True
        assert offload_model.experts.weight2.is_out_proj is True
        offload_main_param = offload_model.experts.weight2.main_param
        assert offload_main_param.merged_offload_expert is True
        assert offload_main_param.is_out_proj is True
        assert any(
            p is offload_main_param
            for group in offload_md_optimizer.param_groups
            for p in group["params"]
        )
    finally:
        Utils.destroy_model_parallel()


def test_md_decoupling_direct_gains_no_clamp_min_round_trip():
    param = torch.nn.Parameter(torch.tensor([[2.0, -4.0], [6.0, -8.0]]))
    original = param.detach().clone()
    param.grad = torch.ones_like(param)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="flat",
        gain_parametrization="direct",
        gains_no_clamp_min=True,
        pg_collection=None,
    )
    optimizer.state[param]["flat_gain"] = torch.tensor(-2.0)

    gain_grads = optimizer._preprocess_gains(param)

    torch.testing.assert_close(param, original / -2.0)
    torch.testing.assert_close(gain_grads["flat_gain"], torch.tensor(2.0))

    optimizer._apply_gains(param)

    torch.testing.assert_close(param, original)


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


def test_md_decoupling_gqa_qkv_split_mechanics():
    param = torch.nn.Parameter(torch.empty(8, 4))
    param.is_qkv = True
    grad = torch.arange(32, dtype=torch.float32).view(8, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
    )

    assert [call.shape for call in calls] == [
        torch.Size([4, 4]),
        torch.Size([2, 4]),
        torch.Size([2, 4]),
    ]
    expected = torch.tensor([1] * 4 + [2] * 2 + [3] * 2).view(8, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_md_decoupling_gqa_split_flat_normalization_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    _assert_qkv_split_flat_norms(optimizer, param, expected_norm=2.0)


def test_md_decoupling_gqa_split_tangential_grad_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    grad = torch.arange(33, 65, dtype=torch.float32).view(8, 4)
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_tangential_grad=True,
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._project_tangent_inplace(param, grad, is_qkv=True)

    _assert_qkv_split_tangent(optimizer, param, grad)


def test_md_decoupling_gqa_split_row_normalization():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="row",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    for part in _split_qkv(param, optimizer.qkv_split_shapes):
        row_norms = torch.linalg.vector_norm(part, dim=1)
        torch.testing.assert_close(row_norms, torch.ones_like(row_norms), rtol=1e-5, atol=1e-6)


@requires_cuda_and_emerging
@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_md_decoupling_num_ns_steps(num_ns_steps):
    torch.manual_seed(42)
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device="cuda")
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert optimizer.num_ns_steps == num_ns_steps


@requires_cuda_and_emerging
@pytest.mark.parametrize("use_nesterov", [True, False])
def test_md_decoupling_nesterov(use_nesterov):
    torch.manual_seed(42)
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device="cuda")
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        use_nesterov=use_nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert optimizer.use_nesterov is use_nesterov
