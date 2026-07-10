# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _log_router_bias_metrics,
    _update_router_qb_beta,
    reset_model_temporary_tensors,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RouterQBBetaModel(torch.nn.Module):
    def __init__(self, config, qb_beta, qb_beta_accum, qb_beta_count):
        super().__init__()
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.router = torch.nn.Module()
        self.router.register_buffer("qb_beta", qb_beta.clone())
        self.router.register_buffer("qb_beta_accum", qb_beta_accum.clone())
        self.router.register_buffer("qb_beta_count", qb_beta_count.clone())


class _RouterBiasModel(torch.nn.Module):
    def __init__(self, config, expert_bias, qb_beta):
        super().__init__()
        self.config = config
        self.router = torch.nn.Module()
        self.router.layer_number = 1
        self.router.register_buffer('expert_bias', expert_bias.clone())
        self.router.register_buffer('qb_beta', qb_beta.clone())


def _router_qb_config(ema=0.25):
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_ema=ema,
    )


class TestRouterBiasMetrics:
    def test_logs_bias_statistics(self, monkeypatch):
        config = _router_qb_config()
        config.moe_router_bias_metrics = True
        model = _RouterBiasModel(
            config,
            expert_bias=torch.tensor([-2.0, 0.0, 4.0]),
            qb_beta=torch.tensor([-1.0, 0.0, 1.0]),
        )
        logged = {}

        monkeypatch.setattr(
            'megatron.core.distributed.finalize_model_grads.get_num_microbatches', lambda: 2
        )

        def save_metric(name, value, layer_number, num_layers):
            logged[name] = value
            assert layer_number == 1
            assert num_layers == 1

        monkeypatch.setattr(
            'megatron.core.distributed.finalize_model_grads.save_to_aux_losses_tracker',
            save_metric,
        )

        _log_router_bias_metrics([model], config)

        assert set(logged) == {
            'qb_beta_mean',
            'qb_beta_std',
            'qb_beta_min',
            'qb_beta_max',
        }
        torch.testing.assert_close(logged['qb_beta_mean'], torch.tensor(0.0))
        torch.testing.assert_close(logged['qb_beta_min'], torch.tensor(-2.0))
        torch.testing.assert_close(logged['qb_beta_max'], torch.tensor(2.0))

        logged.clear()
        config.moe_router_load_balancing_type = "aux_loss"
        config.moe_router_enable_expert_bias = True
        _log_router_bias_metrics([model], config)

        assert set(logged) == {
            'expert_bias_mean',
            'expert_bias_std',
            'expert_bias_min',
            'expert_bias_max',
        }
        torch.testing.assert_close(logged['expert_bias_mean'], torch.tensor(4.0 / 3.0))
        torch.testing.assert_close(logged['expert_bias_min'], torch.tensor(-4.0))
        torch.testing.assert_close(logged['expert_bias_max'], torch.tensor(8.0))

    def test_disabled_by_default(self, monkeypatch):
        config = _router_qb_config()
        model = _RouterBiasModel(config, torch.tensor([0.0]), torch.tensor([0.0]))

        def fail_save(*args, **kwargs):
            raise AssertionError('bias metrics should be disabled')

        monkeypatch.setattr(
            'megatron.core.distributed.finalize_model_grads.save_to_aux_losses_tracker',
            fail_save,
        )

        _log_router_bias_metrics([model], config)


class TestUpdateRouterQBBeta:
    def test_update_router_qb_beta_ema_centers_and_resets(self, monkeypatch):
        config = _router_qb_config(ema=0.25)
        model = _RouterQBBetaModel(
            config,
            qb_beta=torch.tensor([1.0, -1.0, 0.0]),
            qb_beta_accum=torch.tensor([4.0, 2.0, 0.0]),
            qb_beta_count=torch.tensor(2, dtype=torch.long),
        )
        dp_cp_group = object()
        all_reduce_calls = []
        waits = []

        class FakeWork:
            def __init__(self, index):
                self.index = index

            def wait(self):
                waits.append(self.index)

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            all_reduce_calls.append(
                {"value": tensor.clone(), "op": op, "group": group, "async_op": async_op}
            )
            return FakeWork(len(all_reduce_calls) - 1)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=dp_cp_group)

        local_avg = torch.tensor([2.0, 1.0, 0.0])
        expected = 0.25 * torch.tensor([1.0, -1.0, 0.0]) + 0.75 * local_avg
        expected = expected - expected.mean()
        torch.testing.assert_close(all_reduce_calls[0]["value"], torch.tensor([[4.0, 2.0, 0.0]]))
        torch.testing.assert_close(all_reduce_calls[1]["value"], torch.tensor([2]))
        assert all_reduce_calls[0]["op"] == torch.distributed.ReduceOp.SUM
        assert all_reduce_calls[1]["op"] == torch.distributed.ReduceOp.SUM
        assert all_reduce_calls[0]["group"] is dp_cp_group
        assert all_reduce_calls[1]["group"] is dp_cp_group
        assert all_reduce_calls[0]["async_op"] is True
        assert all_reduce_calls[1]["async_op"] is True
        assert waits == [0, 1]
        torch.testing.assert_close(model.router.qb_beta, expected)
        torch.testing.assert_close(model.router.qb_beta.mean(), torch.zeros(()))

        reset_model_temporary_tensors(config, [model])
        torch.testing.assert_close(
            model.router.qb_beta_accum, torch.zeros_like(model.router.qb_beta_accum)
        )
        assert model.router.qb_beta_count.item() == 0

    def test_update_router_qb_beta_skips_eval_modules(self, monkeypatch):
        config = _router_qb_config(ema=0.0)
        model = _RouterQBBetaModel(
            config,
            qb_beta=torch.tensor([1.0, -1.0, 0.0]),
            qb_beta_accum=torch.tensor([4.0, 2.0, 0.0]),
            qb_beta_count=torch.tensor(1, dtype=torch.long),
        )
        model.router.eval()
        before = model.router.qb_beta.clone()

        def fail_all_reduce(*args, **kwargs):
            raise AssertionError("all_reduce should not run for eval QB routers")

        monkeypatch.setattr(torch.distributed, "all_reduce", fail_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=object())

        torch.testing.assert_close(model.router.qb_beta, before)

    def test_update_router_qb_beta_preserves_beta_without_observations(self, monkeypatch):
        config = _router_qb_config(ema=0.0)
        model = _RouterQBBetaModel(
            config,
            qb_beta=torch.tensor([2.0, -1.0, -1.0]),
            qb_beta_accum=torch.tensor([0.0, 0.0, 0.0]),
            qb_beta_count=torch.tensor(0, dtype=torch.long),
        )

        class FakeWork:
            def wait(self):
                pass

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            del tensor, op, group, async_op
            return FakeWork()

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        _update_router_qb_beta([model], config, dp_cp_group=object())

        torch.testing.assert_close(model.router.qb_beta, torch.tensor([2.0, -1.0, -1.0]))


class TestAllReduceLNGrads:

    def init_model(self, share_embeddings_and_output_weights: bool = False):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=True,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):
        self.tp_size = tp_size
        self.pp_size = 1
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_non_tensor_model_parallel_grads(
            [self.model], self.transformer_config, parallel_state.get_tensor_model_parallel_group()
        )

    @pytest.mark.parametrize(
        ("freeze_model", "pp_size", "share_embeddings"),
        [(True, 2, True), (False, 2, True), (True, 2, False), (False, 2, False)],
    )
    def test_allreduce_word_embedding_grads(self, freeze_model, pp_size, share_embeddings):
        self.tp_size = 1
        self.pp_size = pp_size
        Utils.initialize_model_parallel(pipeline_model_parallel_size=self.pp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model(share_embeddings)
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group()

        _allreduce_word_embedding_grads([self.model], self.transformer_config, embd_group, pp_group)
