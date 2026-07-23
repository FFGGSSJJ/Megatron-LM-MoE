# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestGPTInferenceWrapper:

    def setup_model(self, tensor_parallel_size, pipeline_parallel_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.vocab_size = 100
        self.batch_size = 4
        self.sequence_length = 32
        hidden_size = 32

        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=self.vocab_size,
            max_sequence_length=self.sequence_length,
            parallel_output=True,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()

        inference_context = StaticInferenceContext(self.batch_size, self.sequence_length)

        self.inference_wrapped_model = GPTInferenceWrapper(gpt_model, inference_context)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_pipeline_parallel(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=2, pipeline_parallel_size=2)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.config.materialize_only_last_token_logits = (
            materialize_only_last_token_logits
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )

        logits_seq_len = 1 if materialize_only_last_token_logits else 5

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )
        # Logits are not returned in all ranks in PP
        if parallel_state.is_pipeline_last_stage():
            assert logits.shape == (
                self.batch_size,
                logits_seq_len,
                self.vocab_size,
            ), f"Shape mismatch . Expected {(self.batch_size, logits_seq_len, self.vocab_size)}, but got {logits.shape}"

    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_inference_only_tensor_parallel(self, materialize_only_last_token_logits):
        self.setup_model(tensor_parallel_size=4, pipeline_parallel_size=1)

        batch_prompt_tokens = (
            torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.sequence_length))
            .int()
            .cuda()
        )
        self.inference_wrapped_model.prep_model_for_inference()
        self.inference_wrapped_model.inference_context.config.materialize_only_last_token_logits = (
            materialize_only_last_token_logits
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=batch_prompt_tokens
        )

        inference_input_for_context_window = (
            self.inference_wrapped_model.get_batch_for_context_window(inference_input, 0, 5)
        )

        logits_seq_len = 1 if materialize_only_last_token_logits else 5

        logits = self.inference_wrapped_model.run_one_forward_step(
            inference_input_for_context_window
        )

        assert logits.shape == (
            self.batch_size,
            logits_seq_len,
            self.vocab_size,
        ), f"Shape mismatch . Expected {(self.batch_size, logits_seq_len, self.vocab_size)}, but got {logits.shape}"

    @torch.inference_mode()
    def test_dummy_forward_with_absorbed_mla_without_inference_context(self):
        """The eager EP dummy path must work after MLA deletes its fused KV up projection."""
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        vocab_size = 100
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            qk_layernorm=True,
            cache_mla_latents=True,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        model = (
            GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                    multi_latent_attention=True, qk_layernorm=True
                ),
                vocab_size=vocab_size,
                max_sequence_length=8,
                parallel_output=True,
            )
            .cuda()
            .bfloat16()
            .eval()
        )
        wrapper = GPTInferenceWrapper(model, StaticInferenceContext(1, 8))
        attention = model.decoder.layers[0].self_attention

        assert hasattr(attention, "linear_kv_up_proj")
        first_logits = wrapper.dummy_forward()
        assert not hasattr(attention, "linear_kv_up_proj")
        assert hasattr(attention, "linear_kv_up_proj_linear")

        # Exercise the already-absorbed state a second time; this is the state used
        # by dummy forwards after the first real inference step.
        second_logits = wrapper.dummy_forward()

        for logits in (first_logits, second_logits):
            assert logits.shape == (1, 1, vocab_size)
            assert torch.isfinite(logits).all()
