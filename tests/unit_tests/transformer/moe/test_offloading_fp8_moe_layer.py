# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.moe.experts import (
    OffloadingExpertsMLP
)

from megatron.core.transformer.moe.experts_util import (
    grouped_swiglu_mlp,
    MergedSwiGLU,
)

from megatron.core.transformer.moe.experts_offloading_util import (
    offloading_grouped_swiglu_mlp,
    OffloadingExpertsGroupedSwiMLP,
    StreamManager,
)

from megatron.core.transformer.moe.experts_offloading_fp8_util import (
    OffloadingExpertsFP8GroupedSwiMLP,
    FP8ExpertsParameterManager,
    offloading_fp8_grouped_swiglu_mlp
)



class TestOffloadingMoELayerFP8:
    """Test MoE layer with FP8 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("num_moe_experts", [32])
    def test_offloading_experts_bf16_forward(
        self, num_moe_experts, 
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        # _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 2048
        moe_ffn_hidden_size = 1024

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=16,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,

            gated_linear_unit=True,
            moe_offloading_num_chunks=4,
            moe_offloading_num_stages=2,
            # moe_offloading_chunk_size=4,
        )

        offloading_expert = OffloadingExpertsMLP(
            num_moe_experts,
            transformer_config
        )

        tokens_per_expert = torch.randint(
            2047, 2048, (num_moe_experts,), 
            device="cpu"
        )

        hidden_states = torch.randn(
            tokens_per_expert.sum().item(),
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        permuted_probs = torch.rand(
            tokens_per_expert.sum().item(),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        # Forward pass
        # output, _ = offloading_expert(
        #     hidden_states,
        #     tokens_per_expert,
        #     permuted_probs,
        # )

        wait, warmup, active = 1, 1, 2
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
            ),
        ) as prof:
            for _ in range(num_steps):
                output, _ = offloading_expert(
                    hidden_states,
                    tokens_per_expert,
                    permuted_probs,
                )
                prof.step()
        prof.export_chrome_trace(f"test.json")


    def test_offloading_moe_forward_backward(
        self, num_moe_experts, profile=False, num_repeats=10
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""

        micro_batch_size = 2
        sequence_length = 4096
        hidden_size = 7168
        moe_ffn_hidden_size = 2048
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        tokens_per_expert = torch.randint(
            1024, 1025, (num_moe_experts,), 
            device="cpu"
        )
        tokens_per_expert_ref = tokens_per_expert.detach().clone()

        hidden_states = (torch.randn(
            tokens_per_expert.sum().item(),
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=False,
        )).detach().requires_grad_(True)
        hidden_states_ref = hidden_states.detach().clone()

        permuted_probs = torch.rand(
            tokens_per_expert.sum().item(),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        permuted_probs_ref = permuted_probs.detach().clone()

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=16,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,

            gated_linear_unit=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
            gradient_accumulation_fusion=True,
        )

        # Draw weights on CUDA so the CUDA RNG stream advances identically to
        # the non-offloading test, then mirror to pinned CPU for the offload path.
        w1_init = torch.randn(
            num_moe_experts, moe_ffn_hidden_size * 2, hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )
        w2_init = torch.randn(
            num_moe_experts, hidden_size, moe_ffn_hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )
        cpu_w1 = torch.nn.Parameter(
            torch.empty_like(w1_init, device="cpu", pin_memory=True).copy_(w1_init),
            requires_grad=True,
        )
        cpu_w2 = torch.nn.Parameter(
            torch.empty_like(w2_init, device="cpu", pin_memory=True).copy_(w2_init),
            requires_grad=True,
        )
        
        cpu_w1_list = list(torch.unbind(cpu_w1, dim=0))
        cpu_w2_list = list(torch.unbind(cpu_w2, dim=0))

        # main grad
        cpu_w1.main_grad = torch.zeros_like(cpu_w1, device="cuda", dtype=torch.float32)
        cpu_w2.main_grad = torch.zeros_like(cpu_w2, device="cuda", dtype=torch.float32)

        gpu_w1_buffer = [
            [torch.empty(
                moe_ffn_hidden_size*2, hidden_size, device="cuda", dtype=torch.float8_e4m3fn, requires_grad=False
            ) for _ in range(moe_offloading_chunk_size)] for _ in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [torch.empty(
                 hidden_size, moe_ffn_hidden_size, device="cuda", dtype=torch.float8_e4m3fn, requires_grad=False
            ) for _ in range(moe_offloading_chunk_size)] for _ in range(moe_offloading_num_stages)
        ]

        # allocate tensors for gpu buffers
        experts1_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages * moe_offloading_chunk_size * moe_ffn_hidden_size*2 * hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float8_e4m3fn,
        ).view(
            moe_offloading_num_stages, moe_offloading_chunk_size, moe_ffn_hidden_size*2, hidden_size
        )
        experts2_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages * moe_offloading_chunk_size * moe_ffn_hidden_size * hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float8_e4m3fn,
        ).view(
            moe_offloading_num_stages, moe_offloading_chunk_size, hidden_size, moe_ffn_hidden_size
        )

        # organize as [num_stages, chunk_size, (in, out)]
        gpu_w1_buffer = [
            [experts1_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [experts2_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]

        # organize as [num_stages, (chunk_size, in, out)]
        gpu_w1_chunks = [
            experts1_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_chunks = [
            experts2_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]

        # reference
        gpu_w1 = torch.nn.Parameter(
            cpu_w1.detach().clone().cuda(), requires_grad=True
        )
        
        gpu_w2 = torch.nn.Parameter(
            cpu_w2.detach().clone().cuda(), requires_grad=True
        ) 
        

        gpu_w1.main_grad = torch.zeros_like(gpu_w1, device="cuda", dtype=torch.float32)
        gpu_w2.main_grad = torch.zeros_like(gpu_w2, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()

        stream_manager = StreamManager(moe_offloading_num_stages, 4)
        FP8ExpertsParameterManager.create_instance(
            config=transformer_config,
        )

        # Realistic upstream gradient: simulate an MSE loss against a random
        # target, so grad_y has the same per-token magnitude structure as in
        # training. Using .sum().backward() (grad_y = 1) makes grad_w1/grad_w2
        # errors depend on bf16-vs-fp32 accumulation order rather than on the
        # quantization paths we care about.
        target = torch.randn(
            tokens_per_expert.sum().item(), hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )

        if not profile:
            hidden_states_ref_list = list(torch.split(hidden_states_ref, tokens_per_expert_ref.tolist(), dim=0))
            output_ref_list = []
            for i in range(num_moe_experts):
                output_ref_list.append(
                    torch.mm(hidden_states_ref_list[i], gpu_w1[i].t())
                )
            outputs_fc1 = torch.cat(output_ref_list, dim=0)
            output_ref = outputs_fc1
            output_ref = MergedSwiGLU.apply(outputs_fc1, permuted_probs_ref.unsqueeze(-1))
            outputs_act_list = list(torch.split(output_ref, tokens_per_expert_ref.tolist(), dim=0))
            output_ref_list = []
            for i in range(num_moe_experts):
                output_ref_list.append(
                    torch.mm(outputs_act_list[i], gpu_w2[i].t())
                )
            output_ref = torch.cat(output_ref_list, dim=0)
            ((output_ref - target).float() ** 2).sum().backward()
            torch.cuda.synchronize()
        # return
        if profile:
            wait, warmup, active = 1, 5, 2
            num_steps = wait + warmup + active
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
                ),
            ) as prof:
                for _ in range(num_steps):
                    output = offloading_fp8_grouped_swiglu_mlp(
                        cpu_w1,
                        cpu_w2,
                        cpu_w1_list,
                        cpu_w2_list,
                        gpu_w1_buffer,
                        gpu_w2_buffer,
                        gpu_w1_chunks,
                        gpu_w2_chunks,
                        hidden_states,
                        tokens_per_expert,
                        num_moe_experts,
                        permuted_probs,
                        None,
                        stream_manager,
                        transformer_config,
                        []
                    )
                    torch.cuda.synchronize()
                    output.sum().backward()
                    torch.cuda.synchronize()
                    prof.step()
            prof.export_chrome_trace(f"fp8_e2e.json")
        else:
            outputs = []
            for _ in range(num_repeats):
                output = offloading_fp8_grouped_swiglu_mlp(
                    cpu_w1,
                    cpu_w2,
                    cpu_w1_list,
                    cpu_w2_list,
                    gpu_w1_buffer,
                    gpu_w2_buffer,
                    gpu_w1_chunks,
                    gpu_w2_chunks,
                    hidden_states,
                    tokens_per_expert,
                    num_moe_experts,
                    permuted_probs,
                    None,
                    stream_manager,
                    transformer_config,
                    []
                )
                cpu_w1.main_grad.zero_()
                cpu_w2.main_grad.zero_()
                ((output - target).float() ** 2).sum().backward()
                outputs.append(output)
                torch.cuda.synchronize()


        if not profile:
            def diff_tensor_norm(tensor1, tensor2):
                return torch.norm(tensor1.to(torch.float) - tensor2.to(torch.float)).item() / torch.norm(tensor2.to(torch.float)).item()

            for i, output in enumerate(outputs):
                assert output.shape == output_ref.shape, f"Output shape mismatch: {output.shape} vs {output_ref.shape}"
                diff_half_tensor = diff_tensor_norm(output.cuda()[:tokens_per_expert.sum().item()//2], output_ref.cuda()[:tokens_per_expert.sum().item()//2])
                diff_half_2_tensor = diff_tensor_norm(output.cuda()[tokens_per_expert.sum().item()//2:], output_ref.cuda()[tokens_per_expert.sum().item()//2:])
                diff = diff_tensor_norm(output.cuda(), output_ref.cuda())
                if diff > 0.07:
                    print(f"Output norm half 1 difference: {diff_half_tensor:.4f}")
                    print(f"Output norm half 2 difference: {diff_half_2_tensor:.4f}")
                    print(f"Output norm difference: {diff:.4f}")
                    print(output.cuda())
                    print(output_ref.cuda())
                    assert False, f"{i} Norm difference {diff:.4f} exceeds threshold"

            print(diff_tensor_norm(cpu_w1.main_grad, gpu_w1.grad))
            print(diff_tensor_norm(cpu_w2.main_grad, gpu_w2.grad))

    def test_offloading_moe_layer(
        self,
        num_moe_experts, 
        moe_token_dispatcher_type, 
        tp_size, 
        ep_size,
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 2048
        sequence_length = 4096
        micro_batch_size = 2

        moe_ffn_hidden_size = 1024
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=32,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=8,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=True,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=False,

            bf16=True,
            params_dtype=torch.bfloat16,
            gated_linear_unit=True,
            gradient_accumulation_fusion=True,
            activation_func=torch.nn.functional.silu,

            moe_use_legacy_grouped_gemm=True,
            moe_use_offloading_experts=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
            moe_use_inplace_fp8_param=True,
            moe_offloading_experts_debug_mode=False,
        )

        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, 
            moe_grouped_gemm=True,
            moe_use_offloading_experts=True,
        )

        FP8ExpertsParameterManager.create_instance(
            num_local_experts=num_moe_experts,
            config=transformer_config,
        )

        moe_layer = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        moe_layer.router.set_layer_number(1)

        output, _ = moe_layer(hidden_states)

        



        

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    TestOffloadingMoELayerFP8().test_offloading_moe_forward_backward(
        num_moe_experts=64, profile=False, num_repeats=10
    )
    
    
    