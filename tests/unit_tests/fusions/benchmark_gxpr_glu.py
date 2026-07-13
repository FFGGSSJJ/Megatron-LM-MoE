# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Manual GPU benchmark: fused GXPR GLU vs fused SwiGLU throughput.

Not a pytest (no ``test_`` prefix, so it isn't collected/run in CI) -- run directly on a CUDA box:

    python tests/unit_tests/fusions/benchmark_gxpr_glu.py

GXPR's fusion is built the same way as SwiGLU's own fusion in this codebase (``@jit_fuser`` /
torch.compile, not a hand Triton kernel -- see ``fused_gxpr_glu.py``'s docstring for why GXPR's
gate, unlike PolyNorm's, doesn't need a reduction-driven Triton kernel to hit SwiGLU speed).
Because both use the identical underlying fusion mechanism, this benchmark checks two things:
  1. Forward+backward wall-clock at a few (M, D) shapes typical of a dense MLP and of MoE-routed
     tokens -- the direct "is it as fast as SwiGLU" comparison.
  2. That GXPR's shape-to-shape behavior as M (the MoE token count) varies call to call tracks
     SwiGLU's own -- both are ``@jit_fuser``-compiled, so both pay whatever recompilation /
     dynamic-shape cost torch.compile incurs on that path; the two should move together rather
     than GXPR being disproportionately worse.
"""
import time

import torch
import torch.nn.functional as F

from megatron.core.activations import GXPR
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl


def _time_fwd_bwd(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        out = fn(*args)
        out.sum().backward()
        for a in args:
            if torch.is_tensor(a) and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(*args)
        out.sum().backward()
        for a in args:
            if torch.is_tensor(a) and a.grad is not None:
                a.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def bench_shape(M, D, dtype=torch.bfloat16):
    torch.manual_seed(0)

    # GXPR fused, via the actual module (matches what --gxpr runs in the real model).
    gxpr = GXPR(num_local_experts=1).cuda().to(dtype)
    x = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)

    def gxpr_fn(x_, y_):
        return gxpr(x_, y_)

    t_gxpr = _time_fwd_bwd(gxpr_fn, x, y)

    # Fused SwiGLU, via the actual fused impl used elsewhere in this codebase for the
    # bias_activation_fusion path (fc1_output is the [x_glu, x_linear] concatenation).
    fc1_out = torch.randn(M, 2 * D, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.zeros(2 * D, dtype=dtype, device="cuda")

    def swiglu_fn(fc1_):
        return bias_swiglu_impl(fc1_, bias)

    t_swiglu = _time_fwd_bwd(swiglu_fn, fc1_out)

    ratio = t_gxpr / t_swiglu
    print(f"M={M:>6} D={D:>5} dtype={str(dtype):>14}: "
          f"gxpr={t_gxpr:.4f} ms  swiglu={t_swiglu:.4f} ms  gxpr/swiglu={ratio:.2f}x")
    return t_gxpr, t_swiglu


def check_recompile_behavior_tracks_swiglu(D=1024, dtype=torch.bfloat16):
    """Vary M call-to-call (as MoE routing does) for both GXPR and SwiGLU and compare their
    per-call timing profiles side by side. Both use ``@jit_fuser``, so both should show the same
    shape (any recompilation spikes should land on the same calls, at a similar relative cost) --
    that's the point of building GXPR's fusion the same way as SwiGLU's rather than hand Triton.
    """
    print("\n--- varying M across calls (simulates MoE routed-token variability) ---")
    gxpr = GXPR(num_local_experts=1).cuda().to(dtype)
    bias = torch.zeros(2 * D, dtype=dtype, device="cuda")
    gxpr_times, swiglu_times = [], []
    for M in [37, 512, 129, 2048, 5, 900, 4096, 17]:
        x = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
        y = torch.randn(M, D, dtype=dtype, device="cuda", requires_grad=True)
        fc1_out = torch.randn(M, 2 * D, dtype=dtype, device="cuda", requires_grad=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gxpr(x, y).sum().backward()
        torch.cuda.synchronize()
        t_gxpr = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bias_swiglu_impl(fc1_out, bias).sum().backward()
        torch.cuda.synchronize()
        t_swiglu = (time.perf_counter() - t0) * 1000

        gxpr_times.append(t_gxpr)
        swiglu_times.append(t_swiglu)
        print(f"  M={M:>5}: gxpr={t_gxpr:.3f} ms  swiglu={t_swiglu:.3f} ms  ratio={t_gxpr/t_swiglu:.2f}x")

    print(f"  gxpr max/min across calls:   {max(gxpr_times)/min(gxpr_times):.2f}x")
    print(f"  swiglu max/min across calls: {max(swiglu_times)/min(swiglu_times):.2f}x "
          f"(if these two ratios are similar, GXPR is tracking SwiGLU's own recompile behavior "
          f"rather than paying an extra, disproportionate cost)")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    print("=== dense-MLP-scale shapes ===")
    for M, D in [(4096, 1024), (8192, 4096)]:
        bench_shape(M, D)
    print("\n=== MoE-expert-scale shapes (narrow ffn, small per-expert token counts) ===")
    for M, D in [(64, 256), (256, 512), (512, 256)]:
        bench_shape(M, D)
    check_recompile_behavior_tracks_swiglu()
