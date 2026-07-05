# Copyright (c) 2026, Swiss AI Institute
"""Unit tests for the MoE expert activation-offload primitives (moe_offload_activations).

These exercise the machinery that offloads the saved expert input activation (fp8_x) to pinned
host memory during the combine phase and reloads it before the expert backward:

* ``MoEActivationOffloadManager.offload`` D2H-copies fp8_x and frees its GPU storage (peak-memory win);
* ``MoEActivationOffloadManager.reload`` restores it bit-for-bit into a fresh GPU tensor;
* ``MoEActReloadTrigger`` is a forward identity whose backward triggers the reload;
* ``PinnedActBufferPool`` recycles pinned buffers (never freeing them) with an event guard.

Numeric grad-parity of the full offloading-experts path is covered separately as an integration
test (it needs DeepGEMM/grouped_gemm + distributed init); here we validate the offload plumbing
in isolation, which is the part introduced by moe_offload_activations.
"""
import pytest
import torch

from megatron.core.transformer.moe.moe_offload import (
    ActivationOffloadHandle,
    MoEActivationOffloadManager,
    MoEActReloadTrigger,
    PinnedActBufferPool,
    StreamManager,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="activation offload requires CUDA (streams/events/pinned)"
)

FP8 = torch.float8_e4m3fn


def _fresh_pool():
    PinnedActBufferPool._instance = None
    return PinnedActBufferPool.get_instance()


def _make_fp8(m, n):
    # Random bf16 values cast to fp8 so the bit pattern is representative.
    return (torch.randn(m, n, device="cuda", dtype=torch.bfloat16)).to(FP8)


def _bits(t):
    return t.view(torch.uint8)


def test_offload_frees_gpu_storage_and_reload_restores_bitexact():
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    fp8_x = _make_fp8(256, 128)
    reference = fp8_x.detach().clone().cpu()  # snapshot exact bytes (on CPU) before offload

    handle = MoEActivationOffloadManager.offload(fp8_x, sm)
    torch.cuda.synchronize()

    # Handle is active and the GPU storage of fp8_x has actually been freed (the memory win).
    assert handle.active
    assert handle.shape == (256, 128) and handle.dtype == FP8
    assert fp8_x.untyped_storage().size() == 0, "fp8_x GPU storage should be freed after offload"
    assert handle.cpu_base.is_pinned() and handle.cpu_flat.is_pinned()
    # cpu_flat is a length-numel fp8 view sliced out of the (possibly larger) capacity buffer.
    assert handle.cpu_flat.numel() == 256 * 128
    assert torch.equal(_bits(handle.cpu_flat), _bits(reference.reshape(-1))), "D2H must be bit-exact"

    # Reload restores the tensor bit-for-bit into a fresh GPU tensor.
    MoEActivationOffloadManager.reload(handle)
    torch.cuda.synchronize()
    assert handle.gpu_slot is not None
    assert handle.gpu_slot.shape == (256, 128) and handle.gpu_slot.dtype == FP8
    assert torch.equal(_bits(handle.gpu_slot.cpu()), _bits(reference)), "reload must be bit-exact"


def test_reload_trigger_forward_identity_backward_reloads():
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    fp8_x = _make_fp8(128, 64)
    reference = fp8_x.detach().clone().cpu()
    handle = MoEActivationOffloadManager.offload(fp8_x, sm)
    torch.cuda.synchronize()

    # Forward is an identity on the (unrelated) combine-output tensor.
    combine_out = torch.randn(8, device="cuda", requires_grad=True)
    out = MoEActReloadTrigger.apply(combine_out, handle)
    assert torch.equal(out, combine_out)
    assert handle.gpu_slot is None, "reload must not run in forward"

    # Backward triggers the reload.
    out.sum().backward()
    torch.cuda.synchronize()
    assert handle.gpu_slot is not None, "reload must run in MoEActReloadTrigger.backward"
    assert torch.equal(_bits(handle.gpu_slot.cpu()), _bits(reference))


def test_reload_trigger_inactive_handle_is_noop():
    combine_out = torch.randn(8, device="cuda", requires_grad=True)
    # Inactive/None handles must be pass-through with no reload attempted.
    for handle in (None, ActivationOffloadHandle(active=False)):
        out = MoEActReloadTrigger.apply(combine_out, handle)
        out.sum().backward()
        assert torch.equal(out, combine_out)


def test_pinned_pool_capacity_based_recycle_and_growth():
    pool = _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)
    MB = 1024 * 1024

    # Flat uint8 buffers with capacity >= request; distinct shapes with the same byte size share a
    # buffer, so shape variance does not fragment the pool.
    buf1 = pool.allocate(3 * MB, sm.act_d2h_stream)
    assert buf1.dtype == torch.uint8 and buf1.dim() == 1 and buf1.is_pinned()
    assert buf1.numel() >= 3 * MB

    # A recycled buffer serves any request that fits (best-fit by capacity), regardless of shape.
    pool.free(buf1, ready_event=None)
    buf2 = pool.allocate(1 * MB, sm.act_d2h_stream)
    assert buf2 is buf1, "a smaller request should reuse the larger recycled buffer"

    # With buf1 still checked out, a concurrent request must allocate a distinct buffer.
    buf3 = pool.allocate(1 * MB, sm.act_d2h_stream)
    assert buf3 is not buf1

    # A request larger than every free buffer's capacity grows the pool.
    pool.free(buf2, ready_event=None)  # buf2 is buf1
    big = buf1.numel() + 1
    buf4 = pool.allocate(big, sm.act_d2h_stream)
    assert buf4 is not buf1 and buf4.numel() >= big


def test_offload_reload_roundtrip_reuses_pooled_buffer_across_shapes_bitexact():
    """Two offload/reload cycles with *different* fp8_x shapes must reuse the same capacity buffer
    (shape variance must not fragment the pool) and must not corrupt data (event-guarded recycle)."""
    pool = _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    # First cycle.
    x1 = _make_fp8(200, 128)
    ref1 = x1.detach().clone().cpu()
    h1 = MoEActivationOffloadManager.offload(x1, sm)
    MoEActivationOffloadManager.reload(h1)
    torch.cuda.synchronize()
    assert torch.equal(_bits(h1.gpu_slot.cpu()), _bits(ref1))
    pool.free(h1.cpu_base, h1.h2d_done)  # mirror the backward-path recycle

    # Second cycle, DIFFERENT shape but same (small) capacity class -> reuse the pooled buffer.
    x2 = _make_fp8(177, 128)
    ref2 = x2.detach().clone().cpu()
    h2 = MoEActivationOffloadManager.offload(x2, sm)
    MoEActivationOffloadManager.reload(h2)
    torch.cuda.synchronize()
    assert h2.cpu_base is h1.cpu_base, "differently-shaped offload should reuse the capacity buffer"
    assert h2.numel != h1.numel, "the two cycles should have different element counts"
    assert torch.equal(_bits(h2.gpu_slot.cpu()), _bits(ref2))


# ---------------------------------------------------------------------------
# fc1_output offload (moe_offload_fc1_output): a second, larger slot on the same handle.
# ---------------------------------------------------------------------------


def test_offload_fc1_output_both_slots_bitexact_and_storage_freed():
    """Offloading fp8_x then fp8_fc1 onto one handle parks both on pinned host (bit-exact) and frees
    both GPU storages; reload restores both into fresh GPU tensors with distinct h2d_done events."""
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    fp8_x = _make_fp8(128, 64)
    ref_x = fp8_x.detach().clone().cpu()
    handle = MoEActivationOffloadManager.offload(fp8_x, sm)

    # fc1_output is ~5x larger (tokens x 2*ffn) and fp8; offload onto the same handle.
    fp8_fc1 = _make_fp8(128, 320)
    ref_fc1 = fp8_fc1.detach().clone().cpu()
    MoEActivationOffloadManager.offload_fc1(handle, fp8_fc1)
    torch.cuda.synchronize()

    assert handle.active and handle.fc1_active
    assert fp8_x.untyped_storage().size() == 0, "fp8_x GPU storage must be freed"
    assert fp8_fc1.untyped_storage().size() == 0, "fp8_fc1 GPU storage must be freed"
    assert handle.cpu_flat.numel() == 128 * 64
    assert handle.cpu_flat_fc1.numel() == 128 * 320
    assert torch.equal(_bits(handle.cpu_flat), _bits(ref_x.reshape(-1))), "fp8_x D2H must be bit-exact"
    assert torch.equal(_bits(handle.cpu_flat_fc1), _bits(ref_fc1.reshape(-1))), "fc1 D2H must be bit-exact"

    MoEActivationOffloadManager.reload(handle)
    torch.cuda.synchronize()
    # Two distinct slots / events; fc1 (larger) serializes after fp8_x on the single h2d stream.
    assert handle.gpu_slot is not None and handle.gpu_slot_fc1 is not None
    assert handle.h2d_done is not None and handle.h2d_done_fc1 is not None
    assert handle.h2d_done is not handle.h2d_done_fc1
    assert handle.gpu_slot.shape == (128, 64)
    assert handle.gpu_slot_fc1.shape == (128, 320)
    assert torch.equal(_bits(handle.gpu_slot.cpu()), _bits(ref_x)), "fp8_x reload must be bit-exact"
    assert torch.equal(_bits(handle.gpu_slot_fc1.cpu()), _bits(ref_fc1)), "fc1 reload must be bit-exact"


def test_fc1_disabled_keeps_handle_fp8_only():
    """Without offload_fc1, the handle stays fp8_x-only: fc1 slot inactive and reload leaves
    gpu_slot_fc1 / h2d_done_fc1 unset (the fp8_x path is untouched by the fc1 feature)."""
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)
    fp8_x = _make_fp8(64, 64)
    ref = fp8_x.detach().clone().cpu()
    handle = MoEActivationOffloadManager.offload(fp8_x, sm)

    assert not handle.fc1_active
    MoEActivationOffloadManager.reload(handle)
    torch.cuda.synchronize()
    assert handle.gpu_slot is not None
    assert handle.gpu_slot_fc1 is None and handle.h2d_done_fc1 is None
    assert torch.equal(_bits(handle.gpu_slot.cpu()), _bits(ref))


def test_reload_trigger_reloads_both_slots_via_backward():
    """MoEActReloadTrigger.backward must reload BOTH the fp8_x and fc1 slots of a handle that carries
    fc1, so a single trigger node covers the two-tensor offload."""
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)
    fp8_x = _make_fp8(64, 64)
    ref_x = fp8_x.detach().clone().cpu()
    fp8_fc1 = _make_fp8(64, 256)
    ref_fc1 = fp8_fc1.detach().clone().cpu()
    handle = MoEActivationOffloadManager.offload(fp8_x, sm)
    MoEActivationOffloadManager.offload_fc1(handle, fp8_fc1)

    combine_out = torch.randn(8, device="cuda", requires_grad=True)
    out = MoEActReloadTrigger.apply(combine_out, handle)
    assert handle.gpu_slot is None and handle.gpu_slot_fc1 is None, "reload must not run in forward"
    assert torch.equal(out, combine_out)

    out.sum().backward()
    torch.cuda.synchronize()
    assert handle.gpu_slot is not None and handle.gpu_slot_fc1 is not None
    assert torch.equal(_bits(handle.gpu_slot.cpu()), _bits(ref_x))
    assert torch.equal(_bits(handle.gpu_slot_fc1.cpu()), _bits(ref_fc1))
