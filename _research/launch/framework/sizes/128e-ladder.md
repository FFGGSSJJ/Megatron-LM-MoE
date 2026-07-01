# Fine-grained (128e) MoE size ladder

Scaling/ablation ladder of architecture proxies for the **in-house 670B-A40B
target**: 128 routed experts, top-4, 1 shared expert, `moe_ffn = shared =
hidden/1.75`, h=7168, ~60 layers (≈663B total / ~40B active, 5.85% non-embed).

Two attention variants of the **same geometry**:
- **GQA** — `<size>-moe-128e.sh`
- **MLA** — `<size>-moe-128e-mla.sh` (matches the target's attention; DeepSeek-V3 style)

Distinct from the 64e/top-2 ladder (`*-moe.sh`).

## Rungs (GQA)

| size file | L | hidden | heads/kv | aspect | dense | moe_ffn=shared | total | active | NE-sp | MoE layers | tokens (100/act) | iters (GBS=128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `1.5b-moe-128e` | 10 | 768  | 6/3   | 77 | 1 | 448  | 1.53B  | 0.38B | 5.69% | 9  | 38B  | 35,929  |
| `3b-moe-128e`   | 12 | 1024 | 8/4   | 85 | 1 | 576  | 2.97B  | 0.55B | 5.64% | 11 | 55B  | 52,844  |
| `5b-moe-128e`   | 14 | 1280 | 10/5  | 91 | 1 | 704  | 5.13B  | 0.77B | 5.61% | 13 | 77B  | 73,539  |
| `9.3b-moe-128e` | 17 | 1536 | 12/6  | 90 | 1 | 896  | 9.28B  | 1.09B | 5.44% | 16 | 109B | 103,574 |
| `14b-moe-128e`  | 20 | 1792 | 14/7  | 90 | 1 | 1024 | 14.43B | 1.46B | 5.43% | 19 | 146B | 139,343 |
| `22b-moe-128e`  | 24 | 2048 | 16/8  | 85 | 1 | 1152 | 22.16B | 1.97B | 5.41% | 23 | 197B | 188,150 |
| `120b-moe-128e` | 42 | 3584 | 28/14 | 85 | 1 | 2048 | 119.6B | 7.68B | 5.29% | 41 | 768B | 732,594 |
| _target_        | 60 | 7168 | 128/MLA | 119 | 3 | 4096 | ~663B | ~40B | 5.85% | 57 | — | — |

Named by total params, computed for the apertus `preliminary_mul_200k` (~200k)
vocab — the embedding + untied LM-head slabs (`2 × vocab × hidden`) are the
vocab-dependent part; the non-embedding backbone (`NE-sp`, MoE layers, aspect) is
vocab-independent.

## Design rules

- **128 routed experts, top-4, 1 shared at full routed-expert width** — matches
  the target. moe_ffn = shared = `round(hidden/1.75, ×64)` (target ratio exactly).
- **head_dim = 128** (kernel-friendly + matches the target; heads = hidden/128 =
  6/8/10/12/14/16). **GQA kv = heads/2** (constant 2:1 → 3/4/5/6/7/8). Every
  width-scaled shape (Q/K/V/O = d², MoE/shared ≈ 0.571·d, dense MLP = 2.5·d,
  head_dim = 128) is thus a fixed function of `d` — clean width/LR transfer.
  (router & embeddings scale as `d` not `d²`, the usual muP exception.)
- **1 dense layer** (then all MoE) — matches the target's ~5% dense fraction
  (3/61); 2–3 dense at these depths would be 15–23%, over-weighting the dense part
  and shrinking the MoE stack the ablations study.
- **dense MLP** `FFN_HIDDEN = hidden × 2.5`.
- **Total naming.** L was trimmed per rung to hold clean total targets; the
  non-embedding backbone is fixed by `d`. Totals (and the rung names) are quoted
  for the ~200k apertus vocab, so they sit ~0.4–8% above the same backbone at a
  smaller vocab — largest at the small rungs where the embedding is a big fraction
  (1.5b: +7.5%; 120b: +0.4%). Aspect floats ~77–91, held roughly constant for
  scaling-law fits.

## Token budgets

Tokens scale with **active** params at **100 tok/active-param** (eval-quality: a
~1B-active model gets ~100B tokens — the 9.3b/14b rungs). `TRAIN_SAMPLES` per file
is set accordingly (**seq_len = 8192**) and is a multiple of GBS (else the run
never reaches its target and chain-requeues). Active **includes** the embedding +
LM-head, so the ~200k vocab raises the budget most at the small rungs (1.5b:
27B→38B tokens vs a ~131k vocab). Override for 50 tok/active etc.

**Batch:** `GBS = 128` samples × 8192 = **1.05M tokens/step**. Iters = TRAIN_SAMPLES
/ GBS (column above). `MBS` defaults are ≤2 for the 2× activation memory at
seq_len 8192 (override with `MBS=`).

## MLA variant (`-mla`)

Same geometry; GQA replaced by Multi-Latent Attention to match the target.
Per-head dims fixed at the target's (`qk-head-dim 128`, `qk-pos-emb-head-dim 64`,
`v-head-dim 128`); lora ranks scaled at the target's ratios (`kv-lora ≈ hidden/14`,
`q-lora = 3×kv-lora`). num_heads = hidden/128 (same as the GQA twin → attention
output 1.0×hidden; the target is 2.29×, an artifact of its scale — widen num_heads
to hidden/64 if you want to mirror that exactly).

| size | hidden | num_heads | q-lora | kv-lora | qk/rope/v |
|---|---|---|---|---|---|
| 1.5b | 768  | 6  | 192 | 64  | 128/64/128 |
| 3b   | 1024 | 8  | 192 | 64  | 128/64/128 |
| 5b   | 1280 | 10 | 288 | 96  | 128/64/128 |
| 9.3b | 1536 | 12 | 288 | 96  | 128/64/128 |
| 14b  | 1792 | 14 | 384 | 128 | 128/64/128 |
| 22b  | 2048 | 16 | 480 | 160 | 128/64/128 |
| 120b | 3584 | 28 | 768 | 256 | 128/64/128 |
| _target_ | 7168 | 128 | 1536 | 512 | 128/64/128 |

Enabled via the `MLA_ARGS` array in the size file; `lib/common.sh` emits
`--multi-latent-attention` + the MLA dims and skips GQA when `MLA_ARGS` is
non-empty (backward-compatible — GQA files unchanged).

**Caveats:** `param-count.py` models MHA, so its totals/active are *approximate*
for MLA rungs (FFN+experts identical to the GQA twin; MLA attention differs by a
few %); MLA `TRAIN_SAMPLES` reuse the GQA twin's active. MLA wiring is **untested
at runtime** — smoke-test one small MLA rung before launching.

## Parallelism

**EP defaults to 1 (pure DP)** — overridable via `EP=...`. The 9.3b+ rungs
replicate all 128 experts per GPU and will likely need `EP>1` (or more nodes,
which shards the distributed-optimizer state) to fit memory. The **120b** anchor
**cannot** run pure DP — it needs `EP>1` and likely TP/PP; it's mainly a config
reference / extrapolation anchor (a 768B-token run at 100 tok/active is large).
