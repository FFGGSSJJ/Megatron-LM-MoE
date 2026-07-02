# Experiment framework (megatron-apertus-moe)

A small composable launcher for pretraining runs. You pick a **size** (model +
data scale) and a **recipe** (optimizer / idea); everything else — SLURM, the
container, logging, data, checkpoint/resume — is filled in by `lib/common.sh`.

Ported from `megatron-lm-research-baseline/_research/launch/framework` and
adapted to this repo's code. The focus here is the **md_decoupling** optimizer
recipe (magnitude-direction decoupling), the framework form of `submit_test.sh`.

```
framework/
├── submit.sh         # convenience wrapper around `sbatch train.sbatch`
├── train.sbatch      # the single entrypoint: source size+recipe+cluster, hand to common
├── lib/
│   ├── common.sh     # invariant scaffolding; assembles MEGATRON_ARGS and launches
│   └── dump-args.sh  # print the composed args for a (size,recipe) WITHOUT launching
├── clusters/
│   └── alps3.sh      # GH200 machine knobs (container, mpi, precision, sbatch flags)
├── sizes/            # model + data scale (270m-moe, 420m-moe, 810m-moe, 150m,
│                     #   the 128e scaling ladder *-moe-128e[-mla], ...; 128e-ladder.md)
└── recipes/
    ├── md_decoupling.sh   # the md-decoupling optimizer recipe (= submit_test.sh)
    ├── muon.sh            # Muon on matrices + AdamW scalars (dist_muon by default)
    └── _template.sh       # copy to add a new optimizer/idea
```

## Run it

```bash
# the main baseline at 420m, md_decoupling optimizer
bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling

# sweep a knob (any recipe var can be set from the env)
MLR=8e-3 bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling

# enable the extra per-step diagnostics (act/grad/neuron stats, ~20% slower; off
# by default) for a diagnostic run
EXTRA_LOGGING=1 bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling

# override nodes/time, chain jobs until TRAIN_SAMPLES is reached
bash _research/launch/framework/submit.sh --size 810m-moe --recipe md_decoupling --nodes 4 --auto-requeue

# inspect the composed args without submitting
bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling --dry-run
SIZE=420m-moe RECIPE=md_decoupling bash _research/launch/framework/lib/dump-args.sh
```

`submit.sh` reads `DEFAULT_NODES`/`DEFAULT_TIME` from the size file and the
per-cluster sbatch flags (`--account=infra01`, cpus, mem) from
`clusters/alps3.sh`, names the job/run `size-tag-knobs`, and submits with
`--dependency=singleton` (a resubmit of the same point queues instead of
double-writing the checkpoint dir).

## Debug it interactively

`debug.sh` is the interactive twin of `submit.sh`: same `--size/--recipe`, but it
runs inline via `torchrun` (no SLURM srun wrapper) for a few iters, so you can
iterate in an allocation. Grab a node **inside the framework's container**, then:

```bash
CONTAINER=/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-a139.toml
srun --account=infra01 --time=01:00:00 --nodes=1 --gpus-per-node=4 \
     --cpus-per-task=72 --mem=460000 --mpi=pmix \
     --network=disable_rdzv_get --environment=$CONTAINER --pty bash
# ... now inside the container, from the repo root ...

bash _research/launch/framework/debug.sh --size 270m-moe --recipe md_decoupling --iters 10
bash _research/launch/framework/debug.sh --size 270m-moe --recipe md_decoupling --dry-run
bash _research/launch/framework/debug.sh --size 270m-moe --recipe md_decoupling -- --lr 1e-4
```

A short run is just a small `TRAIN_SAMPLES = iters × GBS` override (the run stays
sample-based, identical in shape to a real launch). Defaults: 4 ranks, 20 iters,
10-min cap; debug runs land in a `debug-` wandb board. `debug.sh` calls
`_research/launch/interactive-run.sh`, which sources the sbatch with the srun
stubbed (`COMMON_NO_LAUNCH=1`) and launches `torchrun --standalone` directly.

## The md_decoupling recipe

`--optimizer md_decoupling`: hypersphere normalization (direction) on 2D
matrices + router, learnable per-axis gains (magnitude, softplus), and Muon
orthogonalized updates; embedding + LM head use the Adam branch. It is
**incompatible** with the standard `--use-distributed-optimizer`, so the recipe
sets `DIST_OPT_ARGS=(--use-layer-wise-distributed-optimizer --overlap-grad-reduce)`
to shard optimizer state. Learnable gains require `--ckpt-format torch` (set in
`common.sh`). Knobs: `LR` (base / scalar group), `MLR` (matrix group),
`GAINS_LR` (per-axis gains), `MIN_LR` (floor; `--min-lr-mode absolute` floors
every group at it).

This is the framework form of `submit_test.sh`'s `OPT_ARGS`, with the new
code's flag names (`--muon-use-nesterov`, `--md-router-use-orthogonal-updates`).

## Data & tokenizer

**Tokenizer**: the apertus multilingual 200k tokenizer (`preliminary_mul_200k`),
vendored as raw HF json under `_research/data/apertus-mul-200k-tokenizer/`
(`tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json`, pulled
from `swiss-ai/apertus-tokenizer-development`). Loaded from that local dir via
`--tokenizer-type HuggingFaceTokenizer` (no HF-hub/LFS access at runtime).
Override with `TOKENIZER_MODEL` (a local dir or a hub id).

**Data**: sources under `DATA_ROOT` are glob'd for `.bin/.idx` shards (possibly
nested) and handed to `--train-data-path` as a token-proportional blend. Pick a
blend three ways, most-specific-wins:
- `DATA_PRESET=<name>` — a named pre-tokenized blend defined in `common.sh`. The
  **128e ladder** sets `DATA_PRESET=fineweb2hq-mul200k` = the fineweb-2-hq mmbert
  quality_10 set (`_apertus_v2`, mul_200k-tokenized) under
  `/capstor/.../datasets_tokenized`. Default is the **fwedu** SPP-annotated split;
  set `INCLUDE_DCLM=1` to use the **dclm-edu** split instead (alternatives, not
  blended).
- `DATA_ROOT` + `DATA_SOURCES` — an explicit glob root + source list (the default
  when no preset: the swissai blend under `/iopsstor/scratch/cscs/jpcoles/a06`).
- `MEGATRON_DATA_PATH` — a single `--data-path` prefix (e.g. a GPT2BPE debug set;
  pair with a matching `TOKENIZER_MODEL` if the vocab must match the data).

## Container

`clusters/alps3.sh` defaults `CONTAINER` to gfu's pytorch2512 image (bundles
deep_gemm + emerging_optimizers + grouped-GEMM). An absolute `CONTAINER` path is
used as-is; a bare filename resolves under `_research/launch/` (e.g.
`CONTAINER=alps3.toml` for the skinny NGC fallback, FP8-disabled). Precision
defaults to **bf16** (matching the proven md_decoupling run); enable the
DeepSeek-V3 FP8 blockwise recipe with
`MIXED_PRECISION_ARGS=(--bf16 --fp8-format e4m3 --fp8-recipe blockwise)`.

## The contract (writing a size / recipe)

A **size** (`sizes/<name>.sh`) MUST set `NUM_LAYERS HIDDEN FFN_HIDDEN NUM_HEADS
NUM_KV_HEADS SEQ_LEN MBS GBS TRAIN_SAMPLES SAVE_INTERVAL APERTUS_TRACK` and
`MOE_ARGS=(...)` (empty `()` for dense — only expert *geometry*; the
DeepSeek-V3 routing policy is invariant and lives in `common.sh`). MAY set
`TP PP CP EP INIT_STD EXIT_DURATION_MINS DEFAULT_NODES DEFAULT_TIME DATA_PRESET`
and `MLA_ARGS=(...)` (non-empty → `common.sh` emits `--multi-latent-attention` +
the lora ranks / per-head dims and skips GQA; default is GQA via `NUM_KV_HEADS`).

The `*-moe-128e[-mla].sh` rungs are the fine-grained (128 experts, top-4)
scaling ladder for the 670B-A40B target — see `sizes/128e-ladder.md`. MLA wiring
is untested at runtime; smoke-test a small MLA rung before a real launch.

A **recipe** (`recipes/<name>.sh`) MUST set `OPTIMIZER`, `EXP_TAG`,
`RECIPE_ARGS=(...)`. MAY set `LR MIN_LR KNOB_STR WEIGHT_DECAY ADAM_BETA1
ADAM_BETA2 CLIP_GRAD LR_DECAY_STYLE LR_WARMUP_SAMPLES DIST_OPT_ARGS
EXTRA_REG_ARGS`. See `_template.sh`.
