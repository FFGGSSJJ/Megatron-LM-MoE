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
├── sizes/            # model + data scale (270m-moe, 420m-moe, 810m-moe, 150m, ...)
└── recipes/
    ├── md_decoupling.sh   # the md-decoupling optimizer recipe (= submit_test.sh)
    └── _template.sh       # copy to add a new optimizer/idea
```

## Run it

```bash
# the main baseline at 420m, md_decoupling optimizer
bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling

# sweep a knob (any recipe var can be set from the env)
MLR=8e-3 bash _research/launch/framework/submit.sh --size 420m-moe --recipe md_decoupling

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

Default data is the swissai blend under `/iopsstor/scratch/cscs/jpcoles/a06`
(token-proportional across sources), tokenized with the apertus/swissai
HuggingFace tokenizer (`--tokenizer-type HuggingFaceTokenizer --tokenizer-model
alehc/swissai-tokenizer`). Override the tokenizer with `TOKENIZER_MODEL`, the
mixture with `DATA_ROOT`/`DATA_SOURCES`, or set `MEGATRON_DATA_PATH` for a
single `--data-path` prefix (e.g. the GPT2BPE climbmix debug set — pair it with
a matching tokenizer override since climbmix is GPT2BPE-tokenized).

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
`TP PP CP EP INIT_STD EXIT_DURATION_MINS DEFAULT_NODES DEFAULT_TIME`.

A **recipe** (`recipes/<name>.sh`) MUST set `OPTIMIZER`, `EXP_TAG`,
`RECIPE_ARGS=(...)`. MAY set `LR MIN_LR KNOB_STR WEIGHT_DECAY ADAM_BETA1
ADAM_BETA2 CLIP_GRAD LR_DECAY_STYLE LR_WARMUP_SAMPLES DIST_OPT_ARGS
EXTRA_REG_ARGS`. See `_template.sh`.
