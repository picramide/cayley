# Benchmarking Notes

## What the old scripts did

I found the previous RoBERTa benchmark code in:

- `/home/picramide/btp/roberta_eval.py`
- `/home/picramide/Downloads/roberta_eval.py`
- `/home/picramide/Downloads/roberta_eval_bigbird.py`

The core setup was:

- Hugging Face `RobertaForSequenceClassification`
- GLUE datasets through `datasets.load_dataset("glue", task_name)`
- Hugging Face `Trainer`
- GLUE metrics through `evaluate.load("glue", task_name)`
- dense baseline when no `--mask_path` is supplied
- sparse quality run when `--mask_path` is supplied
- mask semantics: `True` means keep an attention edge, `False` means block it
- mask shapes accepted: `[seq, seq]`, `[heads, seq, seq]`, `[1, heads, seq, seq]`

The previous mask patch applied RoBERTa's normal padding mask first, then applied
the structural sparse mask before softmax. That is the right order for measuring
task quality under a fixed attention graph.

## Important limitation

The patch still computes dense attention scores with `QK^T` and then masks them.
So these scripts benchmark downstream task quality, not sparse-kernel speed or
memory savings. That is acceptable for choosing promising Cayley graphs before
writing or using an optimized sparse attention kernel.

## Replicated setup in this repo

- `scripts/generate_masks.py` creates boolean keep masks.
- `scripts/benchmark_roberta_glue.py` trains/evaluates RoBERTa on GLUE with the
  same mask semantics as the previous scripts.
- `scripts/run_glue_grid.py` runs every supported GLUE benchmark over every
  built-in attention pattern, including the dense baseline.
- `cayley/roberta_sparse_attention.py` contains the RoBERTa attention patch.
- `cayley/masks.py` contains reusable mask builders.

Dense baseline:

```bash
python scripts/benchmark_roberta_glue.py \
  --task_name mrpc \
  --do_train --do_eval \
  --output_dir outputs/mrpc_dense \
  --results_file results/glue_runs.jsonl
```

Sparse run:

```bash
python scripts/generate_masks.py \
  --kind hypercube \
  --seq 128 \
  --output masks/hypercube_128.pt

python scripts/benchmark_roberta_glue.py \
  --task_name mrpc \
  --mask_path masks/hypercube_128.pt \
  --verbose_mask \
  --do_train --do_eval \
  --output_dir outputs/mrpc_hypercube \
  --results_file results/glue_runs.jsonl
```

For fair comparisons, keep seed, max length, batch size, learning rate, epochs,
and training/eval split identical between dense and sparse runs.

BigBird defaults to blocks of 2 tokens, the first 2 tokens as one global block,
3 random blocks per query block per head, and a local window covering the query
block plus one block on each side. Random blocks exclude global and local blocks
and are sampled without replacement (up to the number of available blocks).
The global tokens attend to all tokens and are visible to all tokens.

Earlier defaults used only 1 random block. The all-masks runner now records
mask-generation settings and stores generated masks under the output directory.
Each result includes a hash of the mask file used.

Masks are generated for the maximum sequence length and cropped for shorter
batches; padding is also masked. Thus, 3 sampled random blocks do not guarantee
3 usable random blocks for each shorter example.

## Corrected benchmark protocol (version 2)

Dense and sparse runs explicitly use eager attention. The sparse patch accepts
both additive and boolean padding masks, blocks padding before softmax, and
handles fully blocked queries without NaNs. Earlier default SDPA configurations
could either bypass the patch (Transformers 4) or supply boolean masks that the
patch incorrectly added to scores (Transformers 5). Affected models must be
retrained; reevaluating old weights does not undo incorrect training.

Training and data sampling both use the requested seed. Learning rate, epoch
count and warmup remain unchanged; these fixes do not guarantee that an RTE run
will converge. Compare multiple seeds with the same recipe for every mask.

Training with evaluation now restores the best validation checkpoint before
exporting and reporting results. The selection metric is F1 for MRPC/QQP,
Matthews correlation for CoLA, Pearson correlation for STS-B, and accuracy for
the other tasks. MNLI selects on matched validation accuracy and then reports
both matched and mismatched validation results. All selection metrics are
maximized. These are validation-selected scores, not held-out test scores.
The best checkpoint is also restored through `from_pretrained` with a strict
weight copy before export/evaluation. This handles RoBERTa LayerNorm key aliases
that Transformers 5.3's raw Trainer checkpoint loader otherwise misses.

Results record the protocol version, actual seeds, library versions, attention
backend, best checkpoint/epoch/metric and training metrics. Classification
metrics include per-class prediction counts to help identify constant predictors.
The run's `trainer_state.json` preserves per-epoch evaluation and logged training
loss/gradient history independently of checkpoint rotation.

`benchmark_all_masks.py --skip_completed` only reuses completed train/eval runs
from protocol version 2 with matching task, model/data paths, mask settings and
training recipe. Older rows remain in the JSONL but do not suppress new runs.
Use fresh output/results directories to keep historical comparisons separate.
The runner uses the active Python interpreter, honors `--dry_run` without
creating outputs or launching jobs, and exits nonzero if any benchmark fails.

Offline rerun on the server, using new directories:

```bash
python scripts/benchmark_all_masks.py \
  --offline_data_dir /home/22cs30008/offline \
  --output_dir outputs/benchmark_v2 \
  --results_dir results/benchmark_v2 \
  --skip_completed
```

Small offline regression tests (no pretrained downloads):

```bash
python -m unittest discover -s tests -v
```

Full grid:

```bash
python scripts/run_glue_grid.py
```

This writes one JSON object per run to `results/glue_grid.jsonl`, including
`task_name`, `mask_name`, `mask_path`, and the resulting metrics.
