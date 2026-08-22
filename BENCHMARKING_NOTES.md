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
