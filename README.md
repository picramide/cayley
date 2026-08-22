# Cayley RoBERTa Benchmarks

Environment setup for testing Cayley graph based transformer patterns in `roberta-base` on Google Colab.

## Benchmark scripts

Generate one mask:

```bash
python scripts/generate_masks.py --kind hypercube --seq 128 --output masks/hypercube_128.pt
```

Run dense RoBERTa on one GLUE task:

```bash
python scripts/benchmark_roberta_glue.py --task_name mrpc --do_train --do_eval --output_dir outputs/mrpc_dense
```

Run benchmark X on mask Y by passing `--task_name`, `--mask_name`, and
optionally `--mask_path` to `scripts/benchmark_roberta_glue.py`.

Full MRPC dense run:

```bash
python scripts/benchmark_roberta_glue.py \
  --task_name mrpc \
  --mask_name dense \
  --do_train --do_eval \
  --output_dir outputs/mrpc_dense \
  --results_file results/mrpc_dense.jsonl
```

Full MRPC local-window run:

```bash
python scripts/generate_masks.py --kind local --seq 128 --window 16 --output masks/local_w16_128.pt

python scripts/benchmark_roberta_glue.py \
  --task_name mrpc \
  --mask_name local_w16 \
  --mask_path masks/local_w16_128.pt \
  --do_train --do_eval \
  --output_dir outputs/mrpc_local_w16 \
  --results_file results/mrpc_local_w16.jsonl
```

The Colab notebook currently runs those two MRPC commands and downloads the two
separate result files.

Run a sparse-mask RoBERTa benchmark:

```bash
python scripts/benchmark_roberta_glue.py \
  --task_name mrpc \
  --mask_path masks/hypercube_128.pt \
  --do_train --do_eval \
  --output_dir outputs/mrpc_hypercube \
  --results_file results/glue_runs.jsonl
```

Run all supported GLUE benchmarks over every built-in attention pattern, including dense:

```bash
python scripts/run_glue_grid.py
```

By default this runs `mrpc`, `sst2`, `mnli`, `stsb`, `qnli`, `cola`, `rte`, and `qqp` over:

- dense baseline with no mask
- dense boolean mask
- hypercube Cayley mask
- circulant Cayley mask
- window plus dilation Cayley mask
- random circulant Cayley masks with degrees 8 and 16
- local window mask
- BigBird-style mask

For a quick smoke test:

```bash
python scripts/run_glue_grid.py --tasks mrpc --max_train_samples 64 --max_eval_samples 64 --epochs 0.1
```

The Colab notebook version of this smoke run automatically downloads
`results/smoke_all_variants.jsonl` to your browser after the benchmark command
finishes.

See `BENCHMARKING_NOTES.md` for what was copied from the previous benchmarking setup and what the patch measures.

## Colab

Open `colab_roberta_cayley_setup.ipynb` in Colab or in VS Code with the Google Colab extension.

The notebook can load project files from:

- Google Drive
- GitHub
- uploaded zip

For GitHub-based runs, set:

```python
PROJECT_SOURCE = "github"
GITHUB_REPO = "https://github.com/picramide/cayley.git"
GITHUB_BRANCH = "main"
```

If the repo is private, add a Colab Secret named `GITHUB_TOKEN` with read access to the repository contents before running the clone cell.

See `COLAB.md` for details.
