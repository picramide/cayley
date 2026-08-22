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

Run a sparse-mask RoBERTa benchmark:

```bash
python scripts/benchmark_roberta_glue.py \
  --task_name mrpc \
  --mask_path masks/hypercube_128.pt \
  --do_train --do_eval \
  --output_dir outputs/mrpc_hypercube \
  --results_file results/glue_runs.jsonl
```

Run the default dense plus mask grid over major GLUE tasks:

```bash
python scripts/run_glue_grid.py --tasks mrpc sst2 qnli cola rte stsb mnli
```

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
