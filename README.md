# Cayley RoBERTa Benchmarks

Compare dense and sparse attention masks on RoBERTa-base across eight GLUE tasks.

```bash
pip install -e .
python scripts/benchmark_all_masks.py --offline_data_dir /path/to/offline
```

The runner saves masks and checkpoints in `outputs/benchmark_all/` and metrics in `results/benchmark_all.jsonl` by default.
