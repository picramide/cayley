"""Shared, dependency-free benchmark identity for safe resumption."""

import json


BENCHMARK_PROTOCOL_VERSION = 2


def run_key(result: dict) -> tuple:
    """Do not reuse results from another protocol, task, mask, or recipe."""
    fields = (
        "benchmark_protocol_version", "task_name", "mask_name", "mask_config",
        "dataset_name", "model_name", "seed", "max_length", "learning_rate",
        "num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size",
        "weight_decay", "max_train_samples", "max_eval_samples",
    )
    return tuple(json.dumps(result.get(field), sort_keys=True) for field in fields)
