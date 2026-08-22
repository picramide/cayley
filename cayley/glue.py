import numpy as np


SUPPORTED_GLUE_TASKS = ["mrpc", "sst2", "mnli", "stsb", "qnli", "cola", "rte", "qqp"]


def normalize_task_name(task_name: str) -> str:
    normalized = task_name.lower().strip()
    if normalized == "sts-b":
        normalized = "stsb"
    if normalized not in SUPPORTED_GLUE_TASKS:
        raise ValueError(
            f"Unsupported task_name '{task_name}'. Supported tasks: {', '.join(SUPPORTED_GLUE_TASKS)}"
        )
    return normalized


def get_task_config(task_name: str) -> dict[str, object]:
    configs = {
        "mrpc": ("sentence1", "sentence2", 2, False),
        "qqp": ("question1", "question2", 2, False),
        "qnli": ("question", "sentence", 2, False),
        "cola": ("sentence", None, 2, False),
        "rte": ("sentence1", "sentence2", 2, False),
        "sst2": ("sentence", None, 2, False),
        "mnli": ("premise", "hypothesis", 3, False),
        "stsb": ("sentence1", "sentence2", 1, True),
    }
    sentence1_key, sentence2_key, num_labels, is_regression = configs[task_name]
    return {
        "sentence1_key": sentence1_key,
        "sentence2_key": sentence2_key,
        "num_labels": num_labels,
        "is_regression": is_regression,
    }


def preprocess_examples(examples, tokenizer, max_length: int, sentence1_key: str, sentence2_key: str | None):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        max_length=max_length,
    )


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def binary_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    true_positive = int(((preds == 1) & (labels == 1)).sum())
    false_positive = int(((preds == 1) & (labels == 0)).sum())
    false_negative = int(((preds == 0) & (labels == 1)).sum())
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float((2 * true_positive) / denominator)


def matthews_corrcoef(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0
    return float(((tp * tn) - (fp * fn)) / np.sqrt(denominator))


def pearson_corr(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.astype(np.float64)
    labels = labels.astype(np.float64)
    if preds.size < 2 or np.std(preds) == 0.0 or np.std(labels) == 0.0:
        return 0.0
    return float(np.corrcoef(preds, labels)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr(preds: np.ndarray, labels: np.ndarray) -> float:
    return pearson_corr(average_ranks(preds), average_ranks(labels))


def compute_metrics_fn(task_name: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.asarray(labels)
        if task_name == "stsb":
            preds = np.asarray(logits).squeeze(-1)
            return {
                "pearson": pearson_corr(preds, labels),
                "spearmanr": spearman_corr(preds, labels),
            }

        preds = np.argmax(logits, axis=-1)
        if task_name in {"mrpc", "qqp"}:
            return {
                "accuracy": accuracy(preds, labels),
                "f1": binary_f1(preds, labels),
            }
        if task_name == "cola":
            return {"matthews_correlation": matthews_corrcoef(preds, labels)}
        return {"accuracy": accuracy(preds, labels)}

    return compute_metrics
