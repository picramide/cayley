import numpy as np
import evaluate


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


def compute_metrics_fn(task_name: str):
    glue_metric = evaluate.load("glue", task_name)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task_name == "stsb":
            preds = logits.squeeze(-1)
        else:
            preds = np.argmax(logits, axis=-1)
        return glue_metric.compute(predictions=preds, references=labels)

    return compute_metrics
