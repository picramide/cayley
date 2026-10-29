import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import DebertaV2Config, PreTrainedTokenizerFast, RobertaConfig

from scripts import benchmark_controlled as study


def tiny_model(name):
    config = dict(vocab_size=10, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                  num_attention_heads=4, max_position_embeddings=130, pad_token_id=1,
                  hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    config = (RobertaConfig(**config) if name == "roberta" else DebertaV2Config(
        **config, position_biased_input=False, relative_attention=True, max_relative_positions=16,
        pos_att_type=["c2p", "p2c"], pooler_hidden_size=16, share_att_key=True,
        norm_rel_ebd="layer_norm", position_buckets=8,
    ))
    return study.MODELS[name](config)


def tokenizer_fixture():
    backend = Tokenizer(WordLevel({"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
                                   "a": 4, "b": 5, "c": 6}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> $B </s>",
                                                 special_tokens=[("<s>", 0), ("</s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<s>",
                                  eos_token="</s>", unk_token="<unk>",
                                  model_input_names=["input_ids", "attention_mask"])


class ControlledBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tearDown(self):
        study.roberta_attention.GLOBAL_SPARSE_MASK = None

    def test_saved_split_shared_disjoint_and_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Dataset.from_dict({"sentence": [str(i) for i in range(100)], "label": [0] * 70 + [1] * 30})
            train, held, path = study.saved_split(data, "cola", Path(tmp))
            self.assertEqual((len(train), len(held)), (90, 10))
            self.assertEqual(sum(data["label"][i] for i in held), 3)
            self.assertFalse(set(train) & set(held))
            self.assertEqual(study.saved_split(data, "cola", Path(tmp)), (train, held, path))
            regression = Dataset.from_dict({"label": np.linspace(0, 5, 23)})
            self.assertEqual(len(study.saved_split(regression, "stsb", Path(tmp))[1]), 3)
            with self.assertRaisesRegex(ValueError, "does not match"):
                study.saved_split(data.select(range(50)), "cola", Path(tmp))

    def test_gradual_bias_and_hard_padding_on_both_models(self):
        for name in study.MODELS:
            with self.subTest(model=name):
                model = tiny_model(name).eval()
                mask = torch.eye(128, dtype=torch.bool)
                schedule = study.MaskSchedule(model, name, mask, "G")
                ids = torch.tensor([[0, 4, 5, 2, 1, 1], [0, 6, 5, 4, 4, 2]])
                inputs = dict(input_ids=ids, attention_mask=ids.ne(1).long(), labels=torch.tensor([0, 1]))
                seen = []
                handle = schedule.layers[0].attention.self.dropout.register_forward_pre_hook(
                    lambda module, args: seen.append(args[0].detach().clone()))
                schedule.condition = "P"
                schedule.mask = None
                schedule.apply()
                dense = model(**inputs)
                dense.loss.backward()
                gradients = {key: value.grad.clone() for key, value in model.named_parameters() if value.grad is not None}
                model.zero_grad(set_to_none=True)
                schedule.condition, schedule.mask = "G", mask
                schedule.apply(training=True, step=0, total=100)
                soft = model(**inputs)
                soft.loss.backward()
                torch.testing.assert_close(dense.logits, soft.logits, rtol=0, atol=0)
                for key, value in model.named_parameters():
                    if key in gradients:
                        torch.testing.assert_close(gradients[key], value.grad, rtol=0, atol=0)
                schedule.apply(training=True, step=25, total=100)
                model(**inputs)
                bias = torch.from_numpy(study.soft_mask_bias(mask.numpy(), 10))[:6, :6]
                expected = seen[0] * bias.exp()
                expected = expected / expected.sum(-1, keepdim=True).clamp_min(1e-30)
                torch.testing.assert_close(seen[-1], expected, rtol=1e-5, atol=1e-7)
                self.assertTrue((seen[-1][0, :, :4, 4:] == 0).all())
                self.assertTrue((seen[-1][:, :, 0, 1] > 0).all())
                schedule.apply(training=False, step=1, total=100)
                hard = model(**inputs)
                self.assertTrue((seen[-1].masked_select(~mask[:6, :6][None, None]) == 0).all())
                schedule.apply(training=True, step=50, total=100)
                torch.testing.assert_close(hard.logits, model(**inputs).logits, rtol=0, atol=0)
                handle.remove()

    def test_real_trainer_selection_order_and_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = tokenizer_fixture()
            data = Dataset.from_dict({"sentence1": ["a b", "a", "c a b"] * 10,
                                      "sentence2": ["c", "a b"] * 15, "label": [0, 1] * 15})
            split = study.saved_split(data, "rte", root / "splits")
            encoded = study.encode(data, tokenizer, "rte")
            datasets = {"train": encoded, "validation": encoded.select(range(4))}
            for model_name in study.MODELS:
                model_path = root / model_name
                initial = tiny_model(model_name)
                (initial.deberta if model_name == "deberta" else initial).save_pretrained(model_path)
                heads, orders, steps = [], [], []
                for condition in ("P", "S", "S+", "G"):
                    output = root / f"{model_name}_{condition}"
                    output.mkdir()
                    args = SimpleNamespace(model=model_name, model_path=model_path, task="rte",
                                           condition=condition, precision="fp32")
                    order = []
                    original_iter = study.EpochSampler.__iter__
                    original_load = study.StudyTrainer._load_best_model

                    def record(sampler):
                        indices = list(original_iter(sampler))
                        order.append((sampler.epoch, indices))
                        return iter(indices)

                    def check_restore(trainer):
                        original_load(trainer)
                        expected = type(trainer.model).from_pretrained(trainer.state.best_model_checkpoint)
                        for key, value in trainer.model.state_dict().items():
                            torch.testing.assert_close(value.cpu(), expected.state_dict()[key], rtol=0, atol=0)

                    calls = []

                    def metrics(_prediction):
                        calls.append(1)
                        return {"accuracy": 1.0 / len(calls)}

                    with patch.object(study.EpochSampler, "__iter__", record), \
                            patch.object(study.StudyTrainer, "_load_best_model", check_restore), \
                            patch.object(study, "compute_metrics_fn", return_value=metrics):
                        runtime, _ = study.run_one(args, "local", 1, tokenizer, datasets, split,
                                                  torch.eye(128, dtype=torch.bool), "test", output)
                    self.assertGreater(runtime, 0)
                    state = json.loads((output / "trainer_state.json").read_text())
                    self.assertEqual(state["best_metric"], 1.0)
                    self.assertTrue(state["best_model_checkpoint"].endswith("checkpoint-4"))
                    self.assertEqual(len([row for row in state["log_history"] if "eval_loss" in row]), 15 if condition == "S+" else 5)
                    self.assertTrue(any("loss" in row for row in state["log_history"]))
                    rows = [json.loads(line) for line in (output / "dev_predictions.jsonl").read_text().splitlines()]
                    self.assertEqual([row["idx"] for row in rows], list(range(4)))
                    self.assertEqual([row["length"] for row in rows], [sum(row) for row in datasets["validation"]["attention_mask"]])
                    self.assertTrue(all(isinstance(row["pred"], int) for row in rows))
                    heads.append(json.loads((output / "run_config.json").read_text())["initial_head_sha256"])
                    saved_args = json.loads((output / "run_config.json").read_text())["training_args"]
                    self.assertEqual(saved_args["warmup_steps"], 0 if condition == "P" else (6 if condition == "S+" else 2))
                    orders.append(order[:5])
                    steps.append(state["global_step"])
                self.assertEqual(len(set(heads)), 1)
                self.assertTrue(all(order == orders[0] for order in orders))
                self.assertEqual([epoch for epoch, _ in orders[0]], list(range(5)))
                self.assertEqual(steps, [20, 20, 60, 20])

                regression = data.remove_columns("label").add_column("label", [float(i % 6) for i in range(30)])
                reg_split = study.saved_split(regression, "stsb", root / "splits")
                encoded_reg = study.encode(regression, tokenizer, "stsb")
                output = root / f"{model_name}_stsb"
                output.mkdir()
                args = SimpleNamespace(model=model_name, model_path=model_path, task="stsb", condition="P", precision="fp32")
                study.run_one(args, "dense", 1, tokenizer, {"train": encoded_reg, "validation": encoded_reg.select(range(4))},
                              reg_split, None, None, output)
                rows = [json.loads(line) for line in (output / "dev_predictions.jsonl").read_text().splitlines()]
                self.assertTrue(all(isinstance(row["pred"], float) and isinstance(row["label"], float) for row in rows))

    def test_repository_masks_match_audits(self):
        for model in study.MODELS:
            for name in study.MASKS:
                mask, digest = study.audited_mask(model, "cola", name)
                self.assertEqual(mask is None, name == "dense")
                self.assertEqual(digest is None, name == "dense")

    def test_cli_preserves_completed_runs_and_crash_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            tiny_model("roberta").save_pretrained(model_path)
            tokenizer_fixture().save_pretrained(model_path)
            data = Dataset.from_dict({"sentence1": ["a b", "a"] * 10,
                                      "sentence2": ["c", "a b"] * 10, "label": [0, 1] * 10})
            DatasetDict(train=data, validation=data.select(range(4))).save_to_disk(root / "glue" / "rte")
            argv = ["benchmark_controlled.py", "--model", "roberta", "--task", "rte", "--masks", "local",
                    "--model-path", str(model_path), "--data-root", str(root / "glue"),
                    "--output-dir", str(root / "runs"), "--precision", "fp32"]
            output = root / "runs" / "roberta/exp1/rte/local/P/seed1_fp32"
            with patch.object(study, "ROOT", root), patch.object(study, "validate_request"), \
                    patch.object(study, "audited_mask", return_value=(torch.eye(128, dtype=torch.bool), "test")):
                with patch.object(sys, "argv", argv), \
                        patch.object(study.StudyTrainer, "train", side_effect=FloatingPointError("NaN")):
                    with self.assertRaises(FloatingPointError):
                        study.main()
                self.assertIn("NaN", json.loads((output / "crash.json").read_text())["error"])
                with patch.object(sys, "argv", argv + ["--retry-crashed"]):
                    study.main()
                self.assertEqual(len(list(output.parent.glob("seed1_fp32_crashed_*/crash.json"))), 1)
                original = (output / "dev_predictions.jsonl").read_bytes()
                with patch.object(sys, "argv", argv + ["--skip-completed"]):
                    study.main()
                self.assertEqual((output / "dev_predictions.jsonl").read_bytes(), original)
                rows = [json.loads(line) for line in (root / "runs/results.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(set(rows[0]), {"model", "experiment", "task", "mask", "condition", "seed",
                                               "precision", "run_dir", "train_runtime_s"})
                self.assertEqual(root / "runs" / rows[0]["run_dir"], output)
                with patch.object(sys, "argv", argv + ["--retry-crashed"]):
                    with self.assertRaises(FileExistsError):
                        study.main()


if __name__ == "__main__":
    unittest.main()
