import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

from cayley.benchmarking import BENCHMARK_PROTOCOL_VERSION
from cayley.glue import get_task_config
from cayley.masks import BigBirdMaskConfig, build_bigbird_mask


class OfflineBenchmarkTests(unittest.TestCase):
    """Exercise the real CLI without network access or pretrained downloads."""

    def test_classification_regression_and_mnli_restore_best_checkpoint(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3,
                     'the': 4, 'cat': 5, 'sat': 6, 'dog': 7, 'ran': 8}
            backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
            backend.pre_tokenizer = Whitespace()
            backend.post_processor = TemplateProcessing(
                single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>',
                special_tokens=[('<s>', 0), ('</s>', 2)],
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token='<pad>', bos_token='<s>',
                eos_token='</s>', unk_token='<unk>',
                model_input_names=['input_ids', 'attention_mask'],
            )
            tokenizer.save_pretrained(root / 'model')
            torch.manual_seed(42)
            config = RobertaConfig(
                vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                num_attention_heads=4, num_hidden_layers=1, num_labels=2,
                max_position_embeddings=64,
            )
            RobertaForSequenceClassification(config).save_pretrained(root / 'model')
            torch.save(build_bigbird_mask(32, 4, BigBirdMaskConfig()), root / 'mask.pt')
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo), os.environ.get('PYTHONPATH', '')]), HF_HUB_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            for task, mask in [('rte', 'dense'), ('rte', 'bigbird'), ('stsb', 'bigbird'), ('mnli', 'bigbird')]:
                with self.subTest(task=task, mask=mask):
                    task_config = get_task_config(task)
                    labels = [float(i) for i in range(6)] if task == 'stsb' else [i % task_config['num_labels'] for i in range(6)]
                    data = Dataset.from_dict({
                        task_config['sentence1_key']: ['the cat sat', 'the dog'] * 3,
                        task_config['sentence2_key']: ['cat', 'the dog ran'] * 3,
                        'label': labels,
                    })
                    splits = {'train': data}
                    if task == 'mnli':
                        splits.update(validation_matched=data, validation_mismatched=data)
                    else:
                        splits['validation'] = data
                    path = root / f'{task}_{mask}'
                    DatasetDict(splits).save_to_disk(path / 'dataset')
                    output = path / 'output'
                    result_file = path / 'results.jsonl'
                    cmd = [sys.executable, str(repo / 'scripts/benchmark_roberta_glue.py'),
                           '--model_name', str(root / 'model'), '--dataset_name', str(path / 'dataset'),
                           '--task_name', task, '--do_train', '--do_eval', '--num_train_epochs', '2',
                           '--max_length', '32', '--per_device_train_batch_size', '3',
                           '--per_device_eval_batch_size', '3', '--seed', '17', '--mask_name', mask,
                           '--output_dir', str(output), '--results_file', str(result_file)]
                    if mask != 'dense':
                        cmd += ['--mask_path', str(root / 'mask.pt')]
                    run = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=120)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    result = json.loads(result_file.read_text())
                    self.assertEqual(result['benchmark_protocol_version'], BENCHMARK_PROTOCOL_VERSION)
                    self.assertEqual(result['trainer_seed'], 17)
                    self.assertEqual(result['data_seed'], 17)
                    self.assertEqual(result['attention_implementation'], 'eager')
                    self.assertTrue(result['load_best_model_at_end'])
                    self.assertEqual(result['metric_for_best_model'], task_config['metric_for_best_model'])
                    self.assertIn(result['best_epoch'], (1.0, 2.0))
                    self.assertIn('train_loss', result['train_metrics'])
                    self.assertEqual(result['mask_sha256'] is None, mask == 'dense')
                    history = json.loads(Path(result['trainer_state_path']).read_text())['log_history']
                    self.assertTrue({1.0, 2.0}.issubset({e['epoch'] for e in history if 'eval_loss' in e}))
                    metric_key = 'eval_' + task_config['metric_for_best_model']
                    self.assertAlmostEqual(result['best_metric'], max(e[metric_key] for e in history if metric_key in e))
                    if task != 'stsb':
                        prefix = 'eval_matched' if task == 'mnli' else 'eval'
                        counts = [result['metrics'][f'{prefix}_predicted_class_{i}_count'] for i in range(task_config['num_labels'])]
                        self.assertEqual(sum(counts), 6)
                    if task == 'mnli':
                        self.assertIn('eval_mismatched_accuracy', result['metrics'])
                    # The exported model must actually be the selected checkpoint,
                    # including when it predates the final training epoch.
                    saved = RobertaForSequenceClassification.from_pretrained(output)
                    best = RobertaForSequenceClassification.from_pretrained(result['best_checkpoint'])
                    for name, a in saved.state_dict().items():
                        torch.testing.assert_close(a, best.state_dict()[name], rtol=0, atol=0,
                                                   msg=lambda message: f'{name}: {message}\n{run.stderr}')


if __name__ == '__main__':
    unittest.main()
