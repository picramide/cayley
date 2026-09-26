import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cayley.benchmarking import BENCHMARK_PROTOCOL_VERSION, run_key
from scripts import benchmark_all_masks as runner


class RunnerTests(unittest.TestCase):
    def test_dry_run_never_launches_processes_or_creates_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'outputs'
            results = Path(tmp) / 'results'
            argv = ['benchmark_all_masks.py', '--dry_run', '--output_dir', str(output),
                    '--results_dir', str(results), '--offline_data_dir', '/offline']
            original = copy.deepcopy(runner.BENCHMARKS)
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run') as run:
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    runner.main()
            run.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(results.exists())
            self.assertEqual(runner.BENCHMARKS, original)
            self.assertIn('--num_random_blocks 3', stdout.getvalue())
            self.assertIn('/offline/glue/rte', stdout.getvalue())

    def test_resume_requires_current_protocol_and_matching_recipe(self):
        current = {
            **runner.BENCHMARKS[3], 'mask_name': 'bigbird',
            'mask_config': {'kind': 'bigbird', 'args': ['--num_random_blocks', '3']},
            'benchmark_protocol_version': BENCHMARK_PROTOCOL_VERSION,
            'do_train': True, 'do_eval': True, 'metrics': {'eval_accuracy': 0.5},
            'weight_decay': 0.01,
        }
        old = {k: v for k, v in current.items() if k != 'benchmark_protocol_version'}
        eval_only = {**current, 'do_train': False, 'seed': 123}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in (old, current, eval_only)))
            completed = runner.load_existing_results(path)
        self.assertEqual(completed, {run_key(current)})
        for change in ({'seed': 17}, {'learning_rate': 1e-5}, {'max_train_samples': 8},
                       {'num_train_epochs': 3.0},
                       {'model_name': '/different/model'},
                       {'mask_config': {'kind': 'bigbird', 'args': ['--num_random_blocks', '1']}}):
            self.assertNotIn(run_key({**current, **change}), completed)


if __name__ == '__main__':
    unittest.main()
