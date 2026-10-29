import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.check_deberta_rtd_auroc import RTDHead, count_unigrams, make_validation_inputs


class Tokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id = 0
    sep_token_id = 2

    def __call__(self, texts, add_special_tokens=False):
        vocab = {"a": 3, "b": 4, "c": 5, "special": 2}
        return {"input_ids": [[vocab[word] for word in text.split()] for text in texts]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [1, *ids, 2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1, *([0] * len(ids)), 1]


class RTDAurocTests(unittest.TestCase):
    def test_head_uses_position_zero_for_each_token(self):
        torch.manual_seed(0)
        config = SimpleNamespace(hidden_size=8, layer_norm_eps=1e-7, hidden_act="gelu")
        head = RTDHead(config).eval()
        hidden = torch.randn(1, 4, 8)
        altered = hidden.clone()
        altered[0, 0, :4] += 5
        with torch.no_grad():
            original = head(hidden)
            changed = head(altered)
        self.assertEqual(tuple(original.shape), (1, 4))
        self.assertFalse(torch.equal(original[:, 1:], changed[:, 1:]))

    def test_unigram_replacements_exclude_special_tokens(self):
        tokenizer = Tokenizer()
        train = [{"text": "a b a special"}, {"text": "b c"}, {"text": " "}]
        counts = count_unigrams(train, tokenizer, vocab_size=6)
        np.testing.assert_array_equal(counts, [0, 0, 0, 2, 2, 1])

        validation = [{"text": "a b c a b c a b c a b c"}]
        first = make_validation_inputs(validation, tokenizer, counts, seed=42, length=8)
        second = make_validation_inputs(validation, tokenizer, counts, seed=42, length=8)
        for left, right in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(left, right)
        inputs, padding, eligible, labels, selected = first
        self.assertEqual(inputs.shape, (2, 8))
        self.assertEqual(selected, round(0.15 * eligible.sum()))
        self.assertTrue(np.all(inputs[eligible] >= 3))
        self.assertFalse(labels[~eligible].any())
        self.assertTrue(padding[:, 0].all())


if __name__ == "__main__":
    unittest.main()
