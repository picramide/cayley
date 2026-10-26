import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from cayley.deberta_sparse_attention import (
    CayleyDebertaV2ForSequenceClassification,
    load_mask,
    pair_attention_mask,
)


class DebertaMaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = DebertaV2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4,
            position_biased_input=False, relative_attention=True,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        )

    def test_pair_mask_combines_padding(self):
        structural = torch.eye(5, dtype=torch.bool)
        structural[1, 2] = True
        padding = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        pair = pair_attention_mask(padding, structural)
        self.assertEqual(pair.shape, (2, 5, 5))
        self.assertTrue(pair[0, 1, 2])
        self.assertTrue(pair[1, 1, 2])
        self.assertEqual(pair[1, 3:].count_nonzero().item(), 0)
        self.assertEqual(pair[1, :, 3:].count_nonzero().item(), 0)

    def test_saved_mask_preserves_head_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.pt"
            mask = torch.eye(8, dtype=torch.bool).expand(4, -1, -1).clone()
            torch.save(mask, path)
            self.assertTrue(torch.equal(load_mask(path), mask[0]))
            mask[1, 0, 1] = True
            torch.save(mask, path)
            self.assertTrue(torch.equal(load_mask(path), mask))

    def test_head_specific_mask(self):
        model = CayleyDebertaV2ForSequenceClassification(self.config).eval()
        mask = torch.eye(8, dtype=torch.bool).expand(4, -1, -1).clone()
        mask[1, 0, 1] = True
        model.set_structural_mask(mask)
        ids = torch.tensor([[1, 5, 2, 0]])
        padding = ids.ne(0).long()
        allowed = pair_attention_mask(padding, mask)
        self.assertEqual(allowed.shape, (1, 4, 4, 4))
        with torch.no_grad():
            attentions = model(ids, attention_mask=padding, output_attentions=True).attentions
        for probabilities in attentions:
            self.assertEqual(probabilities.masked_select(~allowed).count_nonzero().item(), 0)

    def test_classifier_uses_position_zero(self):
        model = CayleyDebertaV2ForSequenceClassification(self.config).eval()
        states = torch.randn(2, 5, self.config.hidden_size)
        changed = states.clone()
        changed[:, 1:] += 10
        self.assertTrue(torch.equal(model.pooler(states), model.pooler(changed)))

    def test_dense_and_all_ones_are_exact(self):
        dense = CayleyDebertaV2ForSequenceClassification(self.config).eval()
        full = copy.deepcopy(dense).eval()
        full.set_structural_mask(torch.ones(8, 8, dtype=torch.bool))
        ids = torch.tensor([[1, 5, 2, 0, 0], [1, 8, 9, 10, 2]])
        padding = ids.ne(0).long()
        labels = torch.tensor([0, 1])
        first = dense(ids, attention_mask=padding, labels=labels)
        second = full(ids, attention_mask=padding, labels=labels)
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertTrue(torch.equal(first.loss, second.loss))
        first.loss.backward()
        second.loss.backward()
        for left, right in zip(dense.parameters(), full.parameters()):
            if left.grad is not None:
                self.assertTrue(torch.equal(left.grad, right.grad))

    def test_unmodified_dense_is_exact(self):
        original = DebertaV2ForSequenceClassification(self.config).eval()
        adapted = CayleyDebertaV2ForSequenceClassification(self.config).eval()
        adapted.load_state_dict(original.state_dict())
        ids = torch.tensor([[1, 5, 2, 0, 0], [1, 8, 9, 10, 2]])
        padding = ids.ne(0).long()
        labels = torch.tensor([0, 1])
        first = original(ids, attention_mask=padding, labels=labels)
        second = adapted(ids, attention_mask=padding, labels=labels)
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertTrue(torch.equal(first.loss, second.loss))
        first.loss.backward()
        second.loss.backward()
        for left, right in zip(original.parameters(), adapted.parameters()):
            if left.grad is not None:
                self.assertTrue(torch.equal(left.grad, right.grad))

    def test_blocked_probabilities_are_zero(self):
        model = CayleyDebertaV2ForSequenceClassification(self.config).eval()
        model.set_structural_mask(torch.eye(8, dtype=torch.bool))
        ids = torch.tensor([[1, 5, 2, 0, 0]])
        padding = ids.ne(0).long()
        with torch.no_grad():
            output = model(ids, attention_mask=padding, output_attentions=True)
        allowed = pair_attention_mask(padding, model.structural_mask)
        for probabilities in output.attentions:
            self.assertEqual(probabilities.masked_select(~allowed[:, None]).count_nonzero().item(), 0)
            self.assertTrue(torch.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
