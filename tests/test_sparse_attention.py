import copy
import unittest

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from transformers.models.roberta import modeling_roberta

import cayley.roberta_sparse_attention as sparse


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.original = modeling_roberta.RobertaSelfAttention.forward
        self.config = RobertaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        )
        self.config._attn_implementation = 'eager'

    def tearDown(self):
        modeling_roberta.RobertaSelfAttention.forward = self.original
        if hasattr(modeling_roberta.RobertaSelfAttention, '_cayley_sparse_patch_applied'):
            del modeling_roberta.RobertaSelfAttention._cayley_sparse_patch_applied
        sparse.GLOBAL_SPARSE_MASK = None

    def test_full_mask_matches_original_logits_and_gradients(self):
        original = RobertaForSequenceClassification(self.config).eval()
        model = copy.deepcopy(original)
        ids = torch.tensor([[0, 5, 2, 1, 1], [0, 8, 9, 10, 2]])
        padding = ids.ne(1).long()
        labels = torch.tensor([0, 1])
        baseline = original(ids, attention_mask=padding, labels=labels)
        baseline.loss.backward()
        sparse.patch_roberta_attention()
        sparse.GLOBAL_SPARSE_MASK = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        actual = model(ids, attention_mask=padding, labels=labels)
        actual.loss.backward()
        torch.testing.assert_close(actual.logits, baseline.logits)
        for a, b in zip(original.parameters(), model.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad)
        changed = ids.clone()
        changed[0, 3:] = torch.tensor([20, 21])
        torch.testing.assert_close(model(changed, attention_mask=padding).logits, actual.logits)
        torch.testing.assert_close(model(ids[:1, :3], attention_mask=padding[:1, :3]).logits,
                                   actual.logits[:1])

    def test_boolean_and_additive_padding_masks_block_the_same_edges(self):
        sparse.patch_roberta_attention()
        attention = modeling_roberta.RobertaSelfAttention(self.config).eval()
        states = torch.randn(1, 4, 16)
        keep = torch.tensor([[[[True, True, False, False]]]])
        additive = torch.zeros(keep.shape).masked_fill(~keep, torch.finfo(torch.float32).min)
        sparse.GLOBAL_SPARSE_MASK = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        sparse.GLOBAL_SPARSE_MASK[:, :, 0, 1] = False
        out_bool, probs = attention(states, attention_mask=keep, output_attentions=True)
        out_float, _ = attention(states, attention_mask=additive, output_attentions=True)
        torch.testing.assert_close(out_bool, out_float)
        self.assertEqual(probs[..., 2:].count_nonzero().item(), 0)
        self.assertEqual(probs[:, :, 0, 1].count_nonzero().item(), 0)

    def test_fully_blocked_rows_have_zero_attention_and_finite_gradients(self):
        sparse.patch_roberta_attention()
        attention = modeling_roberta.RobertaSelfAttention(self.config).eval()
        states = torch.randn(1, 4, 16, requires_grad=True)
        keep = torch.tensor([[[[True, True, False, False]]]])
        sparse.GLOBAL_SPARSE_MASK = torch.eye(4, dtype=torch.bool)[None, None]
        output, probs = attention(states, attention_mask=keep, output_attentions=True)
        self.assertEqual(probs[:, :, 2:, :].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(states.grad).all())
        for parameter in attention.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
