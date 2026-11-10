import unittest

import torch
import torch.nn.functional as F

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

VOCAB_SIZE = 64
SEQ_LEN = 12


def build_config(**overrides) -> CEHRGPTConfig:
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=128,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=False,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    kwargs.update(overrides)
    return CEHRGPTConfig(**kwargs)


def make_padded_batch():
    input_ids = torch.randint(2, VOCAB_SIZE, (2, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    # Right-pad the second sequence by 5 positions
    input_ids[1, -5:] = 0
    attention_mask[1, -5:] = 0
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ages=torch.randint(20, 80, (2, SEQ_LEN)),
        labels=labels,
    )


class TestNextTokenLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_loss_matches_cross_entropy_with_ignore_index(self):
        model = CEHRGPT2LMHeadModel(build_config()).eval()
        batch = make_padded_batch()
        with torch.no_grad():
            out = model(**batch)
        shift_logits = out.logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = batch["labels"][:, 1:].reshape(-1)
        expected = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        torch.testing.assert_close(out.token_loss, expected)
        torch.testing.assert_close(out.loss, expected)

    def test_all_labels_ignored_gives_zero_loss(self):
        model = CEHRGPT2LMHeadModel(build_config()).eval()
        batch = make_padded_batch()
        batch["labels"] = torch.full_like(batch["labels"], -100)
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(out.token_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
