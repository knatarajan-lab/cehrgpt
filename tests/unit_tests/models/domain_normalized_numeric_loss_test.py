import unittest

import torch
import torch.nn.functional as F

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel


class DomainNormalizedNumericLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = CEHRGPTConfig(
            vocab_size=12,
            value_vocab_size=5,
            n_positions=16,
            n_embd=12,
            n_layer=1,
            n_head=3,
            n_inner=24,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
            lab_token_ids=[5],
            include_values=True,
            use_sub_time_tokenization=False,
            include_ttv_prediction=False,
            include_motor_time_to_event=False,
            domain_normalize_numeric_losses=True,
            lab_token_loss_weight=0.5,
            value_prediction_loss_weight=0.75,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        self.model = CEHRGPT2LMHeadModel(self.config).eval()

    def test_losses_are_normalized_within_their_domains(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        values = torch.tensor([[0, 1, 2, 3, 4, 0]])
        value_indicators = torch.tensor([[False, True, False, True, False, False]])

        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            values=values,
            value_indicators=value_indicators,
            true_values=values,
            true_value_indicators=value_indicators,
            return_dict=True,
        )

        targets = input_ids[:, 1:].reshape(-1)
        per_token_loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, self.config.vocab_size),
            targets,
            reduction="none",
        )
        lab_mask = targets == 5
        expected_token_loss = (
            per_token_loss[~lab_mask].mean()
            + self.config.lab_token_loss_weight * per_token_loss[lab_mask].mean()
        )

        value_mask = value_indicators[:, :-1].reshape(-1)
        per_value_loss = F.cross_entropy(
            output.next_value_logits[:, :-1].reshape(-1, self.config.value_vocab_size),
            values[:, 1:].reshape(-1),
            reduction="none",
        )
        expected_value_loss = per_value_loss[value_mask].mean()

        torch.testing.assert_close(output.token_loss, expected_token_loss)
        torch.testing.assert_close(output.token_value_loss, expected_value_loss)
        torch.testing.assert_close(
            output.loss,
            expected_token_loss
            + self.config.value_prediction_loss_weight * expected_value_loss,
        )

    def test_missing_numeric_targets_produce_finite_loss(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        values = torch.zeros_like(input_ids)
        value_indicators = torch.zeros_like(input_ids, dtype=torch.bool)

        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            values=values,
            value_indicators=value_indicators,
            true_values=values,
            true_value_indicators=value_indicators,
            return_dict=True,
        )

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.token_value_loss.item(), 0.0)

    def test_legacy_token_weighting_is_unchanged_when_flag_is_disabled(self):
        self.model.config.domain_normalize_numeric_losses = False
        self.model.config.lab_token_penalty = True
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            return_dict=True,
        )

        targets = input_ids[:, 1:].reshape(-1)
        per_token_loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, self.config.vocab_size),
            targets,
            reduction="none",
        )
        lab_mask = targets == 5
        expected = torch.where(
            lab_mask,
            per_token_loss * self.config.lab_token_loss_weight,
            per_token_loss,
        ).mean()
        torch.testing.assert_close(output.token_loss, expected)

    def test_flag_serializes_in_config(self):
        restored = CEHRGPTConfig.from_dict(self.config.to_dict())
        self.assertTrue(restored.domain_normalize_numeric_losses)


if __name__ == "__main__":
    unittest.main()
