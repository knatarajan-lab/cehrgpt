import itertools
import unittest

import torch

from cehrgpt.tools.linear_prob import attention_rollout as ar
from tests.unit_tests.models.motor_tte_loss_test import build_model


def random_model(backbone, linear_prob_n_layer):
    model = build_model(
        backbone=backbone,
        n_layer=3,
        linear_prob_n_layer=linear_prob_n_layer,
        activation_function="silu" if backbone == "qwen2" else "gelu_new",
        decoder_mlp="LlamaMLP" if backbone == "qwen2" else "GPT2MLP",
        attn_implementation="eager",
    ).eval()
    # The default initialisation attends almost uniformly, which would hide a wrong formula
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() >= 2:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.4)
    return model


def padded_batch():
    """Row 0 has 5 tokens and is right padded, row 1 has 6 and is left padded."""
    generator = torch.Generator().manual_seed(2)
    input_ids = torch.randint(9, 60, (2, 8), generator=generator)
    attention_mask = torch.ones((2, 8))
    attention_mask[0, 5:] = 0
    attention_mask[1, :2] = 0
    input_ids[attention_mask == 0] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class AttentionRolloutTest(unittest.TestCase):
    def test_hand_computed_one_layer(self):
        self_attention = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]])
        cross_attention = torch.tensor([[[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]]])
        mask = torch.ones((1, 3))
        # 0.5 * A + 0.5 * I = [[1, 0, 0], [0.25, 0.75, 0], [0.1, 0.15, 0.75]]
        # [0.2, 0.3, 0.5] @ that = [0.2 + 0.075 + 0.05, 0.225 + 0.075, 0.375]
        scores = ar.linear_prob_attention_rollout([self_attention], [cross_attention], mask)
        torch.testing.assert_close(scores, torch.tensor([[0.325, 0.3, 0.375]]))
        # Without the probe it is the row of the last token
        local = ar.linear_prob_attention_rollout([self_attention], [], mask)
        torch.testing.assert_close(local, torch.tensor([[0.1, 0.15, 0.75]]))
        # Uniform causal attention on two tokens: [0.5, 0.5] @ [[1, 0], [0.25, 0.75]]
        baseline = ar.uniform_attention_baseline(torch.ones((1, 2)), 1, 1)
        torch.testing.assert_close(baseline, torch.tensor([[0.625, 0.375]]))

    def test_hand_computed_two_probe_blocks_read_two_decoder_layers(self):
        first = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
        second = torch.tensor([[[1.0, 0.0], [0.4, 0.6]]])
        cross_first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        cross_second = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
        scores = ar.linear_prob_attention_rollout(
            [first, second], [cross_first, cross_second], torch.ones((1, 2))
        )
        # R_1 = [[1, 0], [0.25, 0.75]], R_2 = [[1, 0], [0.2, 0.8]] @ R_1 = [[1, 0], [0.4, 0.6]]
        # The first probe block reads R_1 at the last token: [0.25, 0.75], weight 0.25 / 0.75
        # The second reads [0.5, 0.5] @ R_2 = [0.7, 0.3], weight 0.5 / 0.75
        torch.testing.assert_close(scores, torch.tensor([[0.55, 0.45]]))

    def test_model_rollout(self):
        batch = padded_batch()
        real = batch["attention_mask"].bool()
        for backbone, linear_prob_n_layer in itertools.product(("gpt2", "qwen2"), (None, 2)):
            with self.subTest(backbone=backbone, linear_prob_n_layer=linear_prob_n_layer):
                model = random_model(backbone, linear_prob_n_layer)
                with torch.no_grad():
                    expected = model(**batch, output_attentions=True)
                    with ar.AttentionCapture(model) as capture:
                        captured = model(**batch)
                # Capturing changes nothing, and what is captured is what the model returns
                torch.testing.assert_close(captured.logits, expected.logits)
                self.assertEqual(len(capture.cross_attentions), linear_prob_n_layer or 3)
                for attention, returned in zip(capture.self_attentions, expected.attentions):
                    torch.testing.assert_close(attention[real], returned.mean(dim=1)[real])
                for block in model.cehrgpt.h:
                    self.assertFalse(block.attn._forward_hooks or block.attn._forward_pre_hooks)

                scores = ar.compute_token_importance(model, batch)
                self.assertTrue((scores >= 0).all())
                torch.testing.assert_close(scores.sum(dim=-1), torch.ones(2))
                self.assertEqual(scores[~real].abs().max(), 0)
                # A padded patient has the scores it has alone
                for row, (start, end) in enumerate([(0, 5), (2, 8)]):
                    alone = {key: value[row : row + 1, start:end] for key, value in batch.items()}
                    torch.testing.assert_close(
                        ar.compute_token_importance(model, alone)[0],
                        scores[row, start:end],
                        atol=1e-5,
                        rtol=1e-4,
                    )
                local = ar.compute_token_importance(model, batch, target="local")
                self.assertFalse(torch.allclose(local, scores))


if __name__ == "__main__":
    unittest.main()
