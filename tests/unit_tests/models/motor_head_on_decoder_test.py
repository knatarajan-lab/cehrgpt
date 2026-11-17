import tempfile
import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

VOCAB_SIZE = 64
MOTOR_VOCAB = 8
N_EMBD = 32
TIME_BINS = [0.0, 1.0, 7.0, 365.0, float("inf")]
VE_TOKEN_ID = 5


def build_model(motor_head_on_decoder: bool) -> CEHRGPT2LMHeadModel:
    torch.manual_seed(0)
    config = CEHRGPTConfig(
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=N_EMBD,
        n_layer=2,
        n_head=4,
        n_inner=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=True,
        motor_head_on_decoder=motor_head_on_decoder,
        motor_tte_vocab_size=MOTOR_VOCAB,
        motor_num_time_pieces=len(TIME_BINS) - 1,
        motor_time_bins=list(TIME_BINS),
        linear_prob_n_layer=1,
        linear_prob_token_id=8,
        ve_token_id=VE_TOKEN_ID,
        vs_token_id=4,
        att_token_ids=[6, 7],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return CEHRGPT2LMHeadModel(config)


def make_batch():
    """One patient with two visits; MOTOR predicts at each [VE]."""
    input_ids = torch.tensor([[2, 3, 4, 10, 11, VE_TOKEN_ID, 6, 4, 12, VE_TOKEN_ID]])
    task_indicators = input_ids == VE_TOKEN_ID
    censor_times = torch.zeros(input_ids.shape, dtype=torch.float32)
    censor_times[task_indicators] = torch.tensor([400.0, 30.0])
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
        # One observed event per prediction point, every other task censored
        motor_row_indices=torch.tensor([0, 1]),
        motor_col_indices=torch.tensor([3, 0]),
        motor_values=torch.tensor([100.0, 2.0]),
        motor_censor_times=censor_times,
        motor_tte_task_indicators=task_indicators,
        motor_end_index=torch.tensor([2]),
    )


class MotorHeadOnDecoderTest(unittest.TestCase):
    def test_no_cross_attention_stack_is_built(self):
        model = build_model(motor_head_on_decoder=True)
        self.assertIsNone(model.linear_prob)
        self.assertIsNotNone(model.motor_tte)
        self.assertFalse(any(n.startswith("linear_prob.") for n in model.state_dict()))

        baseline = build_model(motor_head_on_decoder=False)
        self.assertIsNotNone(baseline.linear_prob)

    def test_motor_loss_is_the_head_applied_to_the_final_hidden_state(self):
        model = build_model(motor_head_on_decoder=True).eval()
        batch = make_batch()
        output = model(**batch, output_hidden_states=True)

        self.assertIsNone(output.linear_prob_hidden_states)
        self.assertTrue(torch.isfinite(output.motor_tte_loss))
        motor_inputs = {k: v for k, v in batch.items() if k.startswith("motor_")}
        expected = model.motor_nll_loss(
            hidden_states=output.hidden_states[-1], **motor_inputs
        )
        torch.testing.assert_close(output.motor_tte_loss, expected)
        torch.testing.assert_close(
            output.loss,
            output.token_loss * model.config.next_token_prediction_loss_weight
            + output.motor_tte_loss * model.config.motor_time_to_event_weight,
        )

    def test_hidden_states_are_not_kept_for_the_motor_loss(self):
        model = build_model(motor_head_on_decoder=True).eval()
        output = model(**make_batch(), output_hidden_states=False)
        self.assertIsNone(output.hidden_states)
        self.assertIsNotNone(output.motor_tte_loss)

    def test_both_heads_send_gradients_into_the_decoder(self):
        model = build_model(motor_head_on_decoder=True)
        decoder_weight = model.cehrgpt.h[-1].mlp.c_proj.weight

        output = model(**make_batch())
        (token_grad,) = torch.autograd.grad(
            output.token_loss, decoder_weight, retain_graph=True
        )
        (motor_grad,) = torch.autograd.grad(output.motor_tte_loss, decoder_weight)
        self.assertGreater(token_grad.abs().sum().item(), 0)
        self.assertGreater(motor_grad.abs().sum().item(), 0)

    def test_generation_and_attention_bias_update_work_without_the_stack(self):
        model = build_model(motor_head_on_decoder=True).eval()
        model.update_attn_bias(128)
        inputs = torch.tensor([[2, 3, 4]])
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            output_motor_features=False,
        )
        self.assertEqual(output.logits.shape, (1, 3, VOCAB_SIZE))

    def test_checkpoint_round_trip_keeps_the_variant(self):
        model = build_model(motor_head_on_decoder=True).eval()
        batch = make_batch()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = CEHRGPT2LMHeadModel.from_pretrained(tmp).eval()
        self.assertTrue(reloaded.config.motor_head_on_decoder)
        self.assertIsNone(reloaded.linear_prob)
        torch.testing.assert_close(
            reloaded(**batch).motor_tte_loss, model(**batch).motor_tte_loss
        )


if __name__ == "__main__":
    unittest.main()
