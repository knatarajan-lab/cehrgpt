import math
import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2LMHeadModel,
    bounded_exp2,
    replace_nonfinite_loss,
)

MOTOR_VOCAB = 8
N_EMBD = 32
TIME_BINS = [0.0, 1.0, 7.0, 365.0, float("inf")]


def build_model(**overrides) -> CEHRGPT2LMHeadModel:
    kwargs = dict(
        vocab_size=64,
        n_positions=64,
        n_embd=N_EMBD,
        n_layer=1,
        n_head=4,
        n_inner=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=True,
        motor_tte_vocab_size=MOTOR_VOCAB,
        motor_num_time_pieces=len(TIME_BINS) - 1,
        motor_time_bins=list(TIME_BINS),
        linear_prob_n_layer=1,
        linear_prob_token_id=8,
        ve_token_id=5,
        vs_token_id=4,
        att_token_ids=[6, 7],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return CEHRGPT2LMHeadModel(CEHRGPTConfig(**kwargs))


def make_motor_inputs(censor_times=(30.0, 400.0, 5000.0), seq_len=6):
    """Three prediction points, each with one observed event and the rest censored."""
    n_predictions = len(censor_times)
    torch.manual_seed(1)
    task_indicators = torch.zeros(1, seq_len, dtype=torch.bool)
    task_indicators[0, :n_predictions] = True
    padded_censor_times = torch.zeros(1, seq_len)
    padded_censor_times[0, :n_predictions] = torch.tensor(censor_times)
    return dict(
        hidden_states=torch.randn(1, seq_len, N_EMBD),
        motor_row_indices=torch.arange(n_predictions),
        motor_col_indices=torch.tensor([3, 0, 5])[:n_predictions],
        motor_values=torch.tensor([2.0, 100.0, 3000.0])[:n_predictions],
        motor_censor_times=padded_censor_times,
        motor_tte_task_indicators=task_indicators,
        motor_end_index=torch.tensor([n_predictions]),
    )


def max_log2_cumulative_hazard(model, inputs) -> float:
    """The largest log2(hazard * time in bin) the survival term sees for these inputs."""
    task_indicators = inputs["motor_tte_task_indicators"]
    logits = model.motor_tte(inputs["hidden_states"][task_indicators])
    censor_times = inputs["motor_censor_times"][task_indicators][:, None]
    bins = torch.tensor(TIME_BINS)
    time_in_bin = torch.minimum(
        (censor_times - bins[:-1]).clamp(min=0), bins[1:] - bins[:-1]
    )
    return (logits.max(dim=-1).values + torch.log2(time_in_bin)).max().item()


class BoundedExp2Test(unittest.TestCase):
    def test_matches_exp2_below_cap(self):
        x = torch.linspace(-20, 10, 50)
        torch.testing.assert_close(bounded_exp2(x, 10.0), torch.exp2(x))

    def test_none_disables_cap(self):
        x = torch.tensor([30.0])
        torch.testing.assert_close(bounded_exp2(x, None), torch.exp2(x))

    def test_linear_with_bounded_gradient_above_cap(self):
        x = torch.tensor([10.0, 40.0, 1000.0], requires_grad=True)
        y = bounded_exp2(x, 10.0)
        expected = 1024 * (1 + math.log(2) * (x.detach() - 10))
        torch.testing.assert_close(y.detach(), expected)
        y.sum().backward()
        slope = 1024 * math.log(2)
        torch.testing.assert_close(x.grad, torch.full_like(x, slope))

    def test_negative_infinity_maps_to_zero(self):
        # Bins a patient never reaches carry log2(time) = -inf
        x = torch.tensor([-float("inf"), 0.0], requires_grad=True)
        y = bounded_exp2(x, 10.0)
        torch.testing.assert_close(y.detach(), torch.tensor([0.0, 1.0]))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class MotorNllLossGuardrailTest(unittest.TestCase):
    def test_unchanged_below_cap(self):
        # A freshly initialized head (hazard ~1/day) times thousands of days already
        # exceeds 2**10, so use a cap the unguarded loss stays below
        inputs = make_motor_inputs()
        model = build_model(motor_max_log2_cumulative_hazard=16.0)
        self.assertLess(max_log2_cumulative_hazard(model, inputs), 16.0)
        unguarded = build_model(motor_max_log2_cumulative_hazard=None)
        torch.testing.assert_close(
            model.motor_nll_loss(**inputs), unguarded.motor_nll_loss(**inputs)
        )

    def test_outlier_exposure_does_not_explode_loss_or_gradient(self):
        # An implausibly long follow-up in the open-ended last bin
        inputs = make_motor_inputs(censor_times=(30.0, 400.0, 1e9))

        unguarded = build_model(motor_max_log2_cumulative_hazard=None)
        self.assertGreater(unguarded.motor_nll_loss(**inputs).item(), 1e7)

        model = build_model()
        loss = model.motor_nll_loss(**inputs)
        self.assertLess(loss.item(), 1e5)
        loss.backward()
        grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm()
                    for p in model.motor_tte.parameters()
                    if p.grad is not None
                ]
            )
        )
        self.assertTrue(torch.isfinite(grad_norm))
        self.assertLess(grad_norm.item(), 1e5)

    def test_non_finite_loss_is_replaced_with_zero(self):
        inputs = make_motor_inputs()
        inputs["hidden_states"][0, 0, 0] = float("nan")
        model = build_model()
        loss = replace_nonfinite_loss(
            model.motor_nll_loss(**inputs),
            "motor_tte_loss",
            model.motor_tte.parameters(),
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for p in model.motor_tte.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_config_round_trip(self):
        config = build_model(motor_max_log2_cumulative_hazard=8.0).config
        restored = CEHRGPTConfig.from_dict(config.to_dict())
        self.assertEqual(restored.motor_max_log2_cumulative_hazard, 8.0)


if __name__ == "__main__":
    unittest.main()
