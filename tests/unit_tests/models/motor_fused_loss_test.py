import functools
import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models import hf_cehrgpt
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2LMHeadModel,
    compile_motor_loss_terms,
    get_motor_loss_terms,
    motor_loss_terms,
)

VOCAB_SIZE = 64
MOTOR_VOCAB = 40
NUM_PIECES = 4
TIME_BINS = [0.0, 1.0, 7.0, 30.0, float("inf")]


def build_model(fused: bool) -> CEHRGPT2LMHeadModel:
    torch.manual_seed(0)
    config = CEHRGPTConfig(
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=32,
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
        motor_tte_vocab_size=MOTOR_VOCAB,
        motor_num_time_pieces=NUM_PIECES,
        motor_time_bins=list(TIME_BINS),
        motor_fused_loss=fused,
        linear_prob_n_layer=1,
        linear_prob_token_id=8,
        ve_token_id=5,
        vs_token_id=4,
        att_token_ids=[6, 7],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return CEHRGPT2LMHeadModel(config)


def make_motor_inputs(batch=2, length=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    indicators = torch.zeros(batch, length, dtype=torch.bool)
    indicators[:, 2:10:2] = True  # 4 prediction rows per sequence
    rows = int(indicators.sum())
    censor_times = torch.rand(batch, length, generator=generator) * 100
    # Times chosen to hit every bin, the bin boundaries and the open last bin
    special = torch.tensor([0.0, 1.0, 7.0, 30.0, 0.5, 6.999, 29.0, 500.0])
    num_entries = rows * 6
    row_indices = torch.randint(0, rows, (num_entries,), generator=generator)
    col_indices = torch.randint(0, MOTOR_VOCAB, (num_entries,), generator=generator)
    values = special[torch.randint(0, len(special), (num_entries,), generator=generator)]
    # (row, col) pairs must be unique for the scatter to be well defined
    unique = torch.unique(row_indices * MOTOR_VOCAB + col_indices)
    row_indices, col_indices = unique // MOTOR_VOCAB, unique % MOTOR_VOCAB
    values = values[: len(unique)]
    return dict(
        motor_row_indices=row_indices.long(),
        motor_col_indices=col_indices.long(),
        motor_values=values.float(),
        motor_censor_times=censor_times,
        motor_tte_task_indicators=indicators,
        motor_end_index=torch.tensor([len(unique)]),
    ), rows


def run_loss(model, hidden_states, motor_inputs):
    hidden_states = hidden_states.clone().requires_grad_(True)
    model.zero_grad()
    loss = model.motor_nll_loss(hidden_states=hidden_states, **motor_inputs)
    loss.backward()
    head_grads = [p.grad.clone() for p in model.motor_tte.parameters()]
    return loss.detach(), hidden_states.grad, head_grads


class TestFusedMotorLoss(unittest.TestCase):
    def _compare(self, seed):
        reference_model = build_model(fused=False)
        fused_model = build_model(fused=True)
        fused_model.load_state_dict(reference_model.state_dict())
        motor_inputs, _ = make_motor_inputs(seed=seed)
        hidden = torch.randn(2, 12, 32, generator=torch.Generator().manual_seed(seed))

        ref_loss, ref_hidden_grad, ref_head_grads = run_loss(reference_model, hidden, motor_inputs)
        loss, hidden_grad, head_grads = run_loss(fused_model, hidden, motor_inputs)

        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(hidden_grad, ref_hidden_grad, rtol=1e-4, atol=1e-6)
        for grad, ref_grad in zip(head_grads, ref_head_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-6)

    def test_loss_and_gradients_match_the_unfused_implementation(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                self._compare(seed)

    def test_flag_defaults_to_off(self):
        self.assertFalse(CEHRGPTConfig().motor_fused_loss)
        self.assertFalse(CEHRGPTConfig.from_dict(CEHRGPTConfig().to_dict()).motor_fused_loss)
        self.assertTrue(CEHRGPTConfig(motor_fused_loss=True).motor_fused_loss)

    def test_rows_without_events_are_all_censored(self):
        model_fused, model_reference = build_model(True), build_model(False)
        model_fused.load_state_dict(model_reference.state_dict())
        inputs, rows = make_motor_inputs()
        inputs["motor_row_indices"] = inputs["motor_row_indices"][:0]
        inputs["motor_col_indices"] = inputs["motor_col_indices"][:0]
        inputs["motor_values"] = inputs["motor_values"][:0]
        inputs["motor_end_index"] = torch.tensor([0])
        hidden = torch.randn(2, 12, 32)
        torch.testing.assert_close(
            run_loss(model_fused, hidden, inputs)[0],
            run_loss(model_reference, hidden, inputs)[0],
            rtol=1e-5,
            atol=1e-6,
        )


class TestMotorLossTermsCompile(unittest.TestCase):
    """The function must trace without graph breaks and for any number of rows."""

    def test_traces_with_dynamic_row_counts(self):
        compiled = torch.compile(
            motor_loss_terms, dynamic=True, backend="aot_eager", fullgraph=True
        )
        start = torch.tensor(TIME_BINS[:-1])
        end = torch.tensor(TIME_BINS[1:])
        for rows in (3, 7, 50):
            generator = torch.Generator().manual_seed(rows)
            logits = torch.randn(rows, NUM_PIECES, MOTOR_VOCAB, generator=generator, requires_grad=True)
            times = torch.rand(rows, MOTOR_VOCAB, generator=generator) * 60
            events = torch.rand(rows, MOTOR_VOCAB, generator=generator) < 0.1
            expected = motor_loss_terms(logits, times, events, start, end)
            actual = compiled(logits, times, events, start, end)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-5)
            (actual[0] + actual[1]).backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bfloat16_logits_are_accumulated_in_float32(self):
        start = torch.tensor(TIME_BINS[:-1])
        end = torch.tensor(TIME_BINS[1:])
        logits = torch.randn(4, NUM_PIECES, MOTOR_VOCAB)
        times = torch.rand(4, MOTOR_VOCAB) * 60
        events = torch.rand(4, MOTOR_VOCAB) < 0.2
        hazard, event_logits = motor_loss_terms(logits.bfloat16(), times, events, start, end)
        self.assertEqual(hazard.dtype, torch.float32)
        self.assertEqual(event_logits.dtype, torch.float32)
        expected = motor_loss_terms(logits.bfloat16().float(), times, events, start, end)
        torch.testing.assert_close(hazard, expected[0])


class TestCompileMotorLossTerms(unittest.TestCase):
    def setUp(self):
        hf_cehrgpt._motor_loss_terms_in_use = None

    tearDown = setUp

    def test_a_working_compile_is_verified_and_logged(self):
        compile_fn = functools.partial(torch.compile, backend="aot_eager")
        with self.assertLogs(hf_cehrgpt.MOTOR_LOG.name, level="INFO") as logs:
            result = compile_motor_loss_terms(torch.device("cpu"), compile_fn=compile_fn)
        self.assertIsNot(result, motor_loss_terms)
        self.assertTrue(any("compiled with torch.compile" in line for line in logs.output))

    def test_a_forward_failure_falls_back_with_a_warning(self):
        def broken_compile(function, **kwargs):
            def call(*args):
                raise RuntimeError("no triton")
            return call

        with self.assertLogs(hf_cehrgpt.MOTOR_LOG.name, level="WARNING") as logs:
            result = compile_motor_loss_terms(torch.device("cpu"), compile_fn=broken_compile)
        self.assertIs(result, motor_loss_terms)
        self.assertTrue(any("Could not compile" in line and "no triton" in line for line in logs.output))

    def test_a_backward_failure_is_caught_too(self):
        class FailsOnBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 1

            @staticmethod
            def backward(ctx, grad):
                raise RuntimeError("backward could not be compiled")

        def compile_with_bad_backward(function, **kwargs):
            def call(logits, *args):
                hazard, event_logits = function(FailsOnBackward.apply(logits), *args)
                return hazard, event_logits
            return call

        with self.assertLogs(hf_cehrgpt.MOTOR_LOG.name, level="WARNING") as logs:
            result = compile_motor_loss_terms(
                torch.device("cpu"), compile_fn=compile_with_bad_backward
            )
        self.assertIs(result, motor_loss_terms)
        self.assertTrue(any("backward could not be compiled" in line for line in logs.output))

    def test_cpu_uses_the_eager_function_and_resolves_once(self):
        with self.assertLogs(hf_cehrgpt.MOTOR_LOG.name, level="INFO") as logs:
            first = get_motor_loss_terms(torch.device("cpu"))
            second = get_motor_loss_terms(torch.device("cpu"))
        self.assertIs(first, motor_loss_terms)
        self.assertIs(second, first)
        self.assertEqual(sum("eager version" in line for line in logs.output), 1)


if __name__ == "__main__":
    unittest.main()
