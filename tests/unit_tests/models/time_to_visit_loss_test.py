import unittest
from unittest import mock

import torch

from cehrgpt.models import hf_cehrgpt
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel, VisitTimeToEventHead

VOCAB_SIZE = 64
SEQ_LEN = 12


def build_config(backbone="gpt2", **overrides) -> CEHRGPTConfig:
    kwargs = dict(
        backbone=backbone,
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=128,
        activation_function="silu" if backbone == "qwen2" else "gelu_new",
        decoder_mlp="LlamaMLP" if backbone == "qwen2" else "GPT2MLP",
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=True,
        include_motor_time_to_event=False,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    kwargs.update(overrides)
    return CEHRGPTConfig(**kwargs)


def make_batch(time_to_visits=None, batch_size=2):
    input_ids = torch.randint(2, VOCAB_SIZE, (batch_size, SEQ_LEN))
    if time_to_visits is None:
        # Mix of valid gaps (incl. 0 and a very long gap) and masked (-1) positions.
        time_to_visits = torch.tensor(
            [
                [-1, 0, 3, -1, 30, -1, 1, 365, -1, 2, -1, -1],
                [-1, -1, 7, -1, 0, 1000, -1, -1, 5, -1, 14, -1],
            ],
            dtype=torch.float32,
        )[:batch_size]
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        ages=torch.randint(20, 80, (batch_size, SEQ_LEN)),
        labels=input_ids.clone(),
        time_to_visits=time_to_visits,
    )


def all_grads_finite(model) -> bool:
    return all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


class TestVisitTimeToEventHeadBounds(unittest.TestCase):
    def test_params_bounded_for_extreme_inputs(self):
        head = VisitTimeToEventHead(16)
        for scale in (0.0, 1.0, 1e4, 1e8):
            for sign in (1.0, -1.0):
                x = torch.randn(2, 5, 16) * scale * sign
                lam, k = head(x)
                for p in (lam, k):
                    self.assertEqual(p.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(p).all())
                    self.assertGreaterEqual(p.min().item(), head.PARAM_MIN)
                    self.assertLessEqual(p.max().item(), head.PARAM_MAX)

    def test_bf16_inputs_return_float32_params(self):
        head = VisitTimeToEventHead(16).to(torch.bfloat16)
        lam, k = head(torch.randn(2, 5, 16, dtype=torch.bfloat16) * 1e4)
        self.assertEqual(lam.dtype, torch.float32)
        self.assertEqual(k.dtype, torch.float32)
        self.assertTrue(torch.isfinite(lam).all() and torch.isfinite(k).all())

    def test_softplus_underflow_does_not_reach_zero(self):
        head = VisitTimeToEventHead(16)
        # Drive both heads' outputs to a hugely negative pre-activation.
        with torch.no_grad():
            for seq in (head.linear1, head.linear2):
                seq[-1].weight.zero_()
                seq[-1].bias.fill_(-1e4)
        lam, k = head(torch.randn(1, 3, 16))
        self.assertTrue((lam > 0).all() and (k > 0).all())


class TestTimeToVisitLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _run(self, config=None, batch=None):
        model = CEHRGPT2LMHeadModel(config or build_config()).train()
        out = model(**(batch or make_batch()))
        return model, out

    def test_loss_and_grads_finite_with_masked_positions(self):
        for backbone in ("gpt2", "qwen2"):
            with self.subTest(backbone=backbone):
                model, out = self._run(build_config(backbone))
                self.assertTrue(torch.isfinite(out.loss))
                self.assertTrue(torch.isfinite(out.time_to_visit_loss))
                self.assertGreater(out.time_to_visit_loss.item(), 0.0)
                out.loss.backward()
                self.assertTrue(all_grads_finite(model))
                # The tte head must actually receive gradient from valid positions.
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.tte_head.parameters()
                    )
                )

    def test_all_positions_masked_gives_zero_loss(self):
        batch = make_batch(torch.full((2, SEQ_LEN), -1.0))
        model, out = self._run(batch=batch)
        self.assertEqual(out.time_to_visit_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(all_grads_finite(model))

    def test_masked_positions_do_not_affect_loss(self):
        """Changing what sits under the -1 mask (params are replaced) must be a no-op."""
        batch = make_batch()
        model = CEHRGPT2LMHeadModel(build_config()).eval()
        with torch.no_grad():
            base = model(**batch).time_to_visit_loss.item()
            # Large garbage in masked slots: still masked as long as it is negative.
            garbage = batch["time_to_visits"].clone()
            garbage[garbage < 0] = -123456.0
            batch["time_to_visits"] = garbage
            other = model(**batch).time_to_visit_loss.item()
        self.assertAlmostEqual(base, other, places=5)

    def test_extreme_time_to_visit_targets(self):
        for value in (0.0, 1e-9, 1e6, 1e12):
            with self.subTest(value=value):
                batch = make_batch(torch.full((2, SEQ_LEN), value))
                model, out = self._run(batch=batch)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                self.assertTrue(all_grads_finite(model))

    def test_huge_tte_head_weights_stay_finite(self):
        """Mimics a post-divergence head: pre-activations far outside softplus's range."""
        model = CEHRGPT2LMHeadModel(build_config()).train()
        with torch.no_grad():
            for p in model.tte_head.parameters():
                p.mul_(1e6)
        out = model(**make_batch())
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(all_grads_finite(model))

    def test_bf16_model_loss_finite(self):
        model = CEHRGPT2LMHeadModel(build_config()).to(torch.bfloat16).train()
        out = model(**make_batch())
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(all_grads_finite(model))


class TestNonFiniteLossGuard(unittest.TestCase):
    """The guard must turn a non-finite Gamma loss into 0.0 without poisoning the step."""

    def setUp(self):
        torch.manual_seed(0)

    def _poisoned_forward(self, bad_value):
        model = CEHRGPT2LMHeadModel(build_config()).train()
        real_forward = model.tte_head.forward

        def poisoned(x):
            lam, k = real_forward(x)
            return torch.full_like(lam, bad_value), torch.full_like(k, bad_value)

        with mock.patch.object(model.tte_head, "forward", side_effect=poisoned):
            out = model(**make_batch())
        return model, out

    def test_nonfinite_gamma_params_are_replaced_with_zero(self):
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad_value=bad_value):
                model, out = self._poisoned_forward(bad_value)
                self.assertEqual(out.time_to_visit_loss.item(), 0.0)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                self.assertTrue(all_grads_finite(model))

    def test_guard_logs_a_warning(self):
        with self.assertLogs(hf_cehrgpt.logger.name, level="WARNING") as logs:
            self._poisoned_forward(float("nan"))
        self.assertTrue(any("time_to_visit_loss" in m for m in logs.output))

    def test_guard_keeps_tte_head_in_graph(self):
        """Every tte_head param must get a (zero) grad so DDP does not flag it unused."""
        model, out = self._poisoned_forward(float("nan"))
        out.loss.backward()
        for name, p in model.tte_head.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue((p.grad == 0).all(), name)

    def test_finite_loss_is_untouched_by_the_guard(self):
        model = CEHRGPT2LMHeadModel(build_config()).train()
        with self.assertNoLogs(hf_cehrgpt.logger.name, level="WARNING"):
            out = model(**make_batch())
        self.assertGreater(out.time_to_visit_loss.item(), 0.0)


class TestTotalLossGuard(unittest.TestCase):
    """A NaN/inf from anywhere must not produce a NaN loss, NaN grads, or NaN weights."""

    def setUp(self):
        torch.manual_seed(0)

    @staticmethod
    def _constant_logits(model, bad_value):
        real_forward = model.lm_head.forward
        return mock.patch.object(
            model.lm_head,
            "forward",
            side_effect=lambda x: torch.full_like(real_forward(x), bad_value),
        )

    def _assert_safe_step(self, model, out):
        self.assertEqual(out.loss.item(), 0.0)
        out.loss.backward()
        self.assertTrue(all_grads_finite(model))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        torch.optim.SGD(model.parameters(), lr=1.0).step()
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(before[n], p.detach()), n)

    def test_nonfinite_lm_logits_give_zero_loss(self):
        for bad_value in (float("nan"), float("inf")):
            with self.subTest(bad_value=bad_value):
                model = CEHRGPT2LMHeadModel(build_config()).train()
                with self._constant_logits(model, bad_value):
                    out = model(**make_batch())
                self._assert_safe_step(model, out)

    def test_nan_weights_do_not_propagate(self):
        """A diverged model (NaN embeddings) yields loss 0.0 and finite zero gradients."""
        model = CEHRGPT2LMHeadModel(build_config()).train()
        with torch.no_grad():
            model.cehrgpt.wte.weight[2:5] = float("nan")
        batch = make_batch()
        batch["input_ids"][:] = 3  # every token hits a NaN embedding row
        batch["labels"] = batch["input_ids"].clone()
        out = model(**batch)
        self.assertEqual(out.loss.item(), 0.0)
        out.loss.backward()
        self.assertTrue(all_grads_finite(model))

    def test_guard_names_the_loss_and_warns_once(self):
        model = CEHRGPT2LMHeadModel(build_config()).train()
        with self.assertLogs(hf_cehrgpt.logger.name, level="WARNING") as logs:
            with self._constant_logits(model, float("nan")):
                model(**make_batch())
        self.assertEqual(sum("Non-finite loss" in m for m in logs.output), 1)

    def test_recovers_on_the_next_good_step(self):
        model = CEHRGPT2LMHeadModel(build_config()).train()
        with self._constant_logits(model, float("nan")):
            bad = model(**make_batch())
        self.assertEqual(bad.loss.item(), 0.0)
        good = model(**make_batch())
        self.assertGreater(good.loss.item(), 0.0)
        good.loss.backward()
        self.assertTrue(all_grads_finite(model))


if __name__ == "__main__":
    unittest.main()
