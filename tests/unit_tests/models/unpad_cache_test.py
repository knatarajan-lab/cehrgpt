import inspect
import unittest
from unittest import mock

import torch

from cehrgpt.models import gpt2
from cehrgpt.models.gpt2 import (
    _compute_unpad_data,
    _get_unpad_data,
    reuse_unpad_data,
    reuse_unpad_data_during_forward,
)
from cehrgpt.models.hf_cehrgpt import CEHRGPT2Model, LinearProbModule


def packed_mask():
    # one row holding three packed samples (lengths 3, 2, 4) followed by padding
    return torch.tensor([[1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0]])


def padded_mask():
    return torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])


def assert_same(testcase, a, b):
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    testcase.assertEqual(a[2], b[2])


class TestUnpadCache(unittest.TestCase):
    def test_results_are_identical_with_and_without_the_cache(self):
        for mask in (packed_mask(), padded_mask()):
            plain = _compute_unpad_data(mask)
            with reuse_unpad_data():
                first = _get_unpad_data(mask)
                again = _get_unpad_data(mask)
            assert_same(self, first, plain)
            assert_same(self, again, plain)

    def test_computed_once_per_mask_inside_the_block(self):
        mask = packed_mask()
        with mock.patch.object(gpt2, "_compute_unpad_data", wraps=_compute_unpad_data) as spy:
            with reuse_unpad_data():
                for _ in range(21):  # 14 transformer layers + 7 linear-probe layers
                    _get_unpad_data(mask)
            self.assertEqual(spy.call_count, 1)

    def test_nothing_is_cached_outside_the_block(self):
        mask = packed_mask()
        with mock.patch.object(gpt2, "_compute_unpad_data", wraps=_compute_unpad_data) as spy:
            for _ in range(3):
                _get_unpad_data(mask)
            self.assertEqual(spy.call_count, 3)

    def test_each_block_starts_empty(self):
        mask = packed_mask()
        with mock.patch.object(gpt2, "_compute_unpad_data", wraps=_compute_unpad_data) as spy:
            for _ in range(2):
                with reuse_unpad_data():
                    _get_unpad_data(mask)
                    _get_unpad_data(mask)
            self.assertEqual(spy.call_count, 2)

    def test_a_different_mask_is_computed_separately(self):
        with reuse_unpad_data():
            a = _get_unpad_data(packed_mask())
            b = _get_unpad_data(padded_mask())
        self.assertEqual(a[2], 4)
        self.assertEqual(b[2], 5)

    def test_modifying_the_mask_in_place_invalidates_the_entry(self):
        mask = packed_mask()
        with reuse_unpad_data():
            before = _get_unpad_data(mask)
            mask[0, 3] = 1  # merges the first two samples
            after = _get_unpad_data(mask)
        self.assertNotEqual(before[2], after[2])
        assert_same(self, after, _compute_unpad_data(mask))

    def test_a_new_tensor_never_gets_a_stale_entry(self):
        with reuse_unpad_data():
            first = _get_unpad_data(packed_mask())
            second = _get_unpad_data(packed_mask())  # equal values, a different tensor
        assert_same(self, first, second)

    def test_nested_blocks_restore_the_outer_cache(self):
        mask = packed_mask()
        with mock.patch.object(gpt2, "_compute_unpad_data", wraps=_compute_unpad_data) as spy:
            with reuse_unpad_data():
                _get_unpad_data(mask)
                with reuse_unpad_data():
                    _get_unpad_data(mask)
                _get_unpad_data(mask)
            self.assertEqual(spy.call_count, 2)

    def test_the_cache_is_per_thread(self):
        import threading

        mask = packed_mask()
        results = {}

        def other_thread():
            with mock.patch.object(gpt2, "_compute_unpad_data", wraps=_compute_unpad_data) as spy:
                _get_unpad_data(mask)  # no block active in this thread
                results["calls"] = spy.call_count

        with reuse_unpad_data():
            _get_unpad_data(mask)
            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join()
        self.assertEqual(results["calls"], 1)


class TestUnpadReuseIsVisibleInTheLog(unittest.TestCase):
    def setUp(self):
        gpt2._logged_unpad_reuse = False

    def test_the_first_reuse_is_logged_once(self):
        mask = packed_mask()
        with self.assertLogs(gpt2._UNPAD_LOG.name, level="INFO") as logs:
            with reuse_unpad_data():
                for _ in range(5):
                    _get_unpad_data(mask)
            with reuse_unpad_data():
                for _ in range(5):
                    _get_unpad_data(mask)
        self.assertEqual(sum("Sharing the flash-attention unpad data" in l for l in logs.output), 1)

    def test_no_reuse_no_message(self):
        with self.assertNoLogs(gpt2._UNPAD_LOG.name, level="INFO"):
            for _ in range(3):
                _get_unpad_data(packed_mask())


class TestDecorator(unittest.TestCase):
    def test_forward_runs_inside_the_block_and_keeps_its_signature(self):
        class Layer(torch.nn.Module):
            @reuse_unpad_data_during_forward
            def forward(self, mask, scale: int = 1):
                return [_get_unpad_data(mask) for _ in range(5)]

        mask = packed_mask()
        with mock.patch.object(gpt2, "_compute_unpad_data", wraps=_compute_unpad_data) as spy:
            Layer()(mask)
            self.assertEqual(spy.call_count, 1)
        self.assertEqual(
            list(inspect.signature(Layer.forward).parameters), ["self", "mask", "scale"]
        )

    def test_the_attention_stacks_use_it(self):
        for module in (CEHRGPT2Model, LinearProbModule):
            self.assertTrue(
                hasattr(module.forward, "__wrapped__"),
                f"{module.__name__}.forward is not wrapped",
            )
        # the signature, which the Trainer inspects, is unchanged
        self.assertIn("attention_mask", inspect.signature(CEHRGPT2Model.forward).parameters)


if __name__ == "__main__":
    unittest.main()
