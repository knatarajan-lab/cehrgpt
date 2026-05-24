import math
import unittest

import torch

from cehrgpt.models.hf_cehrgpt import build_local_attention_mask

VS = 1  # synthetic [VS] token id
NEG_INF = torch.finfo(torch.float32).min


def _allowed(mask: torch.Tensor) -> torch.Tensor:
    """Return bool tensor: True where the mask allows attention (value == 0)."""
    return mask == 0.0


def _blocked(mask: torch.Tensor) -> torch.Tensor:
    """Return bool tensor: True where the mask blocks attention (value == NEG_INF)."""
    return mask == NEG_INF


class TestBuildLocalAttentionMask(unittest.TestCase):

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    def test_output_shape(self):
        input_ids = torch.tensor([[0, VS, 2, 3, VS, 5]])  # (1, 6)
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=1)
        self.assertEqual(mask.shape, (1, 1, 6, 6))

    def test_output_shape_batch(self):
        input_ids = torch.zeros(4, 10, dtype=torch.long)
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=2)
        self.assertEqual(mask.shape, (4, 1, 10, 10))

    # ------------------------------------------------------------------
    # Causal constraint: upper triangle must always be blocked
    # ------------------------------------------------------------------
    def test_causal_upper_triangle_blocked(self):
        input_ids = torch.tensor([[0, VS, 2, VS, 4, 5]])
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=10)
        m = mask[0, 0]  # (L, L)
        L = m.shape[0]
        for i in range(L):
            for j in range(i + 1, L):
                self.assertTrue(
                    _blocked(m[i, j]),
                    f"Expected position ({i},{j}) to be blocked (future token), got {m[i,j]}",
                )

    # ------------------------------------------------------------------
    # Demographics (visit_idx == 0, i.e. before the first VS) always visible
    # ------------------------------------------------------------------
    def test_demographics_always_visible(self):
        # Positions 0,1 are demographics (before first VS at position 2)
        # position 2 = VS (visit 1), 3,4 = visit 1 tokens
        # position 5 = VS (visit 2), 6,7 = visit 2 tokens
        input_ids = torch.tensor([[0, 0, VS, 3, 4, VS, 6, 7]])
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=0)
        m = mask[0, 0]  # (8, 8)
        # Every token should be able to attend to positions 0 and 1 (demographics)
        for i in range(8):
            for j in [0, 1]:
                if j <= i:  # causal constraint
                    self.assertTrue(
                        _allowed(m[i, j]),
                        f"Demographics position {j} should be visible to token {i}, got {m[i,j]}",
                    )

    # ------------------------------------------------------------------
    # n_prev_visits=0: only current visit (+ demographics)
    # ------------------------------------------------------------------
    def test_n_prev_visits_0_only_current_visit(self):
        # Layout: [dem, dem, VS, tok, tok, VS, tok, tok]
        #  visit_idx:  0    0   1    1    1   2    2    2
        input_ids = torch.tensor([[0, 0, VS, 3, 4, VS, 6, 7]])
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=0)
        m = mask[0, 0]
        # Token at position 7 (visit 2) should NOT see positions 2,3,4 (visit 1)
        for j in [2, 3, 4]:
            self.assertTrue(
                _blocked(m[7, j]),
                f"Token 7 (visit 2) should not attend to position {j} (visit 1) with n_prev=0",
            )
        # Token at position 7 should see positions 5,6 (same visit 2) and 0,1 (demographics)
        for j in [0, 1, 5, 6]:
            self.assertTrue(
                _allowed(m[7, j]),
                f"Token 7 (visit 2) should attend to position {j}, got {m[7,j]}",
            )

    # ------------------------------------------------------------------
    # n_prev_visits=1: current + one prior visit
    # ------------------------------------------------------------------
    def test_n_prev_visits_1(self):
        # Layout: [dem, VS, tok, VS, tok, VS, tok]
        #  visit_idx: 0   1    1   2    2   3    3
        input_ids = torch.tensor([[0, VS, 2, VS, 4, VS, 6]])
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=1)
        m = mask[0, 0]
        # Token 6 is in visit 3; can see visit 2 (positions 3,4) and visit 3 (positions 5,6)
        # and demographics (position 0), but NOT visit 1 (positions 1,2)
        self.assertTrue(_allowed(m[6, 0]), "Should see demographics")
        self.assertTrue(_blocked(m[6, 1]), "Should not see visit 1 [VS] token")
        self.assertTrue(_blocked(m[6, 2]), "Should not see visit 1 body token")
        self.assertTrue(_allowed(m[6, 3]), "Should see visit 2 [VS] token")
        self.assertTrue(_allowed(m[6, 4]), "Should see visit 2 body token")
        self.assertTrue(_allowed(m[6, 5]), "Should see visit 3 [VS] token")
        self.assertTrue(_allowed(m[6, 6]), "Should see itself (visit 3)")

    # ------------------------------------------------------------------
    # Large n_prev_visits ≥ total visits → equivalent to full causal attention
    # ------------------------------------------------------------------
    def test_large_n_prev_visits_is_full_causal(self):
        input_ids = torch.tensor([[0, VS, 2, VS, 4, VS, 6]])
        mask_full = build_local_attention_mask(input_ids, VS, n_prev_visits=999)
        m = mask_full[0, 0]
        L = m.shape[0]
        for i in range(L):
            for j in range(L):
                if j <= i:
                    self.assertTrue(_allowed(m[i, j]), f"({i},{j}) should be allowed")
                else:
                    self.assertTrue(_blocked(m[i, j]), f"({i},{j}) should be blocked")

    # ------------------------------------------------------------------
    # Batch: each sample is processed independently
    # ------------------------------------------------------------------
    def test_batch_independence(self):
        # Two samples with different VS positions
        row0 = torch.tensor([0, VS, 2, 3, VS, 5])   # VS at 1 and 4
        row1 = torch.tensor([0, 0, 0, VS, 4, 5])    # VS at 3 only
        input_ids = torch.stack([row0, row1])
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=0)
        # Sample 0, token 5 (visit 2): should see positions 4 and 5 (visit 2) and 0 (dem)
        m0 = mask[0, 0]
        self.assertTrue(_allowed(m0[5, 0]))   # demographics
        self.assertTrue(_blocked(m0[5, 1]))   # visit 1 VS — out of window
        self.assertTrue(_blocked(m0[5, 2]))   # visit 1 body
        self.assertTrue(_blocked(m0[5, 3]))   # visit 1 body
        self.assertTrue(_allowed(m0[5, 4]))   # visit 2 VS
        self.assertTrue(_allowed(m0[5, 5]))   # self

        # Sample 1, token 5 (visit 1): should see positions 3,4,5 and 0,1,2 (dem)
        m1 = mask[1, 0]
        for j in [0, 1, 2, 3, 4, 5]:
            if j <= 5:
                self.assertTrue(_allowed(m1[5, j]), f"Sample 1 token 5 should see pos {j}")

    # ------------------------------------------------------------------
    # Convention check: values are exactly 0.0 or NEG_INF (no other values)
    # ------------------------------------------------------------------
    def test_values_are_zero_or_neg_inf(self):
        input_ids = torch.tensor([[0, VS, 2, VS, 4]])
        mask = build_local_attention_mask(input_ids, VS, n_prev_visits=1)
        unique_vals = mask.unique()
        for v in unique_vals:
            self.assertTrue(
                v.item() == 0.0 or v.item() == NEG_INF,
                f"Unexpected value in mask: {v.item()}",
            )


if __name__ == "__main__":
    unittest.main()
