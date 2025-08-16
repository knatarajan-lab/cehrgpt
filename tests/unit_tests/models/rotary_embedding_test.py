import math
import unittest

import torch

from cehrgpt.models.gpt2 import RotaryPositionEmbedding


class TestRotaryPositionEmbedding(unittest.TestCase):

    def test_known_2d_rotation(self):
        emb = RotaryPositionEmbedding(dim=2)
        x = torch.tensor([[1.0, 0.0]])
        time = torch.tensor([[0.5]])
        output = emb(x, time)
        theta = 0.5 * emb.inv_freq[0, 0, 0]
        expected = torch.tensor([[math.cos(theta), math.sin(theta)]])
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-5),
            f"Expected {expected}, got {output}",
        )

    def test_norm_preservation(self):
        emb = RotaryPositionEmbedding(dim=8)
        x = torch.randn(10, 8)
        time = torch.randint(0, 10, size=(10, 1)).float()
        x_rot = emb(x, time)
        original_norm = torch.norm(x, dim=-1)
        rotated_norm = torch.norm(x_rot, dim=-1)
        self.assertTrue(torch.allclose(original_norm, rotated_norm, atol=1e-5))

    def test_relative_position_consistency(self):
        emb = RotaryPositionEmbedding(dim=2)
        x = torch.tensor([[1.0, 0.0]])
        t1 = torch.tensor([[1.0]])
        t2 = torch.tensor([[2.0]])
        t3 = torch.tensor([[3.0]])
        d1 = emb(x, t2) - emb(x, t1)
        d2 = emb(x, t3) - emb(x, t2)
        self.assertFalse(
            torch.allclose(d1, d2, atol=1e-3)
        )  # Should not be exactly equal
        self.assertTrue(torch.isfinite(d1 - d2).all())  # But valid difference

    def test_dot_product_relative_shift_invariance(self):
        dim = 8
        emb = RotaryPositionEmbedding(dim=dim)
        x = torch.randn(1, dim)
        y = torch.randn(1, dim)

        t1 = torch.tensor([[3.0]])
        t2 = torch.tensor([[7.0]])
        delta = 5.0

        # RoPE embeddings at original positions
        x1 = emb(x, t1)
        y1 = emb(y, t2)

        # RoPE embeddings at shifted positions
        x2 = emb(x, t1 + delta)
        y2 = emb(y, t2 + delta)

        dot1 = torch.sum(x1 * y1, dim=-1)
        dot2 = torch.sum(x2 * y2, dim=-1)

        self.assertTrue(
            torch.allclose(dot1, dot2, atol=1e-5),
            f"Dot products differ: {dot1.item():.6f} vs {dot2.item():.6f}",
        )

    def test_batched_inputs(self):
        emb = RotaryPositionEmbedding(dim=6)
        x = torch.randn(2, 6)
        time = torch.tensor([[1.0], [3.0]])
        x_rot = emb(x, time)
        self.assertEqual(x_rot.shape, x.shape)
        self.assertTrue(torch.isfinite(x_rot).all())

    def test_dtype_preservation(self):
        emb = RotaryPositionEmbedding(dim=4)
        x = torch.randn(5, 4, dtype=torch.float32)
        time = torch.randint(0, 10, (5, 1)).float()
        out = emb(x, time)
        self.assertEqual(out.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
