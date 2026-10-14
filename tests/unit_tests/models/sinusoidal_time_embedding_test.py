import unittest

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import (
    SinusoidalTimeEmbedding,
    TimeTokenEmbeddingReplacer,
)


class SinusoidalTimeEmbeddingTest(unittest.TestCase):
    def test_config_repr_summarizes_time_tokens_without_dumping_mappings(self):
        config = CEHRGPTConfig(
            vocab_size=5,
            n_embd=8,
            n_layer=1,
            n_head=1,
            time_token_embedding_type="sinusoidal",
            token_to_time_token_mapping={-1: [0, 0, 0], 2: [1, 0, 0]},
            time_token_values={2: 1.0},
        )

        representation = repr(config)

        self.assertIn("time_token_embedding_type='sinusoidal'", representation)
        self.assertIn("time_token_count=1", representation)
        self.assertNotIn("time_token_values", representation)
        self.assertNotIn("token_to_time_token_mapping", representation)

    def test_supports_fractional_days(self):
        encoder = SinusoidalTimeEmbedding(embedding_dim=16)
        result = encoder(torch.tensor([1.0, 1.5, 2.0]))
        self.assertEqual(result.shape, (3, 16))
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.equal(result[0], result[1]))

    def test_equal_intervals_have_equal_distance(self):
        encoder = SinusoidalTimeEmbedding(embedding_dim=16)
        values = encoder(torch.tensor([1.0, 2.0, 4.0, 5.0]))
        first_distance = torch.linalg.vector_norm(values[0] - values[1])
        second_distance = torch.linalg.vector_norm(values[2] - values[3])
        torch.testing.assert_close(first_distance, second_distance)

    def test_replaces_only_configured_time_tokens(self):
        config = CEHRGPTConfig(
            vocab_size=5,
            n_embd=8,
            n_layer=1,
            n_head=1,
            time_token_embedding_type="sinusoidal",
            token_to_time_token_mapping={-1: [0, 0, 0], 2: [1, 0, 0], 4: [7, 0, 0]},
            time_token_values={2: 1.0, 4: 7.0},
        )
        replacer = TimeTokenEmbeddingReplacer(config)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        original = torch.randn(1, 4, 8)
        result = replacer(input_ids, original)

        torch.testing.assert_close(result[:, 0], original[:, 0])
        torch.testing.assert_close(result[:, 2], original[:, 2])
        self.assertFalse(torch.equal(result[:, 1], original[:, 1]))
        self.assertFalse(torch.equal(result[:, 3], original[:, 3]))

    def test_lookup_values_remain_float32_when_default_dtype_is_bfloat16(self):
        config = CEHRGPTConfig(
            vocab_size=5,
            n_embd=8,
            n_layer=1,
            n_head=1,
            time_token_embedding_type="sinusoidal",
            token_to_time_token_mapping={-1: [0, 0, 0], 2: [1, 0, 0]},
            time_token_values={2: 1.0},
        )
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            replacer = TimeTokenEmbeddingReplacer(config)
        finally:
            torch.set_default_dtype(original_dtype)

        self.assertEqual(replacer.values_by_token_id.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
