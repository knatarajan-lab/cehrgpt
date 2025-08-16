import unittest

import torch

from cehrgpt.models.gpt2 import _get_unpad_data
from cehrgpt.models.hf_cehrgpt import (
    create_sample_packing_attention_mask,
    is_sample_pack,
)


class TestSamplePackingFunctions(unittest.TestCase):

    def test_create_sample_packing_attention_mask(self):
        test_cases = [
            {
                "name": "two segments split by a 0",
                "input": torch.tensor([[1, 1, 0, 1, 1, 1, 1]]),
                "expected": torch.tensor(
                    [
                        [1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1],
                    ]
                ),
            },
            {
                "name": "all ones (single segment)",
                "input": torch.tensor([[1, 1, 1, 1]]),
                "expected": torch.ones((4, 4), dtype=torch.int),
            },
            {
                "name": "all zeros (no attention)",
                "input": torch.tensor([[0, 0, 0, 0]]),
                "expected": torch.zeros((4, 4), dtype=torch.int),
            },
            {
                "name": "multiple segments",
                "input": torch.tensor([[1, 1, 0, 1, 0, 1]]),
                "expected": torch.tensor(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                ),
            },
        ]

        for case in test_cases:
            with self.subTest(msg=case["name"]):
                result = create_sample_packing_attention_mask(case["input"])
                self.assertTrue(
                    torch.equal(result, case["expected"]),
                    f"{case['name']} failed.\nExpected:\n{case['expected']}\nGot:\n{result}",
                )

    def test_create_sample_packing_attention_mask(self):
        # Simulate a packed attention mask with interleaved zeros
        attention_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 0, 0]])
        create_sample_packing_attention_mask(attention_mask)

    def test_is_sample_pack_with_packed_input(self):
        # Simulate a packed attention mask with interleaved zeros
        attention_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 0, 0]])
        result = is_sample_pack(attention_mask)
        self.assertTrue(result, "Expected True for packed attention mask")

    def test_is_sample_pack_with_unpacked_input(self):
        # Simulate an unpacked attention mask (no interleaved zeros)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        result = is_sample_pack(attention_mask)
        self.assertFalse(result, "Expected False for unpacked attention mask")

        # Simulate an unpacked attention mask (no interleaved zeros)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        result = is_sample_pack(attention_mask)
        self.assertFalse(result, "Expected False for unpacked attention mask")

    def test_get_unpad_data_with_unpacked_input(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
        expected_indices = torch.tensor([0, 1, 2, 3, 4, 5])
        expected_cu_seqlens = torch.tensor([0, 6])
        expected_max_seqlen = 6

        self.assertTrue(
            torch.equal(indices, expected_indices),
            "Indices do not match for unpacked input",
        )
        self.assertTrue(
            torch.equal(cu_seqlens, expected_cu_seqlens),
            "Cumulative sequence lengths do not match for unpacked input",
        )
        self.assertEqual(
            max_seqlen,
            expected_max_seqlen,
            "Max sequence length does not match for unpacked input",
        )

    def test_get_unpad_data_with_packed_input(self):
        attention_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 0, 0]])
        indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
        expected_indices = torch.tensor([0, 1, 2, 4, 5])
        expected_cu_seqlens = torch.tensor([0, 3, 5])
        expected_max_seqlen = 3

        self.assertTrue(
            torch.equal(indices, expected_indices),
            "Indices do not match for packed input",
        )
        self.assertTrue(
            torch.equal(cu_seqlens, expected_cu_seqlens),
            "Cumulative sequence lengths do not match for packed input",
        )
        self.assertEqual(
            max_seqlen,
            expected_max_seqlen,
            "Max sequence length does not match for packed input",
        )


if __name__ == "__main__":
    unittest.main()
