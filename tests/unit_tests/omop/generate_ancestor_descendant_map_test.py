import unittest

import polars as pl

from cehrgpt.omop.vocab_utils import generate_ancestor_descendant_map


class TestGenerateAncestorDescendantMap(unittest.TestCase):
    def setUp(self):
        # Mock data for the DataFrame
        data = {
            "ancestor_concept_id": [1, 2, 3, 1, 2],
            "descendant_concept_id": [10, 20, 30, 40, 50],
        }
        self.df = pl.DataFrame(data)

    def test_typical_case(self):
        """Test with typical input data."""
        concept_ids = ["1", "2"]  # Inputs as strings
        expected = {"1": ["10", "40"], "2": ["20", "50"]}
        result = generate_ancestor_descendant_map(self.df, concept_ids)
        self.assertEqual(result, expected)

    def test_nonexistent_ids(self):
        """Test with IDs that do not exist in the DataFrame."""
        concept_ids = ["4", "5"]
        expected = {}
        result = generate_ancestor_descendant_map(self.df, concept_ids)
        self.assertEqual(result, expected)

    def test_non_numeric_filter(self):
        """Test input with non-numeric strings included."""
        concept_ids = ["1", "two", "3"]
        expected = {"1": ["10", "40"], "3": ["30"]}
        result = generate_ancestor_descendant_map(self.df, concept_ids)
        self.assertEqual(result, expected)

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        empty_df = pl.DataFrame(
            {"ancestor_concept_id": [], "descendant_concept_id": []}
        )
        concept_ids = ["1", "2"]
        expected = {}
        result = generate_ancestor_descendant_map(empty_df, concept_ids)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
