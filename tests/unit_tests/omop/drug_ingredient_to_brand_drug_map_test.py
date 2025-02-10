import unittest

import polars as pl

from cehrgpt.omop.vocab_utils import create_drug_ingredient_to_brand_drug_map


class TestCreateDrugIngredientToBrandDrugMap(unittest.TestCase):
    def setUp(self):
        # Enhanced concept DataFrame to include both ingredients and brand drugs
        self.concept_data = {
            "concept_id": [1, 2, 3, 4, 5, 6, 101, 102, 103, 104, 105],
            "domain_id": ["Drug"] * 11,
            "concept_class_id": [
                "Ingredient",
                "Ingredient",
                "Ingredient",
                "Procedure",
                "Chemical",
                "Brand",
                "Brand",
                "Brand",
                "Brand",
                "Brand",
                "Brand",
            ],
        }
        self.concept_df = pl.DataFrame(self.concept_data)

        # Mock data for concept_ancestor DataFrame
        self.concept_ancestor_data = {
            "ancestor_concept_id": [1, 2, 3, 1, 1],
            "descendant_concept_id": [101, 102, 103, 104, 105],  # Brand drugs
        }
        self.concept_ancestor_df = pl.DataFrame(self.concept_ancestor_data)

    def test_typical_use_case(self):
        """Test typical scenario with valid mappings."""
        expected_map = {1: [101, 104, 105], 2: [102], 3: [103]}
        result = create_drug_ingredient_to_brand_drug_map(
            self.concept_df, self.concept_ancestor_df
        )
        self.assertEqual(result, expected_map)

    def test_no_matching_records(self):
        """Test scenario where no ingredients match any ancestors."""
        # Adjust concept data to have no matching 'Ingredient' class
        self.concept_df = self.concept_df.filter(
            pl.col("concept_class_id") != "Ingredient"
        )
        expected_map = {}
        result = create_drug_ingredient_to_brand_drug_map(
            self.concept_df, self.concept_ancestor_df
        )
        self.assertEqual(result, expected_map)

    def test_multiple_mappings(self):
        """Test an ingredient mapped to multiple brand drugs."""
        expected_map = {
            1: [101, 104, 105],  # Expecting multiple mappings from additional setup
            2: [102],
            3: [103],
        }
        result = create_drug_ingredient_to_brand_drug_map(
            self.concept_df, self.concept_ancestor_df
        )
        self.assertEqual(result, expected_map)


if __name__ == "__main__":
    unittest.main()
