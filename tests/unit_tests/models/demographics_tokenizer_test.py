import unittest
from unittest.mock import patch

from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers

from cehrgpt.models.demographic_tokenizer import (
    DemographicTokenizer,
    build_token_index_map,
    map_demographics,
)
from cehrgpt.models.tokenization_hf_cehrgpt import OUT_OF_VOCABULARY_TOKEN


def create_mock_tokenizer(tokens):
    """Utility function to create a mock tokenizer for unit tests."""
    tokenizer = Tokenizer(
        models.WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict())
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.train_from_iterator(
        tokens,
        trainer=trainers.WordLevelTrainer(
            special_tokens=[OUT_OF_VOCABULARY_TOKEN], vocab_size=len(set(tokens)) + 1
        ),
    )
    tokenizer.add_tokens([AddedToken(token, single_word=True) for token in tokens])
    return tokenizer


class TestDemographicTokenizer(unittest.TestCase):
    def setUp(self):
        # Setup mock tokenizers for the tokenizer
        self.year_tokens = ["2020", "2021"]
        self.age_tokens = ["30", "31"]
        self.gender_tokens = ["Male", "Female"]
        self.race_tokens = ["White", "Black"]

        self.year_tokenizer = create_mock_tokenizer(self.year_tokens)
        self.age_tokenizer = create_mock_tokenizer(self.age_tokens)
        self.gender_tokenizer = create_mock_tokenizer(self.gender_tokens)
        self.race_tokenizer = create_mock_tokenizer(self.race_tokens)

        self.tokenizer = DemographicTokenizer(
            initial_year_tokenizer=self.year_tokenizer,
            initial_age_tokenizer=self.age_tokenizer,
            gender_tokenizer=self.gender_tokenizer,
            race_tokenizer=self.race_tokenizer,
        )

    def test_initialization(self):
        # Test correct initialization of properties
        self.assertEqual(self.tokenizer.num_initial_years, 2 + 1)
        self.assertEqual(self.tokenizer.num_initial_ages, 2 + 1)
        self.assertEqual(self.tokenizer.num_genders, 2 + 1)
        self.assertEqual(self.tokenizer.num_races, 2 + 1)

    @patch("cehrgpt.models.demographic_tokenizer.build_token_index_map")
    def test_train_tokenizer(self, mock_create_mapping):
        # Setup the side effects of the mock to return these tokenizers
        mock_create_mapping.side_effect = [
            self.year_tokenizer,
            self.age_tokenizer,
            self.gender_tokenizer,
            self.race_tokenizer,
        ]

        # Prepare a dataset and training arguments
        dataset = Dataset.from_dict({"concept_ids": [["2020", "30", "Male", "White"]]})
        data_args = DataTrainingArguments(
            data_folder=".",
            dataset_prepared_path=".",
            preprocessing_batch_size=10,
            streaming=False,
            preprocessing_num_workers=1,
        )

        # Train the tokenizer
        tokenizer = DemographicTokenizer.train_tokenizer(dataset, data_args)

        # Assert that the correct vocab sizes have been set by the tokenizer
        self.assertEqual(tokenizer.num_initial_years, len(self.year_tokens) + 1)
        self.assertEqual(tokenizer.num_initial_ages, len(self.age_tokens) + 1)
        self.assertEqual(tokenizer.num_genders, len(self.gender_tokens) + 1)
        self.assertEqual(tokenizer.num_races, len(self.race_tokens) + 1)

    def test_build_token_index_map(self):
        tokens = ["hello", "world", "test", "hello"]
        tokenizer = build_token_index_map(tokens)

        # Ensure tokenizer is created
        self.assertIsInstance(tokenizer, Tokenizer)

        # Test encoding and decoding
        encoded = tokenizer.encode(
            "hello world test unknown".split(" "), is_pretokenized=True
        )
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

        # Check if 'unknown' token is handled as [UNK]
        self.assertTrue(OUT_OF_VOCABULARY_TOKEN in decoded.split())

        # Check correctness of decoded tokens (order and correctness)
        self.assertEqual(decoded, "hello world test " + OUT_OF_VOCABULARY_TOKEN)

        # Validate the vocabulary size (should be 3 unique tokens + unk token)
        self.assertEqual(len(tokenizer.get_vocab()), 4)

    def test_map_demographics(self):
        # Test the function that extracts demographic tokens from a batch
        batch = {
            "concept_ids": [
                ["2020", "30", "Male", "White"],
                ["2021", "31", "Female", "Black"],
            ]
        }
        result = map_demographics(batch)
        expected = {
            "start_year": ["2020", "2021"],
            "start_age": ["30", "31"],
            "gender": ["Male", "Female"],
            "race": ["White", "Black"],
        }
        self.assertEqual(result.items(), expected.items())


if __name__ == "__main__":
    unittest.main()
