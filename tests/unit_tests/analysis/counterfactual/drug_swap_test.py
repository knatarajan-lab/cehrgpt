import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import polars as pl

from cehrgpt.analysis.counterfactual import create_drug_swap_context as swap

SOURCE, COMPARATOR_A, COMPARATOR_B = 100, 200, 300
VOCAB = {"[OOV]": 1, "9202": 5, "100": 10, "200": 20, "300": 30, "x": 7}


class StubTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab(self):
        return self.vocab


def stub_from_pretrained(vocab):
    return mock.patch.object(
        swap.CehrGptTokenizer, "from_pretrained", return_value=StubTokenizer(vocab)
    )


class DrugSwapTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        (root / "vocab" / "concept_ancestor").mkdir(parents=True)
        pl.DataFrame(
            {
                # 101 is a drug of the source class that the tokenizer does not know
                "ancestor_concept_id": [SOURCE, COMPARATOR_A, COMPARATOR_B, SOURCE],
                "descendant_concept_id": [SOURCE, COMPARATOR_A, COMPARATOR_B, 101],
            }
        ).write_parquet(root / "vocab" / "concept_ancestor" / "part.parquet")
        pl.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "drug_concept_id": ["100", "100", "200", "300", "101"],
                "drug_epoch_time": [0.0] * 5,
                "era_end_epoch_time": [None] * 5,
                "arm": ["source", "source", "comparator", "comparator", "source"],
            },
            schema_overrides={"era_end_epoch_time": pl.Float64},
        ).write_parquet(root / "drug_info.parquet")
        pl.DataFrame(
            {
                "person_id": [1, 2, 5],
                "concept_ids": [["9202", "100", "x"], ["9202", "100", "x"], ["9202", "101", "x"]],
                "input_ids": [[5, 10, 7], [5, 10, 7], [5, 1, 7]],
                "epoch_times": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            }
        ).write_parquet(root / "treated.parquet")
        self.root = root

    def run_swap(self, comparators, vocab=VOCAB, **kwargs):
        with stub_from_pretrained(vocab):
            swap.create_comparator_context(
                treated_context_path=str(self.root / "treated.parquet"),
                drug_info_path=str(self.root / "drug_info.parquet"),
                vocab_path=str(self.root / "vocab"),
                tokenizer_path="tokenizer",
                source_concept_ids=[SOURCE],
                comparator_concept_ids=comparators,
                output_path=self.root / "swapped.parquet",
                **kwargs,
            )
        return pl.read_parquet(self.root / "swapped.parquet")

    def test_the_input_ids_follow_the_swapped_concept(self):
        swapped = self.run_swap([COMPARATOR_A, COMPARATOR_B])
        self.assertEqual(len(swapped), 2)
        for row in swapped.iter_rows(named=True):
            drug = row["drug_concept_id"]
            self.assertEqual(row["concept_ids"][1], drug)
            self.assertEqual(row["input_ids"], [5, VOCAB[drug]])

    def test_the_swapped_context_ends_with_the_comparator_drug(self):
        swapped = self.run_swap([COMPARATOR_A, COMPARATOR_B])
        for row in swapped.iter_rows(named=True):
            self.assertEqual(row["concept_ids"], ["9202", row["drug_concept_id"]])
            self.assertEqual(row["epoch_times"], [1.0, 2.0])

    def test_the_actual_context_ends_with_the_original_drug(self):
        self.run_swap([COMPARATOR_A, COMPARATOR_B], also_write_treated_path=self.root / "actual.parquet")
        actual = pl.read_parquet(self.root / "actual.parquet")
        self.assertEqual(sorted(actual["person_id"].to_list()), [1, 2])
        for row in actual.iter_rows(named=True):
            self.assertEqual(row["concept_ids"], ["9202", "100"])
            self.assertEqual(row["input_ids"], [5, 10])
            self.assertEqual(row["epoch_times"], [1.0, 2.0])

    def test_cut_after_position_only_cuts_sequence_columns(self):
        row = {"person_id": 1, "concept_ids": ["a", "b", "c"], "ages": [1, 2, 3], "static": [9]}
        self.assertEqual(
            swap.cut_after_position(row, 1),
            {"person_id": 1, "concept_ids": ["a", "b"], "ages": [1, 2], "static": [9]},
        )

    def test_a_comparator_missing_from_the_vocabulary_raises(self):
        vocab = {k: v for k, v in VOCAB.items() if k != "300"}
        with self.assertRaisesRegex(ValueError, "300"):
            self.run_swap([COMPARATOR_A, COMPARATOR_B], vocab=vocab)

    def test_every_missing_comparator_is_listed(self):
        with self.assertRaisesRegex(ValueError, "2 comparator.*200, 300"):
            self.run_swap([COMPARATOR_A, COMPARATOR_B], vocab={"[OOV]": 1})

    def test_a_patient_whose_own_drug_is_unknown_to_the_tokenizer_is_left_out(self):
        swapped = self.run_swap([COMPARATOR_A, COMPARATOR_B])
        self.assertEqual(sorted(swapped["person_id"].to_list()), [1, 2])

    def test_unknown_comparators_are_left_out_on_request_and_the_rest_are_sampled(self):
        vocab = {k: v for k, v in VOCAB.items() if k != "300"}
        swapped = self.run_swap([COMPARATOR_A, COMPARATOR_B], vocab=vocab, drop_unknown_comparators=True)
        self.assertEqual(len(swapped), 2)
        self.assertEqual(set(swapped["drug_concept_id"].to_list()), {"200"})
        for row in swapped.iter_rows(named=True):
            self.assertEqual(row["input_ids"], [5, 20])

    def test_nothing_is_left_when_no_comparator_is_known(self):
        with self.assertRaisesRegex(ValueError, "none of the comparator ingredients"):
            self.run_swap([COMPARATOR_A, COMPARATOR_B], vocab={"[OOV]": 1, "100": 10}, drop_unknown_comparators=True)

    def test_the_strict_error_names_the_option(self):
        vocab = {k: v for k, v in VOCAB.items() if k != "300"}
        with self.assertRaisesRegex(ValueError, "--drop_unknown_comparators"):
            self.run_swap([COMPARATOR_A, COMPARATOR_B], vocab=vocab)

    def test_the_check_happens_before_any_data_is_read(self):
        with stub_from_pretrained({"[OOV]": 1}):
            with self.assertRaises(ValueError):
                swap.create_comparator_context(
                    treated_context_path="does/not/exist",
                    drug_info_path="does/not/exist",
                    vocab_path="does/not/exist",
                    tokenizer_path="tokenizer",
                    source_concept_ids=[SOURCE],
                    comparator_concept_ids=[COMPARATOR_A],
                    output_path=self.root / "swapped.parquet",
                )
        self.assertFalse(os.path.exists(self.root / "swapped.parquet"))


if __name__ == "__main__":
    unittest.main()
