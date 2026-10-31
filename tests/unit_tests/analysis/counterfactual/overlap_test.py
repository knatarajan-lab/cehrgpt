import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd
import polars as pl
import torch
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset

from cehrgpt.analysis.counterfactual import create_drug_swap_context as swap
from cehrgpt.analysis.counterfactual import overlap
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample_data")


def context(person_id, concept_ids):
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "input_ids": list(range(10, 10 + len(concept_ids))),
        "ages": list(range(len(concept_ids))),
    }


class IndexPrefixTest(unittest.TestCase):
    def prefixes(self, rows, source=("drug",)):
        return overlap.index_prefix_frame(pl.DataFrame(rows), set(source), 100, None)

    def test_the_context_is_cut_just_before_the_first_index_drug(self):
        frame = self.prefixes([context(1, ["year", "age", "[VS]", "drug", "x", "drug"])])
        row = frame.row(0, named=True)
        self.assertEqual(row["concept_ids"], ["year", "age", "[VS]"])
        self.assertEqual(row["input_ids"], [10, 11, 12])
        self.assertEqual(row["ages"], [0, 1, 2])

    def test_contexts_without_history_before_the_drug_or_without_the_drug_are_dropped(self):
        frame = self.prefixes(
            [context(1, ["drug", "x"]), context(2, ["a", "b"]), context(3, ["a", "drug"])]
        )
        self.assertEqual(frame["person_id"].to_list(), [3])


class PreferenceScoreTest(unittest.TestCase):
    def trimmed(self, p_source, p_comparator, trim=0.05):
        probabilities = pl.DataFrame(
            {
                "person_id": list(range(len(p_source))),
                "p_source": p_source,
                "p_comparator": p_comparator,
            }
        )
        return overlap.trim_by_preference_score(list(range(len(p_source))), probabilities, trim)

    def test_the_score_is_the_comparator_share_of_the_two_classes(self):
        scores = self.trimmed([0.3, 0.1], [0.1, 0.1])
        self.assertAlmostEqual(scores["preference_score"][0], 0.25)
        self.assertAlmostEqual(scores["preference_score"][1], 0.5)

    def test_patients_outside_the_band_are_dropped_in_either_direction(self):
        scores = self.trimmed([0.99, 0.5, 0.01], [0.01, 0.5, 0.99])
        self.assertEqual(scores["retained"].to_list(), [False, True, False])

    def test_the_band_is_symmetric_so_a_patient_passes_in_both_directions(self):
        forward = self.trimmed([0.7, 0.04], [0.3, 0.96])["retained"].to_list()
        reverse = self.trimmed([0.3, 0.96], [0.7, 0.04])["retained"].to_list()
        self.assertEqual(forward, reverse)

    def test_a_patient_that_cannot_be_scored_is_not_retained(self):
        probabilities = pl.DataFrame({"person_id": [1], "p_source": [0.0], "p_comparator": [0.0]})
        scores = overlap.trim_by_preference_score([1, 2], probabilities, 0.05)
        self.assertEqual(scores["retained"].to_list(), [False, False])
        self.assertEqual(scores["preference_score"].null_count(), 2)

    def test_an_invalid_trim_raises(self):
        for trim in (0.0, 0.5, -0.1):
            with self.assertRaises(ValueError):
                self.trimmed([0.5], [0.5], trim)

    def test_a_class_without_tokens_in_the_vocabulary_raises(self):
        with self.assertRaisesRegex(ValueError, "comparator"):
            overlap.class_token_ids({"a": 1}, {"b"}, "comparator")
        self.assertEqual(overlap.class_token_ids({"a": 1, "b": 2}, {"b", "c"}, "x"), [2])


class ModelScoringTest(unittest.TestCase):
    """The probabilities come from the same generate path as the trajectories."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(SAMPLE, "pretrain", "patient_sequence.parquet")
        frame = pd.read_parquet(path)
        concept = pd.read_parquet(os.path.join(SAMPLE, "omop_vocab", "concept"))
        names = {str(r.concept_id): r.concept_name for r in concept.itertuples()}
        data_args = DataTrainingArguments(
            data_folder="x",
            dataset_prepared_path="y",
            preprocessing_num_workers=1,
            preprocessing_batch_size=100,
        )
        cls.tokenizer = CehrGptTokenizer.train_tokenizer(
            Dataset.from_parquet(path),
            names,
            data_args,
            PretrainedEmbeddings(os.path.join(SAMPLE, "pretrained_embeddings")),
        )
        vocab = cls.tokenizer.get_vocab()
        torch.manual_seed(0)
        config = CEHRGPTConfig(
            vocab_size=cls.tokenizer.vocab_size,
            n_positions=256,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_inner=64,
            bos_token_id=cls.tokenizer.end_token_id,
            eos_token_id=cls.tokenizer.end_token_id,
            pad_token_id=cls.tokenizer.pad_token_id,
            include_values=False,
            use_sub_time_tokenization=False,
            include_ttv_prediction=False,
            include_motor_time_to_event=False,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        cls.model = CEHRGPT2LMHeadModel(config).eval()
        counts = Counter(c for cs in frame["concept_ids"] for c in cs if c in vocab)
        cls.source = {"4251306"}
        cls.comparator = {"9202", "40481087"}
        assert cls.source <= set(counts) and cls.comparator <= set(counts)
        df = pl.from_pandas(frame.head(12))
        cls.df = df.with_columns(
            pl.col("person_id").cast(pl.Int64),
            pl.col("concept_ids")
            .map_elements(
                lambda cs: [int(vocab.get(c, cls.tokenizer.oov_token_id)) for c in cs],
                return_dtype=pl.List(pl.Int64),
            )
            .alias("input_ids"),
            pl.lit(0.0).alias("drug_epoch_time"),
            pl.lit("x").alias("drug_concept_id"),
        )

    def test_batched_probabilities_equal_an_unpadded_forward_pass_per_patient(self):
        prefixes = overlap.index_prefix_frame(self.df, self.source, 128, self.tokenizer)
        lengths = {len(c) for c in prefixes["concept_ids"].to_list()}
        self.assertGreater(len(lengths), 1, "the batch needs padding for the test to mean anything")

        scored = overlap.score_patients(
            self.df,
            self.source,
            self.comparator,
            self.tokenizer,
            self.model,
            torch.device("cpu"),
            context_length=128,
            batch_size=5,
        )
        self.assertEqual(sorted(scored["person_id"]), sorted(prefixes["person_id"]))
        vocab = self.tokenizer.get_vocab()
        source_ids = [vocab[c] for c in self.source]
        comparator_ids = [vocab[c] for c in self.comparator]
        for row in prefixes.iter_rows(named=True):
            input_ids = torch.tensor([row["input_ids"]])
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    ages=torch.tensor([row["ages"]]),
                    attention_mask=torch.ones_like(input_ids),
                ).logits[0, -1]
            probabilities = torch.softmax(logits, dim=-1)
            got = scored.filter(pl.col("person_id") == row["person_id"]).row(0, named=True)
            self.assertAlmostEqual(got["p_source"], float(probabilities[source_ids].sum()), delta=1e-6)
            self.assertAlmostEqual(
                got["p_comparator"], float(probabilities[comparator_ids].sum()), delta=1e-6
            )


class SwapOverlapCheckTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "vocab" / "concept_ancestor").mkdir(parents=True)
        ids = [100, 200]
        pl.DataFrame({"ancestor_concept_id": ids, "descendant_concept_id": ids}).write_parquet(
            self.root / "vocab" / "concept_ancestor" / "part.parquet"
        )
        pl.DataFrame(
            {
                "person_id": [1, 2, 3],
                "drug_concept_id": ["100", "100", "200"],
                "drug_epoch_time": [0.0] * 3,
                "era_end_epoch_time": [None] * 3,
                "arm": ["source", "source", "comparator"],
            },
            schema_overrides={"era_end_epoch_time": pl.Float64},
        ).write_parquet(self.root / "drug_info.parquet")
        pl.DataFrame(
            {
                "person_id": [1, 2],
                "concept_ids": [["9202", "100", "x"]] * 2,
                "input_ids": [[5, 10, 7]] * 2,
            }
        ).write_parquet(self.root / "treated.parquet")

    class Tokenizer:
        def get_vocab(self):
            return {"[OOV]": 1, "9202": 5, "100": 10, "200": 20, "x": 7}

    def run_swap(self, **kwargs):
        fake_probabilities = pl.DataFrame(
            {"person_id": [1, 2], "p_source": [0.5, 0.99], "p_comparator": [0.5, 0.01]}
        )
        with mock.patch.object(
            swap.CehrGptTokenizer, "from_pretrained", return_value=self.Tokenizer()
        ), mock.patch.object(overlap, "load_scoring_model", return_value=object()), mock.patch.object(
            overlap, "score_patients", return_value=fake_probabilities
        ):
            swap.create_comparator_context(
                treated_context_path=str(self.root / "treated.parquet"),
                drug_info_path=str(self.root / "drug_info.parquet"),
                vocab_path=str(self.root / "vocab"),
                tokenizer_path="tokenizer",
                source_concept_ids=[100],
                comparator_concept_ids=[200],
                output_path=self.root / "swapped.parquet",
                also_write_treated_path=self.root / "actual.parquet",
                **kwargs,
            )

    def checked(self):
        self.run_swap(
            model_path="model",
            preference_score_trim=0.05,
            overlap_scores_path=self.root / "scores.parquet",
        )

    def test_a_patient_that_fails_is_dropped_from_both_arms(self):
        self.checked()
        self.assertEqual(pl.read_parquet(self.root / "swapped.parquet")["person_id"].to_list(), [1])
        self.assertEqual(pl.read_parquet(self.root / "actual.parquet")["person_id"].to_list(), [1])

    def test_the_scores_of_every_patient_are_written_with_the_decision(self):
        self.checked()
        scores = pl.read_parquet(self.root / "scores.parquet").sort("person_id")
        self.assertEqual(scores["person_id"].to_list(), [1, 2])
        self.assertEqual(scores["retained"].to_list(), [True, False])
        self.assertAlmostEqual(scores["preference_score"][0], 0.5)

    def test_without_a_trim_nobody_is_dropped(self):
        self.run_swap()
        self.assertEqual(len(pl.read_parquet(self.root / "swapped.parquet")), 2)
        self.assertFalse((self.root / "scores.parquet").exists())

    def test_a_trim_needs_a_model_and_a_scores_path(self):
        with self.assertRaisesRegex(ValueError, "model_path"):
            self.run_swap(preference_score_trim=0.05)
        with self.assertRaisesRegex(ValueError, "model_path"):
            self.run_swap(preference_score_trim=0.05, model_path="model")

    def test_an_invalid_trim_raises_before_any_data_is_read(self):
        with self.assertRaisesRegex(ValueError, "trim"):
            self.run_swap(
                model_path="model",
                preference_score_trim=0.7,
                overlap_scores_path=self.root / "scores.parquet",
            )
        self.assertFalse((self.root / "swapped.parquet").exists())


if __name__ == "__main__":
    unittest.main()
