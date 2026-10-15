import os
import pickle
import unittest
from unittest import mock

import pandas as pd
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset

from cehrgpt.data.hf_cehrgpt_dataset_mapping import HFCehrGptTokenizationMapping
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sample_data")


class FilterOutInvalidTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_parquet(os.path.join(SAMPLE, "pretrain", "patient_sequence.parquet"))
        concept = pd.read_parquet(os.path.join(SAMPLE, "omop_vocab", "concept"))
        names = {str(r.concept_id): r.concept_name for r in concept.itertuples()}
        args = DataTrainingArguments(
            data_folder="x", dataset_prepared_path="y",
            preprocessing_num_workers=1, preprocessing_batch_size=100,
        )
        cls.tokenizer = CehrGptTokenizer.train_tokenizer(
            Dataset.from_parquet(os.path.join(SAMPLE, "pretrain", "patient_sequence.parquet")),
            names, args, PretrainedEmbeddings(os.path.join(SAMPLE, "pretrained_embeddings")),
        )

    def record(self, extra=()):
        row = self.frame.iloc[0]
        ids = [str(t) for t in row["concept_ids"]]
        keep = len(ids)
        record = {k: list(v) for k, v in row.items() if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == keep}
        record["concept_ids"] = ids + list(extra)
        for key in ("ages", "epoch_times"):
            record[key] = list(record[key]) + [0] * len(extra)
        return record

    def test_the_vocabulary_is_built_once_however_many_tokens_and_records(self):
        mapping = HFCehrGptTokenizationMapping(self.tokenizer)
        calls = []
        real = CehrGptTokenizer.get_vocab

        def counting(tokenizer):
            calls.append(1)
            return real(tokenizer)

        with mock.patch.object(CehrGptTokenizer, "get_vocab", counting):
            for _ in range(5):
                mapping.filter_out_invalid_tokens(self.record(extra=["not in the vocabulary"] * 50))
        # a call for every token would be 5 * 121 calls; each call rebuilds the whole vocabulary
        self.assertLessEqual(len(calls), 1)

    def test_tokens_outside_the_vocabulary_are_removed_from_every_aligned_column(self):
        mapping = HFCehrGptTokenizationMapping(self.tokenizer)
        record = self.record(extra=["unknown token a", "unknown token b"])
        length = len(record["concept_ids"])
        filtered = mapping.filter_out_invalid_tokens(record)
        self.assertEqual(len(filtered["concept_ids"]), length - 2)
        self.assertNotIn("unknown token a", filtered["concept_ids"])
        self.assertEqual(len(filtered["ages"]), length - 2)
        self.assertEqual(len(filtered["epoch_times"]), length - 2)

    def test_nothing_is_removed_when_every_token_is_known(self):
        mapping = HFCehrGptTokenizationMapping(self.tokenizer)
        record = self.record()
        original = list(record["concept_ids"])
        # "0" is renamed first; use a patient without it
        if "0" in original:
            record = self.record()
            record["concept_ids"] = ["Unknown" if t == "0" else t for t in original]
        filtered = mapping.filter_out_invalid_tokens(record)
        self.assertEqual(len(filtered["concept_ids"]), len([t for t in record["concept_ids"] if t in self.tokenizer.get_vocab()]))

    def test_the_cached_vocabulary_is_not_pickled_to_the_workers(self):
        mapping = HFCehrGptTokenizationMapping(self.tokenizer)
        before = len(pickle.dumps(mapping))
        mapping.filter_out_invalid_tokens(self.record())  # builds the cache
        self.assertIn("_valid_concept_ids_cache", mapping.__dict__)
        self.assertGreater(len(pickle.dumps(mapping)), before)  # the cache would travel if it existed at pickling time
        fresh = HFCehrGptTokenizationMapping(self.tokenizer)
        self.assertNotIn("_valid_concept_ids_cache", fresh.__dict__)


if __name__ == "__main__":
    unittest.main()
