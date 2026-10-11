import os
import pickle
import tempfile
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from cehrgpt.models.special_tokens import (
    END_TOKEN,
    OUT_OF_VOCABULARY_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

LAB_STATS = ["_numeric_lab_stats", "_categorical_lab_stats", "_numeric_event_statistics"]
MOTOR = ["_motor_task_info", "_motor_time_to_event_codes", "_motor_code_to_id_mapping"]


def word_level_tokenizer(tokens):
    vocab = {t: i for i, t in enumerate([PAD_TOKEN, OUT_OF_VOCABULARY_TOKEN, START_TOKEN, END_TOKEN] + tokens)}
    return Tokenizer(WordLevel(vocab=vocab, unk_token=OUT_OF_VOCABULARY_TOKEN))


def make_tokenizer():
    # A bulky payload standing in for the splines of the lab bins
    bins = [
        {"bin_index": 0, "start_val": 0.0, "end_val": 1.0, "spline": b"x" * 100_000},
        {"bin_index": 1, "start_val": 1.0, "end_val": 2.0, "spline": b"x" * 100_000},
    ]
    return CehrGptTokenizer(
        word_level_tokenizer(["year:2020", "age:40", "Gender/F", "Race/5", "3000963", "D1"]),
        word_level_tokenizer(["BIN:0", "BIN:1"]),
        word_level_tokenizer(["D1"]),
        {},
        {},
        [
            {
                "concept_id": "3000963",
                "unit": "g/dL",
                "mean": 1.0,
                "std": 1.0,
                "count": 5,
                "value_outlier_std": 5,
                "bins": bins,
            }
        ],
        {("3000", "positive"): 3},
        {},
        motor_task_info={"motor_time_to_event_codes": ["B", "A"]},
    )


class TestLazyTokenizerLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make_tokenizer().save_pretrained(self.tmp.name)

    def load(self):
        return CehrGptTokenizer.from_pretrained(self.tmp.name)

    def test_nothing_heavy_is_read_by_from_pretrained(self):
        tokenizer = self.load()
        for name in LAB_STATS + MOTOR:
            self.assertNotIn(name, tokenizer.__dict__)
        # The vocabulary and ids don't need any of it
        self.assertEqual(tokenizer.encode(["year:2020", "3000963"]), [4, 8])
        self.assertEqual(tokenizer.end_token_id, 3)
        for name in LAB_STATS + MOTOR:
            self.assertNotIn(name, tokenizer.__dict__)

    def test_first_use_loads_the_same_values_as_an_eager_tokenizer(self):
        eager, lazy = make_tokenizer(), self.load()
        self.assertEqual(lazy.numeric_concept_ids, eager.numeric_concept_ids)
        self.assertEqual(lazy.categorical_concept_ids, eager.categorical_concept_ids)
        self.assertEqual(lazy.normalize("3000963", "g/dL", 0.5), "BIN:0")
        self.assertEqual(lazy.normalize("3000963", "g/dL", 1.5), "BIN:1")
        self.assertEqual(lazy.normalize("3000963", "g/dL", 9.0), eager.normalize("3000963", "g/dL", 9.0))
        self.assertEqual(lazy._motor_time_to_event_codes, ["B", "A"])
        self.assertEqual(lazy._motor_code_to_id_mapping, {"A": 0, "B": 1})
        self.assertEqual(lazy.lab_token_ids, eager.lab_token_ids)

    def test_groups_are_loaded_independently(self):
        tokenizer = self.load()
        tokenizer.numeric_concept_ids
        for name in LAB_STATS:
            self.assertIn(name, tokenizer.__dict__)
        for name in MOTOR:
            self.assertNotIn(name, tokenizer.__dict__)

    def test_pickle_leaves_out_what_was_loaded_and_the_copy_reads_it_again(self):
        tokenizer = self.load()
        small = len(pickle.dumps(tokenizer))
        tokenizer.numeric_concept_ids
        self.assertIn("_numeric_lab_stats", tokenizer.__dict__)
        loaded = len(pickle.dumps(tokenizer))
        self.assertEqual(loaded, small)
        self.assertLess(small, 50_000)

        copy = pickle.loads(pickle.dumps(tokenizer))
        self.assertNotIn("_numeric_lab_stats", copy.__dict__)
        self.assertEqual(copy.normalize("3000963", "g/dL", 0.5), "BIN:0")

    def test_eager_tokenizer_keeps_everything_when_pickled(self):
        eager = make_tokenizer()
        self.assertGreater(len(pickle.dumps(eager)), 200_000)
        copy = pickle.loads(pickle.dumps(eager))
        self.assertEqual(copy.normalize("3000963", "g/dL", 1.5), "BIN:1")

    def test_release_lazy_data(self):
        tokenizer = self.load()
        tokenizer.numeric_concept_ids
        tokenizer.release_lazy_data()
        for name in LAB_STATS:
            self.assertNotIn(name, tokenizer.__dict__)
        self.assertEqual(tokenizer.normalize("3000963", "g/dL", 0.5), "BIN:0")

    def test_assigned_values_win_and_survive_release_and_pickling(self):
        tokenizer = self.load()
        tokenizer._motor_task_info = {}
        tokenizer._motor_time_to_event_codes = []
        tokenizer._motor_code_to_id_mapping = {}
        tokenizer.numeric_concept_ids  # loads the lab stats only
        tokenizer.release_lazy_data()
        copy = pickle.loads(pickle.dumps(tokenizer))
        self.assertEqual(copy._motor_time_to_event_codes, [])
        self.assertEqual(copy._motor_task_info, {})

    def test_saving_a_lazy_tokenizer_round_trips(self):
        with tempfile.TemporaryDirectory() as other:
            self.load().save_pretrained(other)
            self.assertEqual(
                sorted(os.listdir(other)), sorted(os.listdir(self.tmp.name))
            )
            reloaded = CehrGptTokenizer.from_pretrained(other)
            self.assertEqual(reloaded.normalize("3000963", "g/dL", 1.5), "BIN:1")
            self.assertEqual(reloaded._motor_time_to_event_codes, ["B", "A"])


if __name__ == "__main__":
    unittest.main()
