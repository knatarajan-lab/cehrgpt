import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import GenerationConfig

from cehrgpt.time_to_event.time_to_event_model import (
    TimeToEventModel,
    TimeToEventStoppingCriteria,
)
from cehrgpt.time_to_event.time_to_event_prediction import flush_to_disk_if_full


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"D10": 0, "VS": 1, "OUT": 2, "X": 3, "PAD": 4}

    def get_vocab(self):
        return self.vocab

    def encode(self, tokens):
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids, **kwargs):
        inverse = {value: key for key, value in self.vocab.items()}
        return [inverse[int(token_id)] for token_id in token_ids]


class SpecialTokenTokenizer(FakeTokenizer):
    def __init__(self):
        self.vocab = {
            "year:2000": 0,
            "age:20": 1,
            "Gender/1": 2,
            "Race/1": 3,
            "VS": 4,
            "A": 5,
            "B": 6,
            "C": 7,
            "SPECIAL": 8,
        }

    def encode(self, tokens):
        return [self.vocab["SPECIAL"]] + [self.vocab[token] for token in tokens]


class FakeGenerationModel:
    def __init__(self):
        self.config = SimpleNamespace(n_positions=8)
        self.batch_sizes = []

    def eval(self):
        return self

    def generate(self, inputs, generation_config, **kwargs):
        count = generation_config.num_return_sequences
        self.batch_sizes.append(count)
        continuation = torch.full((count, 1), 3, dtype=inputs.dtype)
        return SimpleNamespace(
            sequences=torch.cat([inputs.expand(count, -1), continuation], dim=1)
        )


class TestTimeToEventStoppingCriteria(unittest.TestCase):
    def test_stops_on_an_eligible_outcome(self):
        criteria = TimeToEventStoppingCriteria(
            FakeTokenizer(), ["OUT"], 0, -1, 0, 30
        )
        stopped = criteria(torch.tensor([[2], [3]]), scores=None)
        self.assertEqual(stopped.tolist(), [True, False])

    def test_time_limit_matches_post_time_token_semantics(self):
        criteria = TimeToEventStoppingCriteria(
            FakeTokenizer(), ["OUT"], 0, -1, 0, 15
        )
        self.assertEqual(
            criteria(torch.tensor([[0]]), scores=None).tolist(), [False]
        )
        self.assertEqual(
            criteria(torch.tensor([[0, 0]]), scores=None).tolist(), [False]
        )
        # The existing reducer checks the accumulated time before processing
        # the current non-time token, so the boundary is crossed here.
        self.assertEqual(
            criteria(torch.tensor([[0, 0, 3]]), scores=None).tolist(), [True]
        )


class TestSimulationBatching(unittest.TestCase):
    def test_last_generation_batch_is_not_over_generated(self):
        tokenizer = FakeTokenizer()
        model = FakeGenerationModel()
        generation_config = GenerationConfig(
            max_length=8,
            max_new_tokens=1,
            num_return_sequences=5,
            do_sample=True,
        )
        predictor = TimeToEventModel(
            tokenizer=tokenizer,
            model=model,
            outcome_events=["OUT"],
            generation_config=generation_config,
            batch_size=2,
        )

        sequences = predictor.simulate(["X"])

        self.assertEqual(model.batch_sizes, [2, 2, 1])
        self.assertEqual(len(sequences), 5)
        self.assertTrue(all(sequence == ["X"] for sequence in sequences))
        self.assertEqual(generation_config.num_return_sequences, 5)

    def test_prompt_truncation_counts_special_tokens_and_keeps_demographics(self):
        tokenizer = SpecialTokenTokenizer()
        predictor = TimeToEventModel(
            tokenizer=tokenizer,
            model=FakeGenerationModel(),
            outcome_events=[],
            generation_config=GenerationConfig(max_length=8, max_new_tokens=1),
        )
        history = [
            "year:2000",
            "age:20",
            "Gender/1",
            "Race/1",
            "VS",
            "A",
            "VS",
            "B",
            "C",
        ]

        prompt = predictor.prepare_prompt(history)

        self.assertEqual(prompt[:4], history[:4])
        self.assertEqual(prompt[4], "VS")
        self.assertLessEqual(len(tokenizer.encode(prompt)), 7)


class TestPredictionFlush(unittest.TestCase):
    def test_force_flush_persists_remainder_before_marking_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_folder = os.path.join(tmp, "predictions")
            lock_folder = os.path.join(tmp, "locks")
            os.makedirs(output_folder)
            os.makedirs(lock_folder)
            sample_id = "1_2020_01_01"
            open(os.path.join(lock_folder, f"{sample_id}.lock"), "w").close()
            outputs = [
                {
                    "_sample_identifier": sample_id,
                    "subject_id": 1,
                    "prediction_time": datetime.datetime(2020, 1, 1),
                    "visit_counter": 1,
                    "boolean_value": False,
                    "predicted_boolean_probability": 0.0,
                    "predicted_boolean_value": None,
                    "time_to_event": None,
                    "trials": None,
                }
            ]

            flush_to_disk_if_full(outputs, output_folder, 16, lock_folder)
            self.assertEqual(os.listdir(output_folder), [])
            self.assertTrue(os.path.exists(os.path.join(lock_folder, f"{sample_id}.lock")))

            flush_to_disk_if_full(
                outputs, output_folder, 16, lock_folder, force=True
            )

            parquet_files = os.listdir(output_folder)
            self.assertEqual(len(parquet_files), 1)
            result = pd.read_parquet(os.path.join(output_folder, parquet_files[0]))
            self.assertEqual(result["subject_id"].tolist(), [1])
            self.assertNotIn("_sample_identifier", result.columns)
            self.assertFalse(os.path.exists(os.path.join(lock_folder, f"{sample_id}.lock")))
            self.assertTrue(os.path.exists(os.path.join(lock_folder, f"{sample_id}.done")))
            self.assertEqual(outputs, [])


if __name__ == "__main__":
    unittest.main()
