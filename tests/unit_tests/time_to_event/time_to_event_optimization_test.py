import datetime
import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import GenerationConfig

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token, is_visit_start
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
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

    @property
    def lab_token_ids(self):
        return []


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
        self.generation_kwargs = []

    def eval(self):
        return self

    def generate(self, inputs, generation_config, **kwargs):
        count = generation_config.num_return_sequences
        self.batch_sizes.append(count)
        self.generation_kwargs.append(kwargs)
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

    def test_matches_old_reducer_stop_boundary_exhaustively(self):
        tokenizer = FakeTokenizer()
        tokens = ["D10", "VS", "OUT", "X"]
        task_windows = [
            (0, -1, 0, 30),
            (1, 2, 0, 30),
            (0, -1, 10, 20),
            (2, 3, 10, 25),
        ]

        def old_stop_index(sequence, future_start, future_end, time_start, time_end):
            visits = 0
            days = 0
            for index, token in enumerate(sequence):
                visits += int(is_visit_start(token))
                if visits > future_end != -1 or days > time_end != -1:
                    return index
                if is_att_token(token):
                    days += extract_time_interval_in_days(token)
                elif visits >= future_start and days >= time_start and token == "OUT":
                    return index
            return None

        for window in task_windows:
            for length in range(1, 6):
                for sequence in itertools.product(tokens, repeat=length):
                    criteria = TimeToEventStoppingCriteria(
                        tokenizer, ["OUT"], *window
                    )
                    actual_stop = None
                    token_ids = []
                    for index, token in enumerate(sequence):
                        token_ids.append(tokenizer.vocab[token])
                        if criteria(torch.tensor([token_ids]), scores=None).item():
                            actual_stop = index
                            break
                    self.assertEqual(
                        actual_stop,
                        old_stop_index(sequence, *window),
                        msg=f"sequence={sequence}, window={window}",
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

    def test_early_stopping_can_be_disabled_for_differential_validation(self):
        tokenizer = FakeTokenizer()
        model = FakeGenerationModel()
        predictor = TimeToEventModel(
            tokenizer=tokenizer,
            model=model,
            outcome_events=["OUT"],
            generation_config=GenerationConfig(
                max_length=8,
                max_new_tokens=1,
                num_return_sequences=1,
                do_sample=True,
            ),
            enable_early_stopping=False,
        )

        predictor.simulate(["X"])

        self.assertNotIn("stopping_criteria", model.generation_kwargs[0])

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


class DifferentialTokenizer:
    def __init__(self):
        tokens = [
            "PAD",
            "END",
            "year:2000",
            "age:20",
            "Gender/1",
            "Race/1",
            "VS",
            "D10",
        ] + [f"C{i}" for i in range(8, 24)]
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}
        self.lab_token_ids = []

    def get_vocab(self):
        return self.vocab

    def decode(self, token_ids, **kwargs):
        return [self.inverse_vocab[int(token_id)] for token_id in token_ids]


class TestGenerationDifferentialEquivalence(unittest.TestCase):
    def _build_model(self, apply_rotary=False):
        torch.manual_seed(7)
        config = CEHRGPTConfig(
            vocab_size=24,
            n_positions=32,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_inner=64,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
            include_values=False,
            use_sub_time_tokenization=False,
            include_ttv_prediction=False,
            include_motor_time_to_event=False,
            apply_rotary=apply_rotary,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        return CEHRGPT2LMHeadModel(config).eval()

    def _generate(self, model, tokenizer, share_prefill, track_ages):
        model._generation_share_prompt_prefill = share_prefill
        model._generation_track_ages_override = track_ages
        prompt_tokens = [
            "year:2000",
            "age:20",
            "Gender/1",
            "Race/1",
            "VS",
            "C8",
        ]
        prompt = torch.tensor(
            [[tokenizer.vocab[token] for token in prompt_tokens]], dtype=torch.long
        )
        generation_config = GenerationConfig(
            max_length=32,
            max_new_tokens=5,
            num_return_sequences=4,
            do_sample=True,
            top_p=1.0,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        torch.manual_seed(12345)
        return model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
        )

    def _assert_same_generation(self, expected, actual):
        torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
        self.assertEqual(len(actual.logits), len(expected.logits))
        for expected_logits, actual_logits in zip(expected.logits, actual.logits):
            torch.testing.assert_close(
                actual_logits, expected_logits, rtol=1e-6, atol=1e-7
            )

    def test_non_rotary_optimized_path_matches_legacy_age_and_prefill_path(self):
        tokenizer = DifferentialTokenizer()
        model = self._build_model(apply_rotary=False)

        legacy = self._generate(model, tokenizer, share_prefill=False, track_ages=True)
        optimized = self._generate(model, tokenizer, share_prefill=True, track_ages=False)

        self._assert_same_generation(legacy, optimized)

    def test_rotary_shared_prefill_matches_repeated_prefill(self):
        tokenizer = DifferentialTokenizer()
        model = self._build_model(apply_rotary=True)

        repeated = self._generate(model, tokenizer, share_prefill=False, track_ages=True)
        shared = self._generate(model, tokenizer, share_prefill=True, track_ages=True)

        self._assert_same_generation(repeated, shared)


if __name__ == "__main__":
    unittest.main()
