import datetime
import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch
from transformers import GenerationConfig, PreTrainedModel, StoppingCriteriaList

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token, is_visit_start
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.gpt2 import StaticKeyValueCache
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

    def test_context_window_can_limit_input_below_model_capacity(self):
        tokenizer = SpecialTokenTokenizer()
        predictor = TimeToEventModel(
            tokenizer=tokenizer,
            model=FakeGenerationModel(),
            outcome_events=[],
            generation_config=GenerationConfig(max_length=8, max_new_tokens=1),
            max_input_length=6,
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
        self.assertLessEqual(len(tokenizer.encode(prompt)), 6)


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

    def _generate(
        self,
        model,
        tokenizer,
        share_prefill,
        track_ages,
        last_token_logits_only=True,
    ):
        model._generation_share_prompt_prefill = share_prefill
        model._generation_track_ages_override = track_ages
        model._generation_last_token_logits_only = last_token_logits_only
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

    def test_last_token_projection_matches_full_prompt_logits_generation(self):
        tokenizer = DifferentialTokenizer()
        model = self._build_model(apply_rotary=False)

        full_prompt_logits = self._generate(
            model,
            tokenizer,
            share_prefill=True,
            track_ages=False,
            last_token_logits_only=False,
        )
        last_token_logits = self._generate(
            model,
            tokenizer,
            share_prefill=True,
            track_ages=False,
            last_token_logits_only=True,
        )

        self._assert_same_generation(full_prompt_logits, last_token_logits)

    def test_generation_projects_only_the_last_prompt_hidden_state(self):
        tokenizer = DifferentialTokenizer()
        model = self._build_model(apply_rotary=False)
        projected_sequence_lengths = []

        def record_lm_head_input(_module, args):
            projected_sequence_lengths.append(args[0].shape[1])

        handle = model.lm_head.register_forward_pre_hook(record_lm_head_input)
        try:
            self._generate(
                model,
                tokenizer,
                share_prefill=True,
                track_ages=False,
                last_token_logits_only=True,
            )
        finally:
            handle.remove()

        self.assertTrue(projected_sequence_lengths)
        self.assertEqual(
            projected_sequence_lengths, [1] * len(projected_sequence_lengths)
        )

    def test_regular_forward_still_returns_full_sequence_logits(self):
        model = self._build_model(apply_rotary=False)
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 8]], dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            ages=torch.zeros_like(input_ids, dtype=torch.float32),
            output_motor_features=False,
            return_dict=True,
        )

        self.assertEqual(outputs.logits.shape, (1, input_ids.shape[1], 24))

    def test_shared_prefill_is_disabled_for_flash_attention(self):
        model = self._build_model(apply_rotary=False)
        model.config._attn_implementation = "flash_attention_2"

        self.assertFalse(model._can_share_generation_prompt_prefill())


class EncodingDifferentialTokenizer(DifferentialTokenizer):
    def encode(self, tokens):
        return [self.vocab[token] for token in tokens]


PROMPT = ["year:2000", "age:20", "Gender/1", "Race/1", "VS", "C8", "C9"]


class TimeTokenizer(EncodingDifferentialTokenizer):
    """A second age token, and time tokens of more than a year, which move the ages forward."""

    def __init__(self):
        super().__init__()
        tokens = [self.inverse_vocab[index] for index in range(len(self.vocab))]
        tokens[15] = "age:30"
        tokens[16:24] = [f"D{400 + index}" for index in range(8)]
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}


class ValueTokenizer(EncodingDifferentialTokenizer):
    """Every token is a lab, so every generated token gets a value from the value head."""

    def __init__(self):
        super().__init__()
        self.lab_token_ids = list(range(24))

    def get_value_vocab(self):
        return {f"BIN:{index}": index for index in range(6)}


def build_generation_model(
    apply_rotary=False,
    attn_implementation=None,
    attn_pdrop=0.0,
    backbone="gpt2",
    include_values=False,
):
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
        include_values=include_values,
        value_vocab_size=6 if include_values else None,
        lab_token_ids=list(range(24)) if include_values else None,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=False,
        apply_rotary=apply_rotary,
        backbone=backbone,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=attn_pdrop,
    )
    if attn_implementation is None:
        return CEHRGPT2LMHeadModel(config).eval()
    config._attn_implementation = attn_implementation
    # flash-attn is not installed where the unit tests run, the tests patch its functions
    with mock.patch.object(
        PreTrainedModel,
        "_check_and_enable_flash_attn_2",
        classmethod(lambda cls, config, *args, **kwargs: config),
    ):
        return CEHRGPT2LMHeadModel(config).eval()


def sampling_config(num_return_sequences=3, max_new_tokens=5, **kwargs):
    return GenerationConfig(
        max_length=32,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_p=1.0,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=None,
        return_dict_in_generate=True,
        **kwargs,
    )


def record_ages(model):
    """The ages of every forward pass of the model, and the handle of the hook recording them."""
    ages = []
    handle = model.cehrgpt.register_forward_pre_hook(
        lambda _module, _args, kwargs: ages.append(kwargs.get("ages")),
        with_kwargs=True,
    )
    return ages, handle


class TestStoppingCriteriaReuse(unittest.TestCase):
    def _predictor(self):
        return TimeToEventModel(
            tokenizer=FakeTokenizer(),
            model=FakeGenerationModel(),
            outcome_events=["OUT"],
            generation_config=GenerationConfig(
                max_length=8, max_new_tokens=1, num_return_sequences=1, do_sample=True
            ),
        )

    def test_the_vocabulary_tables_are_built_once_per_task_window(self):
        predictor = self._predictor()
        with mock.patch(
            "cehrgpt.time_to_event.time_to_event_model.TimeToEventStoppingCriteria",
            wraps=TimeToEventStoppingCriteria,
        ) as criteria_class:
            first = predictor._get_stopping_criteria(0, -1, 0, 30)
            second = predictor._get_stopping_criteria(0, -1, 0, 30)
            other_window = predictor._get_stopping_criteria(0, -1, 0, 60)

        self.assertIs(first, second)
        self.assertIsNot(first, other_window)
        self.assertEqual(criteria_class.call_count, 2)

    def test_replaced_outcome_events_or_tokenizer_get_their_own_criteria(self):
        predictor = self._predictor()
        first = predictor._get_stopping_criteria(0, -1, 0, 30)

        predictor.outcome_events = ["X"]
        other_events = predictor._get_stopping_criteria(0, -1, 0, 30)
        self.assertIsNot(other_events, first)
        # X is the outcome now, OUT is not anymore
        self.assertEqual(
            other_events(torch.tensor([[3], [2]]), scores=None).tolist(), [True, False]
        )

        predictor.tokenizer = FakeTokenizer()
        self.assertIsNot(predictor._get_stopping_criteria(0, -1, 0, 30), other_events)

    def test_a_reused_criteria_forgets_the_previous_generation(self):
        predictor = self._predictor()
        criteria = predictor._get_stopping_criteria(0, -1, 0, 15)
        # 20 days, then a token after the 15 day window: finished
        criteria(torch.tensor([[0]]), scores=None)
        criteria(torch.tensor([[0, 0]]), scores=None)
        self.assertEqual(
            criteria(torch.tensor([[0, 0, 3]]), scores=None).tolist(), [True]
        )

        criteria = predictor._get_stopping_criteria(0, -1, 0, 15)

        # Same batch size as before, the time and the finished state start over
        self.assertEqual(criteria(torch.tensor([[3]]), scores=None).tolist(), [False])
        self.assertEqual(
            criteria(torch.tensor([[3, 0]]), scores=None).tolist(), [False]
        )


class TestPromptCacheReuse(unittest.TestCase):
    def _predictor(self, model, reuse_prompt_cache, num_return_sequences=6):
        return TimeToEventModel(
            tokenizer=EncodingDifferentialTokenizer(),
            model=model,
            outcome_events=["C12"],
            generation_config=GenerationConfig(
                max_length=32,
                max_new_tokens=5,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_p=1.0,
                use_cache=True,
                pad_token_id=0,
                eos_token_id=None,
                return_dict_in_generate=True,
            ),
            batch_size=2,
            enable_early_stopping=False,
            reuse_prompt_cache=reuse_prompt_cache,
        )

    def _simulate(self, model, reuse_prompt_cache, prompt=PROMPT):
        """The sequences of a simulation, and the sequence length of every forward pass."""
        predictor = self._predictor(model, reuse_prompt_cache)
        forward_lengths = []
        handle = model.cehrgpt.register_forward_pre_hook(
            lambda _module, args, kwargs: forward_lengths.append(
                tuple((args[0] if args else kwargs["input_ids"]).shape)
            ),
            with_kwargs=True,
        )
        try:
            torch.manual_seed(2024)
            sequences = predictor.simulate(prompt)
        finally:
            handle.remove()
        return predictor, sequences, forward_lengths

    def test_the_prompt_is_encoded_once_for_all_the_batches(self):
        model = build_generation_model()

        _, _, without_reuse = self._simulate(model, reuse_prompt_cache=False)
        _, _, with_reuse = self._simulate(model, reuse_prompt_cache=True)

        prompt_length = len(PROMPT)
        self.assertEqual(
            sum(length == prompt_length for _, length in without_reuse), 3
        )
        # The first batch encodes the prompt, the other two only read new tokens
        self.assertEqual(sum(length == prompt_length for _, length in with_reuse), 1)
        self.assertEqual(
            [shape for shape in with_reuse if shape[1] != prompt_length],
            [(2, 1)] * (len(with_reuse) - 1),
        )
        # The two batches that start from the first one's prompt skip its forward pass
        self.assertEqual(len(with_reuse), len(without_reuse) - 2)

    def test_the_samples_match_the_ones_without_reuse(self):
        for apply_rotary in (False, True):
            with self.subTest(apply_rotary=apply_rotary):
                model = build_generation_model(apply_rotary=apply_rotary)

                _, expected, _ = self._simulate(model, reuse_prompt_cache=False)
                _, actual, _ = self._simulate(model, reuse_prompt_cache=True)

                self.assertEqual(len(actual), 6)
                self.assertEqual(actual, expected)

    def test_generation_from_the_prompt_prefill_is_identical_to_a_full_prefill(self):
        tokenizer = EncodingDifferentialTokenizer()
        prompt = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)
        generation_config = sampling_config(output_logits=True)
        for backbone, apply_rotary in (
            ("gpt2", False),
            ("gpt2", True),
            ("qwen2", False),
        ):
            with self.subTest(backbone=backbone, apply_rotary=apply_rotary):
                model = build_generation_model(
                    apply_rotary=apply_rotary, backbone=backbone
                )

                torch.manual_seed(99)
                expected = model.generate(
                    inputs=prompt,
                    generation_config=generation_config,
                    cehrgpt_tokenizer=tokenizer,
                    return_prompt_prefill=True,
                )
                prompt_prefill = expected.prompt_prefill
                torch.testing.assert_close(
                    prompt_prefill["prompt"]["input_ids"], prompt, rtol=0, atol=0
                )
                self.assertEqual(prompt_prefill["logits"].shape, (1, 1, 24))
                self.assertEqual(
                    prompt_prefill["past_key_values"][0][0].shape[0::2],
                    (1, len(PROMPT)),
                )
                kept = {
                    "logits": prompt_prefill["logits"].clone(),
                    "key": prompt_prefill["past_key_values"][0][0].clone(),
                }
                torch.manual_seed(99)
                actual = model.generate(
                    inputs=prompt,
                    generation_config=generation_config,
                    cehrgpt_tokenizer=tokenizer,
                    prompt_prefill=prompt_prefill,
                )

                # Not close, identical: the first step uses the numbers of the full prefill
                torch.testing.assert_close(
                    actual.sequences, expected.sequences, rtol=0, atol=0
                )
                self.assertEqual(len(actual.logits), len(expected.logits))
                for expected_logits, actual_logits in zip(expected.logits, actual.logits):
                    torch.testing.assert_close(
                        actual_logits, expected_logits, rtol=0, atol=0
                    )
                self.assertIsNone(actual.prompt_prefill)
                # A generation leaves the prefill it starts from untouched, for the next one
                torch.testing.assert_close(
                    prompt_prefill["logits"], kept["logits"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    prompt_prefill["past_key_values"][0][0], kept["key"], rtol=0, atol=0
                )

    def test_the_value_of_the_first_new_token_comes_from_the_prompt_prefill(self):
        tokenizer = ValueTokenizer()
        model = build_generation_model(include_values=True)
        # The value head of a randomly initialized model gives every value about the same
        # probability whatever the hidden state. Scaled up, the values follow the hidden state.
        with torch.no_grad():
            model.value_head.weight.mul_(200)
        prompt = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)
        value_kwargs = {
            "value_indicators": torch.tensor([[False] * 6 + [True]]),
            "values": torch.tensor([[0] * 6 + [4]]),
        }
        generation_config = sampling_config(num_return_sequences=8)

        torch.manual_seed(11)
        expected = model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            return_prompt_prefill=True,
            **value_kwargs,
        )
        self.assertIsNotNone(expected.prompt_prefill["value_hidden_states"])
        torch.manual_seed(11)
        actual = model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            prompt_prefill=expected.prompt_prefill,
            **value_kwargs,
        )

        # Every token is a lab: the first new token has a value, sampled from the value
        # hidden state of the prefill
        self.assertTrue(expected.sequence_val_masks[:, len(PROMPT)].all())
        torch.testing.assert_close(
            actual.sequence_vals, expected.sequence_vals, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)

        # The values of the 8 sequences depend on that hidden state
        broken = dict(expected.prompt_prefill)
        torch.manual_seed(0)
        broken["value_hidden_states"] = torch.randn_like(broken["value_hidden_states"])
        torch.manual_seed(11)
        from_broken = model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            prompt_prefill=broken,
            **value_kwargs,
        )
        self.assertFalse(
            torch.equal(
                from_broken.sequence_vals[:, len(PROMPT)],
                expected.sequence_vals[:, len(PROMPT)],
            )
        )

    def _prompt_prefill(self, model, tokenizer, prompt_tokens=PROMPT, **kwargs):
        return model.generate(
            inputs=torch.tensor([tokenizer.encode(prompt_tokens)], dtype=torch.long),
            generation_config=sampling_config(num_return_sequences=2, max_new_tokens=2),
            cehrgpt_tokenizer=tokenizer,
            return_prompt_prefill=True,
            **kwargs,
        ).prompt_prefill

    def test_a_prompt_prefill_of_another_prompt_is_refused(self):
        tokenizer = EncodingDifferentialTokenizer()
        model = build_generation_model()
        prompt_prefill = self._prompt_prefill(model, tokenizer)
        generation_config = sampling_config(num_return_sequences=2, max_new_tokens=2)

        for other_prompt in (PROMPT[:-1], PROMPT[:-1] + ["C10"]):
            with self.subTest(other_prompt=other_prompt):
                with self.assertRaisesRegex(ValueError, "the input_ids differ"):
                    model.generate(
                        inputs=torch.tensor(
                            [tokenizer.encode(other_prompt)], dtype=torch.long
                        ),
                        generation_config=generation_config,
                        cehrgpt_tokenizer=tokenizer,
                        prompt_prefill=prompt_prefill,
                    )

    def test_a_prompt_prefill_of_other_values_is_refused(self):
        tokenizer = ValueTokenizer()
        model = build_generation_model(include_values=True)
        indicators = torch.tensor([[False] * 6 + [True]])
        prompt_prefill = self._prompt_prefill(
            model,
            tokenizer,
            value_indicators=indicators,
            values=torch.tensor([[0] * 6 + [4]]),
        )

        with self.assertRaisesRegex(ValueError, "the values differ"):
            model.generate(
                inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
                generation_config=sampling_config(
                    num_return_sequences=2, max_new_tokens=2
                ),
                cehrgpt_tokenizer=tokenizer,
                prompt_prefill=prompt_prefill,
                value_indicators=indicators,
                values=torch.tensor([[0] * 6 + [2]]),
            )

    def test_a_prompt_prefill_of_other_ages_is_refused(self):
        tokenizer = TimeTokenizer()
        model = build_generation_model(apply_rotary=True)
        prompt_prefill = self._prompt_prefill(model, tokenizer)
        self.assertEqual(prompt_prefill["prompt"]["ages"].tolist(), [[20] * len(PROMPT)])

        with self.assertRaisesRegex(ValueError, "the ages differ"):
            model.generate(
                inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
                generation_config=sampling_config(
                    num_return_sequences=2, max_new_tokens=2
                ),
                cehrgpt_tokenizer=tokenizer,
                prompt_prefill=prompt_prefill,
                ages=torch.full((1, len(PROMPT)), 30),
            )

    def test_a_prompt_prefill_needs_one_prompt_for_all_the_sequences(self):
        tokenizer = EncodingDifferentialTokenizer()
        model = build_generation_model()
        two_prompts = torch.tensor(
            [tokenizer.encode(PROMPT), tokenizer.encode(PROMPT[:-1] + ["C10"])],
            dtype=torch.long,
        )
        generation_config = sampling_config(num_return_sequences=1, max_new_tokens=2)

        with self.assertRaisesRegex(ValueError, "the same prompt"):
            model.generate(
                inputs=two_prompts,
                generation_config=generation_config,
                cehrgpt_tokenizer=tokenizer,
                return_prompt_prefill=True,
            )
        with self.assertRaisesRegex(ValueError, "the same prompt"):
            model.generate(
                inputs=two_prompts,
                generation_config=generation_config,
                cehrgpt_tokenizer=tokenizer,
                prompt_prefill=self._prompt_prefill(model, tokenizer),
            )

    def test_a_prompt_prefill_cannot_return_the_states_of_the_prompt(self):
        tokenizer = EncodingDifferentialTokenizer()
        model = build_generation_model()

        with self.assertRaisesRegex(ValueError, "no attentions or hidden states"):
            model.generate(
                inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
                generation_config=sampling_config(
                    num_return_sequences=2, max_new_tokens=2, output_hidden_states=True
                ),
                cehrgpt_tokenizer=tokenizer,
                prompt_prefill=self._prompt_prefill(model, tokenizer),
            )

    def test_a_prompt_prefill_is_refused_with_causal_sfm(self):
        tokenizer = EncodingDifferentialTokenizer()
        model = build_generation_model()
        prompt_prefill = self._prompt_prefill(model, tokenizer)
        model.cehrgpt.config.causal_sfm = True

        with self.assertRaisesRegex(ValueError, "causal_sfm"):
            model.generate(
                inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
                generation_config=sampling_config(
                    num_return_sequences=2, max_new_tokens=2
                ),
                cehrgpt_tokenizer=tokenizer,
                prompt_prefill=prompt_prefill,
            )

    def test_the_ages_move_forward_the_same_with_and_without_reuse(self):
        model = build_generation_model(apply_rotary=True)

        def simulate(reuse_prompt_cache):
            predictor = self._predictor(model, reuse_prompt_cache)
            predictor.tokenizer = TimeTokenizer()
            ages, handle = record_ages(model)
            try:
                torch.manual_seed(2024)
                sequences = predictor.simulate(PROMPT)
            finally:
                handle.remove()
            # The ages of the decoding steps, one per sequence
            return sequences, [
                step.flatten().tolist() for step in ages if step.shape[1] == 1
            ]

        expected_sequences, expected_ages = simulate(reuse_prompt_cache=False)
        actual_sequences, actual_ages = simulate(reuse_prompt_cache=True)

        self.assertEqual(actual_sequences, expected_sequences)
        # A third of the vocabulary is a time token of more than a year: ages did move
        self.assertGreater(max(max(step) for step in expected_ages), 20)
        # Without reuse every batch has a prefill and 4 decoding steps, with reuse the
        # batches after the first one have 4 decoding steps and no prefill
        self.assertEqual(actual_ages, expected_ages)

    def test_different_prompts_in_a_batch_keep_their_own_ages(self):
        tokenizer = TimeTokenizer()
        model = build_generation_model(apply_rotary=True)
        ages, handle = record_ages(model)
        try:
            model.generate(
                inputs=torch.tensor(
                    [
                        tokenizer.encode(PROMPT),
                        tokenizer.encode(["year:2000", "age:30"] + PROMPT[2:]),
                    ],
                    dtype=torch.long,
                ),
                generation_config=sampling_config(
                    num_return_sequences=1, max_new_tokens=1
                ),
                cehrgpt_tokenizer=tokenizer,
            )
        finally:
            handle.remove()

        self.assertEqual(
            ages[0].tolist(), [[20] * len(PROMPT), [30] * len(PROMPT)]
        )

    def test_the_time_token_lookup_is_per_tokenizer(self):
        model = build_generation_model(apply_rotary=True)
        tokenizer, time_tokenizer = EncodingDifferentialTokenizer(), TimeTokenizer()
        device = torch.device("cpu")

        lookup = model._get_generation_time_delta_lookup(tokenizer, device)
        self.assertIs(model._get_generation_time_delta_lookup(tokenizer, device), lookup)
        self.assertEqual(lookup.nonzero().flatten().tolist(), [7])

        time_lookup = model._get_generation_time_delta_lookup(time_tokenizer, device)
        self.assertEqual(time_lookup[16:24].tolist(), list(range(400, 408)))

    def test_the_samples_and_values_match_for_a_model_with_values(self):
        class ValueTokenizer(EncodingDifferentialTokenizer):
            def __init__(self):
                super().__init__()
                self.lab_token_ids = [9, 10, 11]

            def get_value_vocab(self):
                return {f"BIN:{index}": index for index in range(6)}

        torch.manual_seed(7)
        model = CEHRGPT2LMHeadModel(
            CEHRGPTConfig(
                vocab_size=24,
                n_positions=32,
                n_embd=32,
                n_layer=2,
                n_head=4,
                n_inner=64,
                bos_token_id=1,
                eos_token_id=None,
                pad_token_id=0,
                include_values=True,
                value_vocab_size=6,
                lab_token_ids=[9, 10, 11],
                use_sub_time_tokenization=False,
                include_ttv_prediction=False,
                include_motor_time_to_event=False,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        ).eval()

        def simulate(reuse_prompt_cache):
            predictor = self._predictor(model, reuse_prompt_cache)
            predictor.tokenizer = ValueTokenizer()
            torch.manual_seed(3)
            # The last token of the prompt is a lab with a value, read from the prompt and
            # not from the cache
            return predictor.simulate(
                PROMPT,
                value_indicators=[False] * 6 + [True],
                values=[0] * 6 + [4],
                return_values=True,
            )

        expected_sequences, expected_values = simulate(reuse_prompt_cache=False)
        actual_sequences, actual_values = simulate(reuse_prompt_cache=True)

        self.assertEqual(actual_sequences, expected_sequences)
        self.assertEqual(actual_values, expected_values)

    def test_the_same_tokens_with_other_values_are_another_prompt(self):
        model = build_generation_model(include_values=True)
        predictor = self._predictor(model, reuse_prompt_cache=True)
        predictor.tokenizer = ValueTokenizer()
        indicators = [False] * 6 + [True]
        predictor.simulate(PROMPT, value_indicators=indicators, values=[0] * 6 + [4])
        forward_lengths = []
        handle = model.cehrgpt.register_forward_pre_hook(
            lambda _module, args, kwargs: forward_lengths.append(
                (args[0] if args else kwargs["input_ids"]).shape[1]
            ),
            with_kwargs=True,
        )
        try:
            predictor.simulate(PROMPT, value_indicators=indicators, values=[0] * 6 + [2])
        finally:
            handle.remove()

        # The prompt is encoded again, once, for the other values
        self.assertEqual(forward_lengths.count(len(PROMPT)), 1)

    def test_the_cache_of_one_patient_is_not_used_for_another(self):
        model = build_generation_model()
        predictor, _, _ = self._simulate(model, reuse_prompt_cache=True)
        first_key = predictor._prompt_cache[0]
        other_prompt = PROMPT[:-1] + ["C10"]

        self.assertIsNone(
            predictor._get_prompt_cache(
                (tuple(predictor.tokenizer.encode(other_prompt)), None, None)
            )
        )
        predictor.simulate(other_prompt)

        self.assertNotEqual(predictor._prompt_cache[0], first_key)

    def test_a_retry_starts_from_the_cache_of_the_first_trial(self):
        model = build_generation_model()
        predictor = self._predictor(model, reuse_prompt_cache=True)
        forward_lengths = []
        handle = model.cehrgpt.register_forward_pre_hook(
            lambda _module, args, kwargs: forward_lengths.append(
                (args[0] if args else kwargs["input_ids"]).shape[1]
            ),
            with_kwargs=True,
        )
        try:
            predictor.simulate(PROMPT)
            forward_lengths.clear()
            predictor.simulate(PROMPT)
        finally:
            handle.remove()

        self.assertEqual(set(forward_lengths), {1})

    def test_models_that_cannot_share_a_prompt_cache_do_not_reuse_it(self):
        model = build_generation_model()
        predictor = self._predictor(model, reuse_prompt_cache=True)
        self.assertTrue(predictor._can_reuse_prompt_cache())
        model.config.causal_sfm = True
        self.assertFalse(predictor._can_reuse_prompt_cache())
        model.config.causal_sfm = False
        model.config.use_local_attention = True
        self.assertFalse(predictor._can_reuse_prompt_cache())
        model.config.use_local_attention = False
        self.assertFalse(
            self._predictor(model, reuse_prompt_cache=False)._can_reuse_prompt_cache()
        )


def fake_flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False):
    """flash_attn_func on (batch, length, heads, head_dim) tensors, computed with SDPA."""
    query, key, value = (
        states.float().transpose(1, 2) for states in (query, key, value)
    )
    attn_mask = None
    if causal:
        query_length, key_length = query.shape[-2], key.shape[-2]
        # Aligned to the bottom right, like flash-attn from 2.1
        attn_mask = torch.ones(query_length, key_length, dtype=torch.bool).tril(
            diagonal=key_length - query_length
        )
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    return output.transpose(1, 2).to(torch.bfloat16)


def unexpected_varlen(*args, **kwargs):
    raise AssertionError("The keys and values were un-padded")


class TestFlashAttentionWithoutPadding(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch(
                "cehrgpt.models.gpt2.flash_attn_func", fake_flash_attn_func, create=True
            ),
            mock.patch(
                "cehrgpt.models.gpt2.flash_attn_varlen_func",
                unexpected_varlen,
                create=True,
            ),
            mock.patch(
                "cehrgpt.models.gpt2.index_first_axis",
                lambda states, indices: states[indices],
                create=True,
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _generate(self, model):
        tokenizer = EncodingDifferentialTokenizer()
        torch.manual_seed(5)
        return model.generate(
            inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
            generation_config=GenerationConfig(
                max_length=32,
                max_new_tokens=4,
                num_return_sequences=3,
                do_sample=True,
                top_p=1.0,
                use_cache=True,
                pad_token_id=0,
                eos_token_id=None,
                return_dict_in_generate=True,
                output_logits=True,
            ),
            cehrgpt_tokenizer=tokenizer,
        )

    def test_generation_from_an_unpadded_prompt_never_unpads(self):
        flash_model = build_generation_model(attn_implementation="flash_attention_2")
        eager_model = build_generation_model()
        eager_model.load_state_dict(flash_model.state_dict())

        # unexpected_varlen fails the generation if a layer takes the un-padding path
        flash = self._generate(flash_model)
        eager = self._generate(eager_model)

        # The flash path computes in bfloat16, so only the first step is comparable: the
        # next ones can follow different samples
        torch.testing.assert_close(
            flash.logits[0].float(), eager.logits[0], rtol=0, atol=5e-2
        )

    def _causal_flags(self, backbone, is_causal=True):
        """(query length, key length, causal) of every flash_attn_func call of a generation."""
        calls = []

        def record(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False):
            calls.append((query.shape[1], key.shape[1], causal))
            return fake_flash_attn_func(query, key, value, causal=causal)

        model = build_generation_model(
            attn_implementation="flash_attention_2", backbone=backbone
        )
        for block in model.cehrgpt.h:
            block.attn.is_causal = is_causal
        with mock.patch(
            f"cehrgpt.models.{backbone}.flash_attn_func", record, create=True
        ), mock.patch(
            f"cehrgpt.models.{backbone}.flash_attn_varlen_func",
            unexpected_varlen,
            create=True,
        ):
            self._generate(model)
        return calls

    def test_the_prefill_is_causal_and_a_single_query_is_not(self):
        prompt_length = len(PROMPT)
        for backbone in ("gpt2", "qwen2"):
            with self.subTest(backbone=backbone):
                calls = self._causal_flags(backbone)

                # 2 layers: the prefill of the 3 sequences, then 3 decoding steps
                self.assertEqual(
                    calls[:2], [(prompt_length, prompt_length, True)] * 2
                )
                self.assertEqual(
                    calls[2:],
                    [
                        (1, prompt_length + step, False)
                        for step in (1, 2, 3)
                        for _layer in range(2)
                    ],
                )

    def test_an_attention_that_is_not_causal_stays_not_causal(self):
        for backbone in ("gpt2", "qwen2"):
            with self.subTest(backbone=backbone):
                calls = self._causal_flags(backbone, is_causal=False)

                self.assertEqual({causal for _, _, causal in calls}, {False})

    def test_the_whole_prompt_matches_eager_attention(self):
        # Every position of the prompt, not only the last one: a missing causal mask changes
        # the logits of all the positions before the last
        flash_model = build_generation_model(attn_implementation="flash_attention_2")
        eager_model = build_generation_model()
        eager_model.load_state_dict(flash_model.state_dict())
        input_ids = torch.tensor([EncodingDifferentialTokenizer().encode(PROMPT)])
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "output_motor_features": False,
            "return_dict": True,
        }

        flash_logits = flash_model(**kwargs).logits.float()
        eager_logits = eager_model(**kwargs).logits

        self.assertEqual(flash_logits.shape, (1, len(PROMPT), 24))
        torch.testing.assert_close(flash_logits, eager_logits, rtol=0, atol=5e-2)
        with mock.patch(
            "cehrgpt.models.gpt2.flash_attn_func",
            lambda query, key, value, *args, **kwargs: fake_flash_attn_func(
                query, key, value, causal=False
            ),
        ):
            not_causal_logits = flash_model(**kwargs).logits.float()
        # The tolerance above is tight enough to tell a missing causal mask
        self.assertGreater(
            (not_causal_logits - eager_logits).abs().max().item(), 5e-2
        )

    def test_the_mask_is_kept_when_dropping_it_is_switched_off(self):
        model = build_generation_model(attn_implementation="flash_attention_2")
        model.cehrgpt._drop_unpadded_flash_attention_mask = False
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 8]])

        with self.assertRaisesRegex(AssertionError, "un-padded"):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                output_motor_features=False,
                return_dict=True,
            )

    def test_a_padded_batch_keeps_its_mask(self):
        model = build_generation_model(attn_implementation="flash_attention_2")
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 8], [2, 3, 4, 5, 0, 0]])

        with self.assertRaisesRegex(AssertionError, "un-padded"):
            model(
                input_ids=input_ids,
                attention_mask=input_ids != 0,
                output_motor_features=False,
                return_dict=True,
            )

    def test_training_keeps_the_mask(self):
        model = build_generation_model(attn_implementation="flash_attention_2").train()
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 8]])

        with self.assertRaisesRegex(AssertionError, "un-padded"):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                output_motor_features=False,
                return_dict=True,
            )


class TestFlashAttentionDropout(unittest.TestCase):
    def _dropout_rates(self, training):
        """The dropout rates the flash-attn functions are called with by a forward pass."""
        rates = []

        def record_flash(query, key, value, dropout_p=0.0, **kwargs):
            rates.append(dropout_p)
            return fake_flash_attn_func(query, key, value, **kwargs)

        def record_varlen(query, key, value, dropout_p=0.0, **kwargs):
            rates.append(dropout_p)
            # One unpadded sequence, the un-padded states are (length, heads, head_dim)
            return fake_flash_attn_func(
                query[None], key[None], value[None], causal=kwargs["causal"]
            )[0]

        model = build_generation_model(
            attn_implementation="flash_attention_2", attn_pdrop=0.1
        ).train(training)
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 8]])
        with mock.patch(
            "cehrgpt.models.gpt2.flash_attn_func", record_flash, create=True
        ), mock.patch(
            "cehrgpt.models.gpt2.flash_attn_varlen_func", record_varlen, create=True
        ), mock.patch(
            "cehrgpt.models.gpt2.index_first_axis",
            lambda states, indices: states[indices],
            create=True,
        ), mock.patch(
            "cehrgpt.models.gpt2.pad_input",
            lambda states, indices, batch_size, length: states[None],
            create=True,
        ):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                output_motor_features=False,
                return_dict=True,
            )
        return rates

    def test_attention_dropout_is_off_outside_training(self):
        self.assertEqual(self._dropout_rates(training=False), [0.0, 0.0])

    def test_attention_dropout_is_kept_in_training(self):
        self.assertEqual(self._dropout_rates(training=True), [0.1, 0.1])


def record_batch_sizes(model):
    """The batch size of every forward pass of the model, and the handle of the hook."""
    sizes = []
    handle = model.cehrgpt.register_forward_pre_hook(
        lambda _module, args, kwargs: sizes.append(
            (args[0] if args else kwargs["input_ids"]).shape[0]
        ),
        with_kwargs=True,
    )
    return sizes, handle


class StepRecorder:
    """
    What a generation computes at every decoding step, to compare two generations exactly.

    The sampled tokens of a small randomly initialized model hardly depend on what the model
    reads, the logits do: the logits of every row the model ran on are recorded, with the
    rows of the batch they are (None for all of them), and the logits of the values.
    """

    def __init__(self, model):
        self.model = model
        self.logits, self.value_logits, self.rows = [], [], []

    def __enter__(self):
        self._handles = [
            self.model.lm_head.register_forward_hook(
                lambda _module, _args, output: self.logits.append(output[:, -1].clone())
            )
        ]
        if getattr(self.model, "value_head", None) is not None:
            self._handles.append(
                self.model.value_head.register_forward_hook(
                    lambda _module, _args, output: self.value_logits.append(
                        output.clone()
                    )
                )
            )
        scatter = CEHRGPT2LMHeadModel._scatter_active_rows

        def record_rows(probabilities, active_rows, batch_size):
            # Called for the tokens, then for the values when the model has values
            if probabilities.shape[-1] == self.model.config.vocab_size:
                self.rows.append(active_rows)
            return scatter(probabilities, active_rows, batch_size)

        self._patch = mock.patch.object(
            CEHRGPT2LMHeadModel, "_scatter_active_rows", staticmethod(record_rows)
        )
        self._patch.start()
        return self

    def __exit__(self, *args):
        self._patch.stop()
        for handle in self._handles:
            handle.remove()

    def assert_same_steps(self, test, whole_batch):
        """Every step of this generation has the logits of its rows in whole_batch."""
        test.assertEqual(len(self.logits), len(whole_batch.logits))
        test.assertEqual(len(self.rows), len(self.logits))
        for name in ("logits", "value_logits"):
            actual_steps, expected_steps = getattr(self, name), getattr(whole_batch, name)
            test.assertEqual(len(actual_steps), len(expected_steps))
            for step, (actual, expected) in enumerate(zip(actual_steps, expected_steps)):
                rows = self.rows[step]
                if rows is not None:
                    expected = expected.index_select(0, rows)
                elif expected.shape[0] == 1:
                    # The prefill of a shared prompt has one row for all the sequences
                    expected = expected.expand(actual.shape[0], -1)
                elif actual.shape[0] == 1:
                    actual = actual.expand(expected.shape[0], -1)
                # Identical on the same rows. A batch of fewer rows can differ in the last
                # bit of a float32 (1e-7 of the value), far below what reading the cache of
                # another row or the wrong token changes (1e-2).
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0 if rows is None else 1e-5,
                    atol=0 if rows is None else 1e-5,
                    msg=f"{name} of step {step}",
                )


class TestStaticKeyValueCache(unittest.TestCase):
    def _cache(self, capacity=6, batch_size=2):
        torch.manual_seed(0)
        key = torch.randn(batch_size, 3, 4, 5)
        value = torch.randn(batch_size, 3, 4, 5)
        return (
            StaticKeyValueCache.from_key_value(key, value, capacity, torch.float32),
            key,
            value,
        )

    def test_reads_like_the_key_and_value_it_was_built_from(self):
        cache, key, value = self._cache()

        self.assertEqual(len(cache), 2)
        cached_key, cached_value = cache
        torch.testing.assert_close(cached_key, key, rtol=0, atol=0)
        torch.testing.assert_close(cached_value, value, rtol=0, atol=0)
        torch.testing.assert_close(cache[0], key, rtol=0, atol=0)
        self.assertEqual(cache.states()[0].shape, (2, 4, 3, 5))

    def test_append_writes_in_place_after_the_tokens_it_has(self):
        cache, key, value = self._cache()
        buffer = cache.keys.data_ptr()
        new_key, new_value = torch.randn(2, 3, 2, 5), torch.randn(2, 3, 2, 5)

        cache.append(new_key, new_value)

        self.assertEqual(cache.length, 6)
        self.assertEqual(cache.keys.data_ptr(), buffer)
        torch.testing.assert_close(
            cache[0], torch.cat([key, new_key], dim=-2), rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache[1], torch.cat([value, new_value], dim=-2), rtol=0, atol=0
        )

    def test_append_past_the_capacity_is_refused(self):
        cache, _, _ = self._cache(capacity=5)

        with self.assertRaisesRegex(ValueError, "holds 5 tokens"):
            cache.append(torch.randn(2, 3, 2, 5), torch.randn(2, 3, 2, 5))
        with self.assertRaisesRegex(ValueError, "cannot hold"):
            StaticKeyValueCache.from_key_value(
                torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5), 3, torch.float32
            )

    def test_select_rows_keeps_the_rows_and_leaves_the_cache_alone(self):
        cache, key, value = self._cache()

        selected = cache.select_rows(torch.tensor([1]))
        selected.append(torch.zeros(1, 3, 1, 5), torch.zeros(1, 3, 1, 5))

        torch.testing.assert_close(selected[0][:, :, :4], key[1:], rtol=0, atol=0)
        torch.testing.assert_close(selected[1][:, :, :4], value[1:], rtol=0, atol=0)
        self.assertEqual(selected.length, 5)
        self.assertEqual(cache.length, 4)
        torch.testing.assert_close(cache[0], key, rtol=0, atol=0)

    def test_select_rows_keeps_the_order_of_the_rows(self):
        cache, key, value = self._cache(batch_size=3)

        selected = cache.select_rows(torch.tensor([0, 2]))

        torch.testing.assert_close(selected[0], key[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(selected[1], value[[0, 2]], rtol=0, atol=0)


class TestDropFinishedSequences(unittest.TestCase):
    """Rows finish at different steps: a time token of more than a year ends the window."""

    def _generate(
        self,
        model,
        eos_token_id=1,
        tokenizer=None,
        generation_config_changes=None,
        **kwargs,
    ):
        tokenizer = tokenizer or TimeTokenizer()
        generation_config = sampling_config(num_return_sequences=8, max_new_tokens=20)
        generation_config.eos_token_id = eos_token_id
        for name, value in (generation_config_changes or {}).items():
            setattr(generation_config, name, value)
        sizes, handle = record_batch_sizes(model)
        try:
            with StepRecorder(model) as self.steps:
                torch.manual_seed(5)
                outputs = model.generate(
                    inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
                    generation_config=generation_config,
                    cehrgpt_tokenizer=tokenizer,
                    stopping_criteria=StoppingCriteriaList(
                        [TimeToEventStoppingCriteria(tokenizer, ["C12"], 0, -1, 0, 500)]
                    ),
                    **kwargs,
                )
        finally:
            handle.remove()
        return outputs, sizes

    def test_the_sequences_are_the_ones_generated_with_the_whole_batch(self):
        for apply_rotary, changes in (
            (False, None),
            (True, None),
            # A processor that reads the tokens generated so far
            (False, {"repetition_penalty": 1.3}),
        ):
            with self.subTest(apply_rotary=apply_rotary, changes=changes):
                model = build_generation_model(apply_rotary=apply_rotary)

                expected, expected_sizes = self._generate(
                    model, generation_config_changes=changes
                )
                expected_steps = self.steps
                actual, actual_sizes = self._generate(
                    model,
                    generation_config_changes=changes,
                    drop_finished_sequences=True,
                )
                self.steps.assert_same_steps(self, expected_steps)

                torch.testing.assert_close(
                    actual.sequences, expected.sequences, rtol=0, atol=0
                )
                # The prefill of the shared prompt, then the whole batch at every step
                self.assertEqual(set(expected_sizes[1:]), {8})
                # The model runs on fewer and fewer rows
                self.assertEqual(actual_sizes[:2], [1, 8])
                self.assertEqual(
                    actual_sizes[1:], sorted(actual_sizes[1:], reverse=True)
                )
                self.assertLess(actual_sizes[-1], 4)
                self.assertGreater(len(set(actual_sizes[1:])), 2)
                self.assertIsNone(actual.past_key_values)
                self.assertIsNotNone(expected.past_key_values)

    def test_the_values_are_the_ones_generated_with_the_whole_batch(self):
        tokenizer = ValueTokenizer()
        model = build_generation_model(include_values=True)
        with torch.no_grad():
            model.value_head.weight.mul_(200)
        # C12 is an outcome: rows finish when they sample it
        expected, _ = self._generate(model, tokenizer=tokenizer)
        expected_steps = self.steps
        actual, sizes = self._generate(
            model, tokenizer=tokenizer, drop_finished_sequences=True
        )

        self.assertLess(sizes[-1], 8)
        # The value logits of a row are the ones of the token that row sampled
        self.assertEqual(len(self.steps.value_logits), len(self.steps.logits))
        self.steps.assert_same_steps(self, expected_steps)
        torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
        # A finished row is padding, whose values are not read
        generated = actual.sequences != 0
        torch.testing.assert_close(
            actual.sequence_vals[generated],
            expected.sequence_vals[generated],
            rtol=0,
            atol=0,
        )
        self.assertTrue((actual.sequence_vals[generated] != 0).any())

    def test_without_an_end_token_a_finished_sequence_is_padded(self):
        model = build_generation_model()
        # Not the token 0, which a dropped row would get anyway
        changes = {"pad_token_id": 1}

        expected, _ = self._generate(
            model, eos_token_id=None, generation_config_changes=changes
        )
        expected_steps = self.steps
        actual, sizes = self._generate(
            model,
            eos_token_id=None,
            generation_config_changes=changes,
            drop_finished_sequences=True,
        )

        self.assertLess(sizes[-1], 8)
        self.steps.assert_same_steps(self, expected_steps)
        # Without it a finished sequence keeps being generated
        self.assertFalse(torch.equal(actual.sequences, expected.sequences))
        generated = torch.zeros_like(actual.sequences, dtype=torch.bool)
        generated[:, : len(PROMPT)] = True
        for step, rows in enumerate(self.steps.rows):
            generated[:, len(PROMPT) + step] = True
            if rows is not None:
                generated[:, len(PROMPT) + step] = False
                generated[rows, len(PROMPT) + step] = True
        torch.testing.assert_close(
            actual.sequences[generated], expected.sequences[generated], rtol=0, atol=0
        )
        self.assertTrue((~generated).any())
        self.assertEqual(set(actual.sequences[~generated].tolist()), {1})

    def test_greedy_decoding_matches_the_whole_batch(self):
        model = build_generation_model()
        tokenizer = TimeTokenizer()
        prompts = torch.tensor(
            [
                tokenizer.encode(PROMPT),
                tokenizer.encode(PROMPT[:-1] + ["D400"]),
                tokenizer.encode(PROMPT[:-1] + ["C10"]),
            ],
            dtype=torch.long,
        )

        def generate(**kwargs):
            sizes, handle = record_batch_sizes(model)
            try:
                outputs = model.generate(
                    inputs=prompts,
                    generation_config=GenerationConfig(
                        max_length=32,
                        max_new_tokens=6,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=0,
                        eos_token_id=1,
                        return_dict_in_generate=True,
                    ),
                    cehrgpt_tokenizer=tokenizer,
                    stopping_criteria=StoppingCriteriaList(
                        [TimeToEventStoppingCriteria(tokenizer, ["C12"], 0, -1, 0, 300)]
                    ),
                    **kwargs,
                )
            finally:
                handle.remove()
            return outputs, sizes

        expected, _ = generate()
        actual, sizes = generate(drop_finished_sequences=True)

        # The second prompt is past the window after one more token
        self.assertEqual(sizes[:3], [3, 3, 2])
        torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)

    def test_it_cannot_return_per_step_outputs_or_run_without_padding(self):
        model = build_generation_model()
        tokenizer = TimeTokenizer()
        prompt = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "drop_finished_sequences has no"):
            model.generate(
                inputs=prompt,
                generation_config=sampling_config(output_logits=True),
                cehrgpt_tokenizer=tokenizer,
                drop_finished_sequences=True,
            )


class TestStaticCacheGeneration(unittest.TestCase):
    def setUp(self):
        self.key_buffers = []

        self.flags = []

        def record(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False):
            if query.shape[1] == 1:
                self.key_buffers.append((key.data_ptr(), key.shape[1], key.is_contiguous()))
                self.flags.append((dropout_p, causal))
            return fake_flash_attn_func(query, key, value, causal=causal)

        patches = [
            mock.patch("cehrgpt.models.gpt2.flash_attn_func", record, create=True),
            mock.patch(
                "cehrgpt.models.gpt2.flash_attn_varlen_func",
                unexpected_varlen,
                create=True,
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _generate(self, model, num_return_sequences=4, max_new_tokens=12, **kwargs):
        tokenizer = TimeTokenizer()
        self.key_buffers.clear()
        self.flags.clear()
        torch.manual_seed(3)
        return model.generate(
            inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
            generation_config=sampling_config(
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                output_logits=True,
            ),
            cehrgpt_tokenizer=tokenizer,
            **kwargs,
        )

    def test_the_logits_are_identical_to_the_ones_of_a_growing_cache(self):
        for apply_rotary in (False, True):
            with self.subTest(apply_rotary=apply_rotary):
                model = build_generation_model(
                    apply_rotary=apply_rotary, attn_implementation="flash_attention_2"
                )

                expected = self._generate(model)
                actual = self._generate(model, static_kv_cache=True)

                torch.testing.assert_close(
                    actual.sequences, expected.sequences, rtol=0, atol=0
                )
                self.assertEqual(len(actual.logits), 12)
                for expected_logits, actual_logits in zip(expected.logits, actual.logits):
                    torch.testing.assert_close(
                        actual_logits, expected_logits, rtol=0, atol=0
                    )
                self.assertIsNone(actual.past_key_values)

    def test_the_keys_stay_where_they_are(self):
        model = build_generation_model(attn_implementation="flash_attention_2")

        self._generate(model)
        growing = list(self.key_buffers)
        self._generate(model, static_kv_cache=True)
        static = list(self.key_buffers)

        # 11 decoding steps of 2 layers, over the prompt and the tokens generated so far
        lengths = [len(PROMPT) + step for step in range(1, 12) for _layer in range(2)]
        self.assertEqual([length for _, length, _ in growing], lengths)
        self.assertEqual([length for _, length, _ in static], lengths)
        # One buffer per layer for the whole generation, read where it is: not contiguous,
        # the memory of the tokens to come is between the rows
        self.assertEqual(len({pointer for pointer, _, _ in static}), 2)
        self.assertFalse(any(contiguous for _, _, contiguous in static[:-2]))
        # A growing cache is copied to a new place at every step
        self.assertTrue(all(contiguous for _, _, contiguous in growing))

    def test_a_single_query_is_not_causal_and_has_no_dropout_in_evaluation(self):
        model = build_generation_model(
            attn_implementation="flash_attention_2", attn_pdrop=0.1
        )

        self._generate(model, static_kv_cache=True)

        self.assertEqual(len({pointer for pointer, _, _ in self.key_buffers}), 2)
        self.assertEqual(set(self.flags), {(0.0, False)})

    def test_an_attention_that_is_not_causal_stays_not_causal(self):
        model = build_generation_model(attn_implementation="flash_attention_2")
        calls = []

        def record(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False):
            calls.append((query.shape[1], causal))
            return fake_flash_attn_func(query, key, value, causal=causal)

        with mock.patch("cehrgpt.models.gpt2.flash_attn_func", record):
            # More than one query at a time over a static cache: the prompt in two parts
            cache = [None, None]
            for block_index, block in enumerate(model.cehrgpt.h):
                hidden = torch.randn(1, 3, 32)
                _, cache[block_index] = block.attn(hidden, use_cache=True)[:2]
                key, value = cache[block_index]
                static = StaticKeyValueCache.from_key_value(key, value, 8, torch.bfloat16)
                block.attn(torch.randn(1, 2, 32), layer_past=static, use_cache=True)
                block.attn.is_causal = False
                block.attn(torch.randn(1, 2, 32), layer_past=static, use_cache=True)
                self.assertEqual(static.length, 7)

        # (queries, causal) of the three calls of every layer
        self.assertEqual(calls, [(3, True), (2, True), (2, False)] * 2)

    def test_it_works_from_a_prompt_prefill_and_with_dropped_sequences(self):
        model = build_generation_model(attn_implementation="flash_attention_2")
        tokenizer = TimeTokenizer()
        prompt = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)

        def generate(**kwargs):
            generation_config = sampling_config(
                num_return_sequences=8, max_new_tokens=20
            )
            generation_config.eos_token_id = 1
            sizes, handle = record_batch_sizes(model)
            try:
                torch.manual_seed(5)
                outputs = model.generate(
                    inputs=prompt,
                    generation_config=generation_config,
                    cehrgpt_tokenizer=tokenizer,
                    stopping_criteria=StoppingCriteriaList(
                        [TimeToEventStoppingCriteria(tokenizer, ["C12"], 0, -1, 0, 500)]
                    ),
                    **kwargs,
                )
            finally:
                handle.remove()
            return outputs, sizes

        with StepRecorder(model) as expected_steps:
            expected, _ = generate(return_prompt_prefill=True)
        with StepRecorder(model) as steps:
            actual, sizes = generate(
                prompt_prefill=expected.prompt_prefill,
                static_kv_cache=True,
                drop_finished_sequences=True,
            )

        # No forward pass over the prompt: the first step is the one of the prefill
        steps.logits.insert(0, expected.prompt_prefill["logits"][:, -1])
        steps.assert_same_steps(self, expected_steps)
        torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
        self.assertLess(sizes[-1], 8)
        # The buffers of the rows that are left are new ones every time rows are dropped
        self.assertGreater(len({pointer for pointer, _, _ in self.key_buffers}), 2)

    def test_it_is_not_used_where_the_attention_cannot_append_in_place(self):
        tokenizer = TimeTokenizer()
        prompt = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)
        kwargs = {"use_cache": True, "attention_mask": torch.ones_like(prompt).bool()}
        flash_model = build_generation_model(attn_implementation="flash_attention_2")
        self.assertTrue(flash_model._can_use_static_kv_cache(kwargs, False))

        self.assertFalse(flash_model._can_use_static_kv_cache(kwargs, True))
        self.assertFalse(
            flash_model._can_use_static_kv_cache({**kwargs, "use_cache": False}, False)
        )
        padded = {**kwargs, "attention_mask": torch.tensor([[False] + [True] * 6])}
        self.assertFalse(flash_model._can_use_static_kv_cache(padded, False))
        for name in ("use_local_attention", "causal_sfm"):
            setattr(flash_model.cehrgpt.config, name, True)
            self.assertFalse(flash_model._can_use_static_kv_cache(kwargs, False), name)
            setattr(flash_model.cehrgpt.config, name, False)
        flash_model.cehrgpt._drop_unpadded_flash_attention_mask = False
        self.assertFalse(flash_model._can_use_static_kv_cache(kwargs, False))
        flash_model.cehrgpt._drop_unpadded_flash_attention_mask = True
        self.assertFalse(flash_model.train()._can_use_static_kv_cache(kwargs, False))

        with mock.patch(
            "cehrgpt.models.qwen2.flash_attn_func", fake_flash_attn_func, create=True
        ):
            qwen2_flash_model = build_generation_model(
                backbone="qwen2", attn_implementation="flash_attention_2"
            )
            self.assertFalse(qwen2_flash_model._can_use_static_kv_cache(kwargs, False))

        # Eager attention and the qwen2 backbone grow their cache, which is returned
        for backbone in ("gpt2", "qwen2"):
            model = build_generation_model(backbone=backbone)
            self.assertFalse(model._can_use_static_kv_cache(kwargs, False))
            outputs = model.generate(
                inputs=prompt,
                generation_config=sampling_config(max_new_tokens=2),
                cehrgpt_tokenizer=tokenizer,
                static_kv_cache=True,
            )
            self.assertIsInstance(outputs.past_key_values[0][0], torch.Tensor)


class TestFastDecodingPrediction(unittest.TestCase):
    def _predictor(self, model, fast_decoding, tokenizer=None):
        generation_config = sampling_config(num_return_sequences=12, max_new_tokens=20)
        generation_config.eos_token_id = 1
        return TimeToEventModel(
            tokenizer=tokenizer or TimeTokenizer(),
            model=model,
            outcome_events=["C12"],
            generation_config=generation_config,
            batch_size=5,
            fast_decoding=fast_decoding,
        )

    def test_the_trajectories_and_the_prediction_are_the_same(self):
        for backbone, attn_implementation in (
            ("gpt2", None),
            ("gpt2", "flash_attention_2"),
            ("qwen2", None),
        ):
            with self.subTest(
                backbone=backbone, attn_implementation=attn_implementation
            ), mock.patch(
                "cehrgpt.models.gpt2.flash_attn_func", fake_flash_attn_func, create=True
            ), mock.patch(
                "cehrgpt.models.gpt2.flash_attn_varlen_func",
                unexpected_varlen,
                create=True,
            ):
                model = build_generation_model(
                    backbone=backbone, attn_implementation=attn_implementation
                )
                results = {}
                for fast_decoding in (False, True):
                    predictor = self._predictor(model, fast_decoding)
                    sizes, handle = record_batch_sizes(model)
                    try:
                        torch.manual_seed(8)
                        results[fast_decoding] = (
                            predictor.simulate(PROMPT, prediction_window_end=500),
                            list(sizes),
                        )
                        torch.manual_seed(8)
                        results[fast_decoding] += (
                            predictor.predict_time_to_events(
                                PROMPT, prediction_window_end=500
                            ),
                        )
                    finally:
                        handle.remove()

                slow_sequences, slow_sizes, slow_prediction = results[False]
                fast_sequences, fast_sizes, fast_prediction = results[True]
                self.assertEqual(fast_sequences, slow_sequences)
                self.assertEqual(len(fast_sequences), 12)
                self.assertEqual(
                    fast_prediction.outcome_events, slow_prediction.outcome_events
                )
                self.assertEqual(
                    fast_prediction.time_intervals, slow_prediction.time_intervals
                )
                # The same forward passes, on fewer rows
                self.assertEqual(len(fast_sizes), len(slow_sizes))
                self.assertLess(sum(fast_sizes), sum(slow_sizes))

    def test_the_values_of_the_trajectories_are_the_same(self):
        model = build_generation_model(include_values=True)
        with torch.no_grad():
            model.value_head.weight.mul_(200)
        results = []
        for fast_decoding in (False, True):
            predictor = self._predictor(model, fast_decoding, ValueTokenizer())
            torch.manual_seed(8)
            results.append(
                predictor.simulate(
                    PROMPT,
                    prediction_window_end=500,
                    value_indicators=[False] * 6 + [True],
                    values=[0] * 6 + [4],
                    return_values=True,
                )
            )

        (slow_sequences, slow_values), (fast_sequences, fast_values) = results
        self.assertEqual(fast_sequences, slow_sequences)

        # The padding of a finished trajectory has no value to read. The tokenizer of this
        # test decodes it, the real one leaves it out.
        def generated(sequences, values):
            return [
                [value for token, value in zip(tokens, token_values) if token != "PAD"]
                for tokens, token_values in zip(sequences, values)
            ]

        self.assertEqual(
            generated(fast_sequences, fast_values), generated(slow_sequences, slow_values)
        )
        self.assertTrue(any("PAD" in tokens for tokens in fast_sequences))

    def test_the_generation_gets_the_flags(self):
        model = FakeGenerationModel()
        for fast_decoding, pad_token_id, expected in (
            (True, 4, {"static_kv_cache", "drop_finished_sequences"}),
            (True, None, {"static_kv_cache"}),
            (False, 4, set()),
        ):
            model.generation_kwargs.clear()
            predictor = TimeToEventModel(
                tokenizer=FakeTokenizer(),
                model=model,
                outcome_events=["OUT"],
                generation_config=GenerationConfig(
                    max_length=8,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                ),
                fast_decoding=fast_decoding,
            )

            predictor.simulate(["X"])

            self.assertEqual(
                {"static_kv_cache", "drop_finished_sequences"}
                & set(model.generation_kwargs[0]),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
