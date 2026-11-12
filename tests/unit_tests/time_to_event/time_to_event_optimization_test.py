import datetime
import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch
from transformers import GenerationConfig, PreTrainedModel

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


def build_generation_model(
    apply_rotary=False, attn_implementation=None, attn_pdrop=0.0
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
        include_values=False,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=False,
        apply_rotary=apply_rotary,
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
        generation_config = GenerationConfig(
            max_length=32,
            max_new_tokens=5,
            num_return_sequences=3,
            do_sample=True,
            top_p=1.0,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        for apply_rotary in (False, True):
            with self.subTest(apply_rotary=apply_rotary):
                model = build_generation_model(apply_rotary=apply_rotary)

                torch.manual_seed(99)
                expected = model.generate(
                    inputs=prompt,
                    generation_config=generation_config,
                    cehrgpt_tokenizer=tokenizer,
                    return_prompt_prefill=True,
                )
                prompt_prefill = expected.prompt_prefill
                self.assertEqual(prompt_prefill["prompt_length"], len(PROMPT))
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

    def test_a_prompt_prefill_of_another_prompt_length_is_refused(self):
        tokenizer = EncodingDifferentialTokenizer()
        model = build_generation_model()
        generation_config = GenerationConfig(
            max_length=32,
            max_new_tokens=2,
            num_return_sequences=2,
            do_sample=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
        )
        prompt_prefill = model.generate(
            inputs=torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long),
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            return_prompt_prefill=True,
        ).prompt_prefill

        with self.assertRaisesRegex(ValueError, "prompt_prefill is of a prompt"):
            model.generate(
                inputs=torch.tensor([tokenizer.encode(PROMPT[:-1])], dtype=torch.long),
                generation_config=generation_config,
                cehrgpt_tokenizer=tokenizer,
                prompt_prefill=prompt_prefill,
            )

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


if __name__ == "__main__":
    unittest.main()
