import datetime
import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import GenerationConfig

from cehrgpt.gpt_utils import (
    truncate_history_keeping_demographics,
    truncate_history_keeping_demographics_with_indices,
)
from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel
from cehrgpt.time_to_event.time_to_event_prediction import (
    encode_prompt_values,
    flush_to_disk_if_full,
)
from cehrgpt.tokenization.tokenization_constants import NONE_BIN, UNKNOWN_BIN

TOKENS = ["year:2000", "age:20", "Gender/1", "Race/1", "[VS]", "LAB", "X", "[VE]", "D7", "[PAD]"]
VALUES = ["BIN:NONE", "BIN:1", "BIN:2", "BIN:unknown", "[OOV]", "positive", "very high"]


class Tokenizer:
    """Word level, like CehrGptTokenizer: one id per token, [PAD] is a special token."""

    def __init__(self):
        self.vocab = {token: index for index, token in enumerate(TOKENS)}
        self.value_vocab = {value: index for index, value in enumerate(VALUES)}
        self.lab_token_ids = [self.vocab["LAB"]]

    def get_vocab(self):
        return self.vocab

    def get_value_vocab(self):
        return self.value_vocab

    def encode(self, tokens):
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids, skip_special_tokens=True):
        inverse = {index: token for token, index in self.vocab.items()}
        tokens = [inverse[int(token_id)] for token_id in token_ids]
        return [t for t in tokens if t != "[PAD]"] if skip_special_tokens else tokens

    def decode_value(self, value_ids, skip_special_tokens=True):
        inverse = {index: value for value, index in self.value_vocab.items()}
        return [inverse[int(value_id)] for value_id in value_ids]

    def normalize(self, concept_id, unit, value):
        return "BIN:1" if value < 5 else "BIN:2"


class ValueModel:
    """Records the generate kwargs; continues every prompt with LAB (value BIN:2), X, [PAD]."""

    def __init__(self, include_values=True, n_positions=64):
        self.config = SimpleNamespace(n_positions=n_positions, include_values=include_values)
        self.calls = []

    def eval(self):
        return self

    def generate(self, inputs, generation_config, **kwargs):
        self.calls.append(kwargs)
        count = generation_config.num_return_sequences
        continuation = torch.tensor([[5, 6, 9]]).expand(count, -1)
        sequences = torch.cat([inputs.expand(count, -1), continuation], dim=1)
        if not self.config.include_values:
            return SimpleNamespace(sequences=sequences)
        prompt_values = kwargs.get("values", torch.zeros_like(inputs))
        sequence_vals = torch.cat(
            [prompt_values.expand(count, -1), torch.tensor([[2, 0, 0]]).expand(count, -1)], dim=1
        )
        return SimpleNamespace(sequences=sequences, sequence_vals=sequence_vals)


def history(num_visits):
    tokens = ["year:2000", "age:20", "Gender/1", "Race/1"]
    for i in range(num_visits):
        if i:
            tokens.append("D7")
        tokens += ["[VS]", "LAB", "X", "[VE]"]
    return tokens


def predictor(model, max_length=64, max_new_tokens=3, store=False):
    return TimeToEventModel(
        tokenizer=Tokenizer(),
        model=model,
        outcome_events=["X"],
        generation_config=GenerationConfig(
            max_length=max_length, max_new_tokens=max_new_tokens, num_return_sequences=2, do_sample=True
        ),
        enable_early_stopping=False,
        store_generated_trajectories=store,
    )


class TestTruncationIndices(unittest.TestCase):
    def test_indices_point_at_the_kept_tokens(self):
        tokens = history(4)
        for max_tokens in [3, 6, 9, 30]:
            kept, indices = truncate_history_keeping_demographics_with_indices(tokens, max_tokens)
            self.assertEqual(kept, truncate_history_keeping_demographics(tokens, max_tokens))
            self.assertEqual(len(indices), len(kept))
            # the header comes from the header, every other token is the token at its index
            self.assertEqual(indices[:4], [0, 1, 2, 3])
            self.assertEqual(kept[4:], [tokens[i] for i in indices[4:]])

    def test_prepare_prompt_indices(self):
        tokens = history(4)
        model = predictor(ValueModel(), max_length=15, max_new_tokens=3)
        prompt, indices = model.prepare_prompt_with_indices(tokens)
        self.assertEqual(prompt, model.prepare_prompt(tokens))
        self.assertLessEqual(len(prompt), 12)
        self.assertEqual(prompt, [tokens[i] for i in indices])


class TestSimulateWithValues(unittest.TestCase):
    def test_prompt_values_follow_the_truncated_prompt(self):
        tokens = history(4)
        # the value of the lab of visit i is i + 1 (BIN:1 or BIN:2 ...), so the kept ones show
        values = [0] * len(tokens)
        indicators = [False] * len(tokens)
        for visit, index in enumerate(i for i, t in enumerate(tokens) if t == "LAB"):
            values[index], indicators[index] = 1 + visit % 2, True
        model = ValueModel()
        tte = predictor(model, max_length=15, max_new_tokens=3)
        prompt, indices = tte.prepare_prompt_with_indices(tokens)
        tte.simulate(tokens, value_indicators=indicators, values=values)
        kwargs = model.calls[0]
        self.assertEqual(kwargs["values"].tolist(), [[values[i] for i in indices]])
        self.assertEqual(kwargs["value_indicators"].tolist(), [[indicators[i] for i in indices]])
        self.assertEqual(kwargs["values"].shape[1], len(prompt))

    def test_generated_values_are_one_per_generated_token(self):
        model = ValueModel()
        tokens = history(1)
        sequences, values = predictor(model).simulate(
            tokens, value_indicators=[False] * len(tokens), values=[0] * len(tokens), return_values=True
        )
        # [PAD] is dropped from the tokens, and its value with it
        self.assertEqual(sequences, [["LAB", "X"]] * 2)
        self.assertEqual(values, [["BIN:2", "BIN:NONE"]] * 2)

    def test_a_model_without_values_gets_no_values(self):
        model = ValueModel(include_values=False)
        tokens = history(1)
        predictor(model).simulate(tokens, value_indicators=[True] * len(tokens), values=[1] * len(tokens))
        self.assertNotIn("values", model.calls[0])
        self.assertNotIn("value_indicators", model.calls[0])

    def test_without_prompt_values_generate_gets_none(self):
        model = ValueModel()
        predictor(model).simulate(history(1))
        self.assertNotIn("values", model.calls[0])

    def test_values_need_indicators(self):
        with self.assertRaises(ValueError):
            predictor(ValueModel()).simulate(history(1), values=[0] * 8)

    def test_stored_trajectories_keep_their_values(self):
        tokens = history(1)
        result = predictor(ValueModel(), store=True).predict_time_to_events(
            tokens,
            prediction_window_end=-1,
            value_indicators=[False] * len(tokens),
            values=[0] * len(tokens),
        )
        self.assertEqual(result.generated_trajectories, [["LAB", "X"]] * 2)
        self.assertEqual(result.generated_value_trajectories, [["BIN:2", "BIN:NONE"]] * 2)


class TestSavedValues(unittest.TestCase):
    def test_a_value_with_a_space_keeps_its_position(self):
        tte = predictor(ValueModel())
        vocab = Tokenizer().value_vocab
        # [VS] LAB LAB [PAD], the first lab is "very high"
        values = tte._decode_values(
            [4, 5, 5, 9], ["[VS]", "LAB", "LAB"], [vocab["BIN:NONE"], vocab["very high"], vocab["BIN:1"], 0]
        )
        self.assertEqual(values, ["BIN:NONE", "very high", "BIN:1"])

    def test_the_prefix_values_are_saved_next_to_the_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = [
                {
                    "_sample_identifier": f"{i}_2020_01_01",
                    "subject_id": i,
                    "prediction_time": datetime.datetime(2020, 1, 1),
                    "visit_counter": 1,
                    "boolean_value": False,
                    "predicted_boolean_probability": 0.0,
                    "predicted_boolean_value": None,
                    "time_to_event": None,
                    "trials": None,
                    "prefix": ["[VS]", "LAB", "[VE]"],
                    "prefix_values": prefix_values,
                }
                for i, prefix_values in enumerate([["BIN:NONE", "very high", "BIN:NONE"], None])
            ]
            for output in outputs:  # the locks the prediction loop takes
                open(os.path.join(tmp, f"{output['_sample_identifier']}.lock"), "w").close()
            flush_to_disk_if_full(outputs, tmp, 16, tmp, force=True)
            parquet = [f for f in os.listdir(tmp) if f.endswith(".parquet")]
            result = pd.read_parquet(os.path.join(tmp, parquet[0])).sort_values("subject_id")
            self.assertEqual(list(result["prefix_values"].iloc[0]), ["BIN:NONE", "very high", "BIN:NONE"])
            self.assertIsNone(result["prefix_values"].iloc[1])


class TestEncodePromptValues(unittest.TestCase):
    def test_tokenized_records_keep_their_values(self):
        record = {"concept_ids": ["[VS]", "LAB"], "value_indicators": [0, 1], "values": [0, 2]}
        self.assertEqual(encode_prompt_values(record, Tokenizer()), ([False, True], [0, 2]))

    def test_raw_values_are_put_in_bins_like_in_training(self):
        record = {
            "concept_ids": ["[VS]", "LAB", "LAB", "LAB", "LAB"],
            "concept_value_masks": [0, 1, 1, 1, 1],
            "concept_values": [None, 3.0, 7.0, "positive", None],
            "units": ["", "mg", "mg", "", ""],
        }
        indicators, values = encode_prompt_values(record, Tokenizer())
        self.assertEqual(indicators, [False, True, True, True, True])
        vocab = Tokenizer().value_vocab
        self.assertEqual(
            values, [vocab[NONE_BIN], vocab["BIN:1"], vocab["BIN:2"], vocab["positive"], vocab[UNKNOWN_BIN]]
        )
        # the record is not changed
        self.assertNotIn("number_as_values", record)

    def test_records_without_values(self):
        self.assertIsNone(encode_prompt_values({"concept_ids": ["[VS]"]}, Tokenizer()))


if __name__ == "__main__":
    unittest.main()


class TestRealModelReadsPromptValues(unittest.TestCase):
    def test_prompt_values_change_the_generation(self):
        from cehrgpt.models.config import CEHRGPTConfig
        from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

        tokenizer = Tokenizer()
        torch.manual_seed(7)
        model = CEHRGPT2LMHeadModel(
            CEHRGPTConfig(
                vocab_size=len(TOKENS),
                value_vocab_size=len(VALUES),
                n_positions=64,
                n_embd=32,
                n_layer=2,
                n_head=4,
                n_inner=64,
                bos_token_id=0,
                eos_token_id=None,
                pad_token_id=tokenizer.vocab["[PAD]"],
                include_values=True,
                lab_token_ids=tokenizer.lab_token_ids,
                use_sub_time_tokenization=False,
                include_ttv_prediction=False,
                include_motor_time_to_event=False,
            )
        ).eval()
        tte = predictor(model, max_new_tokens=6)
        tte.generation_config.pad_token_id = tokenizer.vocab["[PAD]"]
        tte.generation_config.return_dict_in_generate = True
        tokens = history(3)
        indicators = [token == "LAB" for token in tokens]

        seen = []
        model.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True
        )

        def run(value_id):
            values = [value_id if indicator else 0 for indicator in indicators]
            seen.clear()
            torch.manual_seed(0)
            output = tte.simulate(tokens, value_indicators=indicators, values=values, return_values=True)
            return output, seen[0], values

        (sequences, generated_values), prefill, values = run(2)
        # the prefill reads the prompt with its values
        self.assertEqual(prefill["values"].tolist(), [values])
        self.assertEqual(prefill["value_indicators"].tolist(), [indicators])
        self.assertEqual([len(s) for s in sequences], [len(v) for v in generated_values])
        # and the values change what the model predicts
        ids = torch.tensor([tokenizer.encode(tokens)])
        mask = torch.tensor([indicators])
        with torch.no_grad():
            low = model(input_ids=ids, value_indicators=mask, values=torch.where(mask, 1, 0)).logits
            high = model(input_ids=ids, value_indicators=mask, values=torch.where(mask, 2, 0)).logits
        self.assertFalse(torch.allclose(low[0, -1], high[0, -1]))
