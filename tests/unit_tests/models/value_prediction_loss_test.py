import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

VOCAB_SIZE = 24
VALUE_VOCAB_SIZE = 8
SEQ_LEN = 10
LAB_TOKEN_IDS = [8, 9, 10, 11]


def build_model(**overrides) -> CEHRGPT2LMHeadModel:
    torch.manual_seed(7)
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        value_vocab_size=VALUE_VOCAB_SIZE,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        include_values=True,
        lab_token_ids=LAB_TOKEN_IDS,
        use_sub_time_tokenization=False,
        include_ttv_prediction=False,
        include_motor_time_to_event=False,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    kwargs.update(overrides)
    return CEHRGPT2LMHeadModel(CEHRGPTConfig(**kwargs)).eval()


def make_batch(value_positions):
    """One sequence whose lab tokens sit at value_positions, each with a value bin."""
    torch.manual_seed(11)
    input_ids = torch.randint(12, VOCAB_SIZE, (1, SEQ_LEN))
    value_indicators = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    values = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    for i, position in enumerate(value_positions):
        input_ids[0, position] = LAB_TOKEN_IDS[i % len(LAB_TOKEN_IDS)]
        value_indicators[0, position] = True
        values[0, position] = 2 + i % (VALUE_VOCAB_SIZE - 2)
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        ages=torch.full((1, SEQ_LEN), 40),
        value_indicators=value_indicators,
        values=values,
        labels=input_ids.clone(),
        true_value_indicators=value_indicators.clone(),
        true_values=torch.where(value_indicators, values, -100),
    )


def shifted_value_logits(model, output, batch):
    # Position t predicts the value of token t+1, conditioned on token t+1
    with torch.no_grad():
        return model.compute_next_value_logits(
            output.value_hidden_states[:, :-1], batch["input_ids"][:, 1:]
        )


def expected_value_loss(shift_value_logits, batch):
    # Summed over the values and divided by the number of labelled tokens, like every loss term
    mask = batch["true_value_indicators"][0, 1:]
    logits = shift_value_logits[0][mask]
    targets = batch["true_values"][0, 1:][mask]
    num_tokens = (batch["labels"][0, 1:] != -100).sum()
    return F.cross_entropy(logits, targets, reduction="sum") / num_tokens


class TestValueLossAlignment(unittest.TestCase):
    def test_first_value_of_a_panel_is_trained(self):
        # Position 4 has a value and position 3 does not, so the old mask (indicators[:-1])
        # dropped it and the value loss was exactly zero
        model = build_model()
        batch = make_batch(value_positions=[4])
        output = model(**batch)
        self.assertGreater(output.token_value_loss.item(), 0.0)

    def test_value_loss_is_the_sum_over_values_per_token(self):
        model = build_model()
        batch = make_batch(value_positions=[3, 4, 7])
        with torch.no_grad():
            output = model(**batch)
        torch.testing.assert_close(
            output.token_value_loss,
            expected_value_loss(shifted_value_logits(model, output, batch), batch),
        )

    def test_value_loss_is_not_scaled_by_the_lab_penalty(self):
        # Values used to be multiplied by lab_token_loss_weight on top of
        # value_prediction_loss_weight
        batch = make_batch(value_positions=[3, 4])
        with torch.no_grad():
            penalized = build_model(lab_token_penalty=True, lab_token_loss_weight=0.25)(
                **batch
            ).token_value_loss
            plain = build_model(lab_token_penalty=False)(**batch).token_value_loss
        torch.testing.assert_close(penalized, plain)


class TestLabTokenPenalty(unittest.TestCase):
    def test_lab_tokens_are_down_weighted_in_the_sum(self):
        weight = 0.25
        model = build_model(lab_token_penalty=True, lab_token_loss_weight=weight)
        batch = make_batch(value_positions=[2, 3, 4, 5])
        with torch.no_grad():
            output = model(**batch)
        labels = batch["labels"][0, 1:]
        ce = F.cross_entropy(output.logits[0, :-1], labels, reduction="none")
        weights = torch.where(
            torch.isin(labels, torch.tensor(LAB_TOKEN_IDS)), weight, 1.0
        )
        torch.testing.assert_close(
            output.token_loss, (ce * weights).sum() / labels.numel()
        )

    def test_weight_one_matches_the_plain_mean(self):
        batch = make_batch(value_positions=[2, 3, 4, 5])
        with torch.no_grad():
            penalized = build_model(lab_token_penalty=True, lab_token_loss_weight=1.0)(
                **batch
            ).token_loss
            plain = build_model(lab_token_penalty=False)(**batch).token_loss
        torch.testing.assert_close(penalized, plain)


class TestConditionedValueHead(unittest.TestCase):
    def test_value_logits_depend_on_the_next_token(self):
        model = build_model()
        hidden = torch.randn(1, 32)
        with torch.no_grad():
            first = model.compute_next_value_logits(hidden, torch.tensor([8]))
            second = model.compute_next_value_logits(hidden, torch.tensor([9]))
        self.assertFalse(torch.allclose(first, second))

    def test_next_token_uses_its_pretrained_embedding(self):
        model = build_model(
            use_pretrained_embeddings=True,
            pretrained_embedding_dim=16,
            pretrained_token_ids=[8, 12],
        )
        hidden = torch.randn(3, 32)
        # 8 is a lab with a pretrained embedding, 9 is a lab without one, 12 is not a lab
        next_token_ids = torch.tensor([8, 9, 12])
        with torch.no_grad():
            actual = model.compute_next_value_logits(hidden, next_token_ids)
            embeddings = torch.stack(
                [
                    model.cehrgpt.pretrained_wte(next_token_ids[0]),
                    model.cehrgpt.wte(next_token_ids[1]),
                    model.cehrgpt.pretrained_wte(next_token_ids[2]),
                ]
            )
            expected = model.value_head(
                model.value_condition_layer(torch.cat([hidden, embeddings], dim=-1))
            )
        torch.testing.assert_close(actual, expected)
        # The input embeddings go through the same switch
        with torch.no_grad():
            torch.testing.assert_close(
                model.cehrgpt.concept_embeddings(next_token_ids), embeddings
            )

    def test_value_loss_trains_the_conditioning_layer(self):
        model = build_model().train()
        batch = make_batch(value_positions=[3, 4, 7])
        model(**batch).token_value_loss.backward()
        grads = [p.grad for p in model.value_condition_layer.parameters()]
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in grads))

    def test_models_without_values_have_no_value_layers(self):
        model = build_model(include_values=False)
        self.assertFalse(hasattr(model, "value_condition_layer"))
        self.assertIsNone(model(**make_batch(value_positions=[])).value_hidden_states)


class ValueTokenizer:
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
        ] + [f"C{i}" for i in range(8, VOCAB_SIZE)]
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}
        self.lab_token_ids = LAB_TOKEN_IDS

    def get_vocab(self):
        return self.vocab

    def decode(self, token_ids, **kwargs):
        return [self.inverse_vocab[int(token_id)] for token_id in token_ids]


class TestConditionedValueGeneration(unittest.TestCase):
    def _generate(self, model, share_prefill):
        model._generation_share_prompt_prefill = share_prefill
        tokenizer = ValueTokenizer()
        prompt_tokens = ["year:2000", "age:20", "Gender/1", "Race/1", "VS", "C8"]
        prompt = torch.tensor([[tokenizer.vocab[t] for t in prompt_tokens]])
        generation_config = GenerationConfig(
            max_length=32,
            max_new_tokens=6,
            num_return_sequences=3,
            do_sample=True,
            top_p=1.0,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
        )
        torch.manual_seed(12345)
        return prompt.shape[1], model.generate(
            inputs=prompt,
            generation_config=generation_config,
            cehrgpt_tokenizer=tokenizer,
            value_indicators=torch.tensor([[False] * 5 + [True]]),
            values=torch.tensor([[0] * 5 + [3]]),
        )

    def test_values_are_predicted_for_the_sampled_token(self):
        model = build_model()
        calls = []
        original = model.compute_next_value_logits

        def record(hidden_states, next_token_ids):
            calls.append(next_token_ids.clone())
            return original(hidden_states, next_token_ids)

        with mock.patch.object(model, "compute_next_value_logits", side_effect=record):
            prompt_len, output = self._generate(model, share_prefill=True)

        generated = output.sequences[:, prompt_len:]
        self.assertEqual(len(calls), generated.shape[1])
        for step, next_token_ids in enumerate(calls):
            torch.testing.assert_close(next_token_ids, generated[:, step])

    def test_shared_prefill_matches_repeated_prefill(self):
        model = build_model()
        _, shared = self._generate(model, share_prefill=True)
        _, repeated = self._generate(model, share_prefill=False)
        torch.testing.assert_close(shared.sequences, repeated.sequences, rtol=0, atol=0)
        torch.testing.assert_close(
            shared.sequence_vals, repeated.sequence_vals, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
