import tempfile
import unittest

import torch

from cehrgpt.models.patient_language_model import (
    PatientLanguageModel,
    PatientLanguageModelConfig,
)
from cehrgpt.qa.collator import PatientQACollator
from cehrgpt.qa.text_tokenizer import train_text_tokenizer

QUESTION = "Will the patient have asthma within 90 days?"
TEXTS = [f"Question: {QUESTION} Answer: {a}" for a in ("Yes.", "No.")] * 30


def tiny_config(lm_type, tokenizer, **kwargs):
    return PatientLanguageModelConfig(
        lm_type=lm_type,
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        patient_feature_dim=8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )


class PatientLanguageModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = train_text_tokenizer(TEXTS, vocab_size=300, min_frequency=1)
        cls.collator = PatientQACollator(cls.tokenizer, 64)

    def batch(self, answers=("Yes.", "No.")):
        torch.manual_seed(0)
        examples = [
            {"question": QUESTION, "answer": a, "patient_features": torch.randn(8).tolist()}
            for a in answers
        ]
        return self.collator(examples)

    def test_loss_and_shapes_for_both_architectures(self):
        for lm_type in ("gpt2", "llama"):
            for prefix in (1, 3):
                with self.subTest(lm_type=lm_type, prefix=prefix):
                    model = PatientLanguageModel(
                        tiny_config(lm_type, self.tokenizer, num_prefix_tokens=prefix)
                    )
                    batch = self.batch()
                    out = model(**batch)
                    self.assertTrue(torch.isfinite(out.loss))
                    self.assertEqual(
                        out.logits.shape[1],
                        batch["input_ids"].shape[1] + prefix,
                    )

    def test_prefix_changes_the_prediction(self):
        model = PatientLanguageModel(tiny_config("gpt2", self.tokenizer)).eval()
        batch = self.batch(("Yes.",))
        with torch.no_grad():
            first = model(**batch).logits
            batch["patient_features"] = batch["patient_features"] + 1.0
            second = model(**batch).logits
        self.assertFalse(torch.allclose(first, second))

    def test_gradients_reach_the_projector(self):
        model = PatientLanguageModel(tiny_config("llama", self.tokenizer))
        model(**self.batch()).loss.backward()
        grads = [p.grad for p in model.patient_projector.parameters()]
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in grads))

    def test_save_and_load_round_trip(self):
        model = PatientLanguageModel(tiny_config("gpt2", self.tokenizer)).eval()
        batch = self.batch()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            loaded = PatientLanguageModel.from_pretrained(folder).eval()
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(model(**batch).logits, loaded(**batch).logits, atol=1e-5)
            )

    def test_generate_returns_new_tokens(self):
        model = PatientLanguageModel(tiny_config("llama", self.tokenizer)).eval()
        prompt = self.collator.collate_prompts([QUESTION] * 2, torch.randn(2, 8).tolist())
        out = model.generate(**prompt, max_new_tokens=5, do_sample=False)
        self.assertEqual(out.shape[0], 2)
        self.assertLessEqual(out.shape[1], 5)

    def test_generation_is_clamped_to_the_available_positions(self):
        model = PatientLanguageModel(tiny_config("gpt2", self.tokenizer)).eval()
        prompt = self.collator.collate_prompts([QUESTION], torch.randn(1, 8).tolist())
        long_prompt = prompt["input_ids"].shape[1]
        room = 64 + 1 - (1 + long_prompt)
        out = model.generate(**prompt, max_new_tokens=500, do_sample=False)
        self.assertLessEqual(out.shape[1], room)
        # A prompt that fills every position cannot be continued
        full = {
            "patient_features": torch.randn(1, 8),
            "input_ids": torch.full((1, 64), self.tokenizer.bos_token_id),
        }
        with self.assertRaises(ValueError):
            model.generate(**full, max_new_tokens=3)

    def test_inputs_are_only_the_question_and_the_patient_state(self):
        import inspect

        parameters = set(inspect.signature(PatientLanguageModel.forward).parameters)
        self.assertEqual(
            parameters,
            {"self", "patient_features", "input_ids", "attention_mask", "labels"},
        )
        model = PatientLanguageModel(tiny_config("gpt2", self.tokenizer))
        with self.assertRaises(TypeError):
            model(**self.batch(), age_at_index=torch.tensor([1, 2]))

    def test_wrong_feature_width_is_rejected(self):
        model = PatientLanguageModel(tiny_config("gpt2", self.tokenizer))
        batch = self.batch()
        batch["patient_features"] = torch.cat(
            [batch["patient_features"], batch["patient_features"]], dim=-1
        )
        with self.assertRaisesRegex(ValueError, "disable_combine_global_local"):
            model(**batch)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            PatientLanguageModelConfig(lm_type="bert")
        with self.assertRaises(ValueError):
            PatientLanguageModelConfig(num_prefix_tokens=0)


class PatientVectorIsUsedTest(unittest.TestCase):
    """The answer is only a function of the patient vector, so the model has to read the prefix."""

    def test_learns_to_answer_from_the_patient_vector(self):
        torch.manual_seed(0)
        tokenizer = train_text_tokenizer(TEXTS, vocab_size=300, min_frequency=1)
        collator = PatientQACollator(tokenizer, 64)
        centres = {"Yes.": torch.randn(8), "No.": -torch.randn(8).abs() - 1}
        centres["Yes."] = centres["Yes."].abs() + 1
        model = PatientLanguageModel(tiny_config("gpt2", tokenizer))
        optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)

        def examples(n):
            out = []
            for i in range(n):
                answer = "Yes." if i % 2 == 0 else "No."
                features = centres[answer] + 0.1 * torch.randn(8)
                out.append(
                    {"question": QUESTION, "answer": answer, "patient_features": features.tolist()}
                )
            return out

        model.train()
        losses = []
        for _ in range(150):
            loss = model(**collator(examples(16))).loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        # Answer token + EOS: chance level for the answer token alone would keep the loss high
        self.assertLess(losses[-1], 0.25 * losses[0])

        model.eval()
        test = examples(20)
        prompt = collator.collate_prompts(
            [e["question"] for e in test], [e["patient_features"] for e in test]
        )
        generated = model.generate(**prompt, max_new_tokens=6, do_sample=False)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        correct = sum(d.strip() == e["answer"] for d, e in zip(decoded, test))
        self.assertGreaterEqual(correct, 19)


if __name__ == "__main__":
    unittest.main()
