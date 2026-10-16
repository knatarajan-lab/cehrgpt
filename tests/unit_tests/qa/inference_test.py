import unittest

import numpy as np
import torch

from cehrgpt.models.patient_language_model import (
    PatientLanguageModel,
    PatientLanguageModelConfig,
)
from cehrgpt.qa.collator import PatientQACollator
from cehrgpt.qa.inference import PatientQAPredictor, parse_time_to_event_answer
from cehrgpt.qa.text_tokenizer import train_text_tokenizer

BINARY = "Will the patient have asthma within 90 days?"
TIME = "How many days until the patient's next asthma?"
TEXTS = [
    f"Question: {BINARY} Answer: {a}" for a in ("Yes.", "No.")
] + [f"Question: {TIME} Answer: About {d} days." for d in (10, 20, 30)] * 4
TEXTS += ["Question: x Answer: It was not observed in the 200 days of follow-up."] * 4


def config(tokenizer):
    return PatientLanguageModelConfig(
        lm_type="gpt2",
        vocab_size=len(tokenizer),
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=96,
        patient_feature_dim=8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


class ParseTest(unittest.TestCase):
    def test_days(self):
        for text, days in (("About 40 days.", 40.0), ("About 1 day.", 1.0), ("about 365 days", 365.0)):
            prediction = parse_time_to_event_answer(text)
            self.assertEqual(prediction.days, days)
            self.assertFalse(prediction.event_not_observed)

    def test_not_observed(self):
        prediction = parse_time_to_event_answer(
            "It was not observed in the 200 days of follow-up."
        )
        self.assertIsNone(prediction.days)
        self.assertTrue(prediction.event_not_observed)

    def test_garbage(self):
        prediction = parse_time_to_event_answer("hmm")
        self.assertIsNone(prediction.days)
        self.assertFalse(prediction.event_not_observed)


class PredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tokenizer = train_text_tokenizer(TEXTS, vocab_size=400, min_frequency=1)
        cls.collator = PatientQACollator(cls.tokenizer, 96)
        cls.model = PatientLanguageModel(config(cls.tokenizer))
        centre = torch.randn(8).abs() + 1
        cls.centres = {"Yes.": centre, "No.": -centre}
        # Distinct patterns (not constants: the projector layer-normalises the vector)
        cls.time_patterns = {10: torch.randn(8), 30: torch.randn(8)}
        optimiser = torch.optim.AdamW(cls.model.parameters(), lr=3e-3)

        def example(i):
            kind = i % 4
            if kind in (0, 1):
                answer = "Yes." if kind == 0 else "No."
                features = cls.centres[answer] + 0.1 * torch.randn(8)
                return {"question": BINARY, "answer": answer, "patient_features": features.tolist()}
            # The number of days is a function of the patient vector too
            days = 10 if kind == 2 else 30
            features = cls.time_patterns[days] + 0.05 * torch.randn(8)
            return {
                "question": TIME,
                "answer": f"About {days} days.",
                "patient_features": features.tolist(),
            }

        cls.model.train()
        for _ in range(250):
            loss = cls.model(**cls.collator([example(i) for i in range(16)])).loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        cls.model.eval()
        cls.predictor = PatientQAPredictor(cls.model, cls.tokenizer, 96, batch_size=5)

    def test_probability_of_yes_separates_the_classes(self):
        torch.manual_seed(1)
        features, labels = [], []
        for i in range(24):
            answer = "Yes." if i % 2 == 0 else "No."
            features.append((self.centres[answer] + 0.1 * torch.randn(8)).tolist())
            labels.append(answer == "Yes.")
        scores = self.predictor.probability_of_yes(features, [BINARY] * len(features))
        self.assertEqual(scores.shape, (24,))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        labels = np.array(labels)
        # Every positive scores above every negative: AUROC of 1
        self.assertGreater(scores[labels].min(), scores[~labels].max())

    def test_batching_does_not_change_the_scores(self):
        features = [(self.centres["Yes."] + 0.1 * i).tolist() for i in range(7)]
        questions = [BINARY] * 6 + ["Will the patient have asthma within 365 days?"]
        batched = self.predictor.probability_of_yes(features, questions)
        single = np.array(
            [self.predictor.probability_of_yes([f], [q])[0] for f, q in zip(features, questions)]
        )
        np.testing.assert_allclose(batched, single, atol=1e-4)

    def test_time_to_event_returns_the_days_the_model_learned(self):
        short = [self.time_patterns[10].tolist()]
        long_ = [self.time_patterns[30].tolist()]
        first = self.predictor.predict_time_to_event(short, [TIME])[0]
        second = self.predictor.predict_time_to_event(long_, [TIME])[0]
        self.assertEqual((first.days, second.days), (10.0, 30.0))

    def test_empty_input(self):
        self.assertEqual(len(self.predictor.probability_of_yes([], [])), 0)


class NextTokenLogitsTest(unittest.TestCase):
    def test_left_padding_does_not_change_the_logits(self):
        torch.manual_seed(0)
        tokenizer = train_text_tokenizer(TEXTS, vocab_size=400, min_frequency=1)
        collator = PatientQACollator(tokenizer, 96)
        model = PatientLanguageModel(config(tokenizer)).eval()
        features = torch.randn(2, 8).tolist()
        questions = [TIME, BINARY]  # different lengths, so the shorter one is padded
        batch = collator.collate_prompts(questions, features)
        together = model.next_token_logits(**batch)
        for i in range(2):
            alone = model.next_token_logits(
                **collator.collate_prompts([questions[i]], [features[i]])
            )
            torch.testing.assert_close(together[i], alone[0], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
