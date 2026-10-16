import unittest

from cehrgpt.qa.collator import PatientQACollator
from cehrgpt.qa.text_tokenizer import train_text_tokenizer

TEXTS = [
    "Question: Will the patient have asthma within 90 days? Answer: Yes.",
    "Question: How old is the patient? Answer: 60 years old.",
    "No. Yes. About 40 days.",
] * 20


class CollatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = train_text_tokenizer(TEXTS, vocab_size=400, min_frequency=1)

    def example(self, question, answer, features=(0.0, 1.0)):
        return {"question": question, "answer": answer, "patient_features": list(features)}

    def test_only_the_answer_is_labelled(self):
        batch = PatientQACollator(self.tokenizer, 64)(
            [self.example("How old is the patient?", "60 years old.")]
        )
        labels = batch["labels"][0]
        ids = batch["input_ids"][0]
        answer_positions = (labels != -100).nonzero().flatten()
        self.assertGreater(len(answer_positions), 1)
        # The labelled span is a suffix ending in EOS, and it equals the input there
        self.assertEqual(ids[-1].item(), self.tokenizer.eos_token_id)
        self.assertTrue((labels[answer_positions] == ids[answer_positions]).all())
        self.assertEqual(ids[0].item(), self.tokenizer.bos_token_id)
        self.assertEqual(labels[0].item(), -100)
        decoded = self.tokenizer.decode(ids[answer_positions])
        self.assertIn("60 years old", decoded)

    def test_batch_padding(self):
        batch = PatientQACollator(self.tokenizer, 64)(
            [
                self.example("How old is the patient?", "60 years old."),
                self.example("Will the patient have asthma within 90 days?", "No."),
            ]
        )
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(tuple(batch["patient_features"].shape), (2, 2))
        pad = self.tokenizer.pad_token_id
        shorter = batch["attention_mask"].sum(dim=1).argmin()
        padded = batch["attention_mask"][shorter] == 0
        self.assertTrue((batch["input_ids"][shorter][padded] == pad).all())
        self.assertTrue((batch["labels"][shorter][padded] == -100).all())

    def test_long_question_is_trimmed_before_the_answer(self):
        batch = PatientQACollator(self.tokenizer, 12)(
            [self.example("Will the patient have asthma within 90 days? " * 5, "No.")]
        )
        self.assertEqual(batch["input_ids"].shape[1], 12)
        self.assertGreater((batch["labels"][0] != -100).sum().item(), 1)

    def test_prompts_are_left_padded(self):
        collator = PatientQACollator(self.tokenizer, 64)
        batch = collator.collate_prompts(
            ["How old is the patient?", "Will the patient have asthma within 90 days?"],
            [[0.0, 1.0], [1.0, 0.0]],
        )
        self.assertTrue((batch["attention_mask"][:, -1] == 1).all())
        self.assertNotIn("labels", batch)


if __name__ == "__main__":
    unittest.main()
