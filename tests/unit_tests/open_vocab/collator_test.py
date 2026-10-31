import unittest

import numpy as np
import torch

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator, SamplePackingCehrGptDataCollator
from cehrgpt.open_vocab.assets import OpenVocabAssets
from cehrgpt.open_vocab.labels import QuestionHierarchy

from ..motor_time_to_event_labels_test import FakeTokenizer, random_record

DAY = 86400


def assets_for(tokenizer, num_rows=30):
    """Every clinical token of the fake tokenizer is a question, and so are a few parents."""
    questions = {int(t): [int(t)] for t in tokenizer.clinical}
    parents = [90001, 90002, 90003]  # not in the vocabulary
    for i, t in enumerate(tokenizer.clinical):
        questions[int(t)].append(parents[i % 3])
    for parent in parents:
        questions.setdefault(parent, [parent])
    hierarchy = QuestionHierarchy(questions)
    concept_ids = np.array(sorted({q for qs in questions.values() for q in qs}))[:num_rows if num_rows else None]
    rng = np.random.default_rng(0)
    return OpenVocabAssets(hierarchy, concept_ids, rng.normal(size=(len(concept_ids), 8)).astype(np.float32))


class FakeCollatorTokenizer(FakeTokenizer):
    vocab_size = 400
    pad_value_token_id = 0

    def __init__(self, seed=0):
        super().__init__(seed)
        self._vocab["[END]"] = len(self._vocab)
        self._by_id = {i: t for t, i in self._vocab.items()}
        self.end_token_id = self.random_token_id = self.linear_token_id = self._vocab["[END]"]

    def decode(self, ids, skip_special_tokens=False):
        return [self._by_id[int(i)] for i in ids]


def example(tokenizer, rng):
    record = random_record(tokenizer, rng)
    vocab = tokenizer.get_vocab()
    n = len(record["concept_ids"])
    record["input_ids"] = [vocab[c] for c in record["concept_ids"]]
    record["ages"] = [40] * n
    return record


class OpenVocabCollatorTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeCollatorTokenizer()
        self.assets = assets_for(self.tokenizer, num_rows=0)
        self.rng = np.random.default_rng(0)

    def collator(self, cls=CehrGptDataCollator, packed=False, **kwargs):
        args = (100_000, 100_000) if packed else ()
        return cls(
            *args, tokenizer=self.tokenizer, max_length=100_000, include_motor_time_to_event=True,
            motor_sampling_probability=0.5, open_vocab_assets=self.assets, open_vocab_num_questions=16, **kwargs,
        )

    def examples(self, n=3):
        return [example(self.tokenizer, self.rng) for _ in range(n)]

    def check_batch(self, batch):
        rows = int(batch["motor_tte_task_indicators"].sum())
        self.assertGreater(rows, 0)
        questions = batch["open_vocab_question_rows"]
        self.assertEqual(len(questions), 16)
        self.assertEqual(len(set(questions.tolist())), 16)
        self.assertLess(int(questions.max()), self.assets.num_rows)
        for key in ("open_vocab_event_rows", "open_vocab_event_questions", "open_vocab_event_times"):
            self.assertEqual(batch[key].shape, batch["open_vocab_event_rows"].shape, key)
        events = batch["open_vocab_event_rows"]
        self.assertGreater(len(events), 0)
        self.assertLess(int(events.max()), rows)  # the rows are the prediction points of the batch
        self.assertLess(int(batch["open_vocab_event_questions"].max()), 16)
        self.assertTrue((batch["open_vocab_event_times"] > 0).all())
        keys = list(zip(events.tolist(), batch["open_vocab_event_questions"].tolist()))
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(keys)))
        # no listed event is later than the end of the record seen from its row
        censor = batch["motor_censor_times"][batch["motor_tte_task_indicators"]]
        self.assertTrue((batch["open_vocab_event_times"] <= censor[events] + 1e-3).all())

    def test_a_padded_batch_gets_the_qa_labels(self):
        batch = self.collator()(self.examples())
        self.check_batch(batch)

    def test_a_packed_batch_gets_the_qa_labels(self):
        batch = self.collator(SamplePackingCehrGptDataCollator, packed=True)(self.examples(4))
        self.assertEqual(batch["input_ids"].shape[0], 1)
        self.check_batch(batch)

    def test_the_labels_do_not_cross_patients(self):
        # two patients packed together: the labels of the packed batch equal those of the patients one by one
        # when the same questions are asked
        examples = self.examples(2)
        collator = self.collator(SamplePackingCehrGptDataCollator, packed=True)
        collator.open_vocab_num_questions = self.assets.num_rows  # ask everything
        batch = collator([dict(e) for e in examples])
        together = sorted(zip(batch["open_vocab_event_rows"].tolist(), batch["open_vocab_event_questions"].tolist()))
        self.assertGreater(len(together), 0)
        rows_per_example = []
        pairs = []
        offset = 0
        for e in examples:
            single = collator([dict(e)])
            rows_per_example.append(int(single["motor_tte_task_indicators"].sum()))
            self.assertEqual(sorted(single["open_vocab_question_rows"].tolist()), list(range(self.assets.num_rows)))
            pairs += [(r + offset, int(single["open_vocab_question_rows"][q])) for r, q in
                      zip(single["open_vocab_event_rows"].tolist(), single["open_vocab_event_questions"].tolist())]
            offset += rows_per_example[-1]
        together_by_row = sorted((r, int(batch["open_vocab_question_rows"][q])) for r, q in together)
        self.assertEqual(together_by_row, sorted(pairs))

    def test_nothing_is_added_without_assets_or_motor(self):
        no_assets = CehrGptDataCollator(
            tokenizer=self.tokenizer, max_length=100_000, include_motor_time_to_event=True,
        )(self.examples())
        self.assertFalse(any(k.startswith("open_vocab") for k in no_assets))
        with self.assertRaises(ValueError):
            CehrGptDataCollator(tokenizer=self.tokenizer, max_length=100_000, open_vocab_assets=self.assets)

    def test_no_labels_when_fine_tuning(self):
        batch = self.collator(pretraining=False)(self.examples())
        self.assertFalse(any(k.startswith("open_vocab") for k in batch))


if __name__ == "__main__":
    unittest.main()
