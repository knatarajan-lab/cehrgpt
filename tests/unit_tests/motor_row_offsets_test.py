import unittest

import numpy as np

from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)

from .motor_time_to_event_labels_test import FakeTokenizer, random_record


class CollatorTokenizer(FakeTokenizer):
    """What the collators need besides the motor parents of FakeTokenizer."""

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
    record["input_ids"] = [vocab[c] for c in record["concept_ids"]]
    record["ages"] = [40] * len(record["concept_ids"])
    return record


def motor_rows(batch):
    """(number of prediction rows, row index of every pair) of a batch."""
    if "motor_tte_task_indicators" not in batch:
        return 0, []
    pairs = int(batch["motor_end_index"].sum())
    return int(batch["motor_tte_task_indicators"].sum()), batch["motor_row_indices"].view(-1)[:pairs].tolist()


class MotorRowOffsetsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CollatorTokenizer()

    def collator(self, packed):
        kwargs = dict(
            tokenizer=self.tokenizer, max_length=100_000, include_motor_time_to_event=True,
            motor_sampling_probability=0.5,
        )
        if packed:
            return SamplePackingCehrGptDataCollator(100_000, 100_000, **kwargs)
        return CehrGptDataCollator(**kwargs)

    def patients_alone(self, collator, examples, seed):
        """
        The labels of every patient on its own, as the processor makes them (the collators drop the MOTOR
        keys of a batch with a single pair), with the rows shifted by the prediction rows before it.
        """
        np.random.seed(seed)
        expected, offset, rows_without_pairs = [], 0, 0
        for e in examples:
            labels = collator.cehrgpt_data_processor.transform(dict(e))
            rows = int(np.asarray(labels["motor_tte_task_indicators"]).sum())
            pairs = list(labels["motor_row_indices"])
            expected += [r + offset for r in pairs]
            rows_without_pairs += rows - len(set(pairs))
            offset += rows
        return sorted(expected), offset, rows_without_pairs

    def check(self, packed):
        collator = self.collator(packed)
        rows_without_pairs = 0
        for seed in range(40):
            rng = np.random.default_rng(seed)
            examples = [example(self.tokenizer, rng) for _ in range(4)]
            expected, total_rows, empty_rows = self.patients_alone(collator, examples, seed)
            np.random.seed(seed)
            rows, pairs = motor_rows(collator([dict(e) for e in examples]))
            self.assertEqual(rows, total_rows)
            self.assertEqual(sorted(pairs), expected, f"seed {seed}")
            rows_without_pairs += empty_rows
        # the test is only meaningful if some prediction rows have no pair
        self.assertGreater(rows_without_pairs, 0)

    def test_packed_batches_shift_the_rows_by_the_prediction_rows_before_them(self):
        self.check(packed=True)

    def test_padded_batches_shift_the_rows_by_the_prediction_rows_before_them(self):
        self.check(packed=False)

    def silent_then_active(self):
        """Two patients: the first has prediction rows but no task ever happens after them."""
        tokenizer = self.tokenizer
        silent, active = tokenizer.clinical[:100], tokenizer.clinical[100:]
        for c in silent:
            tokenizer.parents[c] = []
        for c in active:
            tokenizer.parents[c] = ["task1", "task2"]
        rng = np.random.default_rng(0)
        tokenizer.clinical = silent
        first = example(tokenizer, rng)
        tokenizer.clinical = active
        second = example(tokenizer, rng)
        return [first, second]

    def check_silent_patient(self, packed):
        collator = self.collator(packed)
        first, second = self.silent_then_active()
        np.random.seed(0)
        labels = [collator.cehrgpt_data_processor.transform(dict(e)) for e in (first, second)]
        first_rows = int(np.asarray(labels[0]["motor_tte_task_indicators"]).sum())
        second_rows = int(np.asarray(labels[1]["motor_tte_task_indicators"]).sum())
        self.assertGreater(first_rows, 0)
        self.assertEqual(len(labels[0]["motor_row_indices"]), 0)  # nothing happens after any of its rows
        self.assertGreater(second_rows, 0)
        np.random.seed(0)
        rows, pairs = motor_rows(collator([dict(first), dict(second)]))
        self.assertEqual(rows, first_rows + second_rows)
        # the second patient's pairs belong to its own rows, after the silent patient's
        self.assertEqual(sorted(pairs), sorted(r + first_rows for r in labels[1]["motor_row_indices"]))

    def test_a_packed_patient_without_any_pair_still_takes_its_rows(self):
        self.check_silent_patient(packed=True)

    def test_a_padded_patient_without_any_pair_still_takes_its_rows(self):
        self.check_silent_patient(packed=False)


if __name__ == "__main__":
    unittest.main()
