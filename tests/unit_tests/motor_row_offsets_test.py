import unittest

import numpy as np
import torch

from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

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


DAY = 86400


def hand_made(tokenizer, visits):
    """
    A patient from a list of visits (day, clinical tokens), one ATT token between visits, as the
    collators get it. With motor_sampling_probability 0 the prediction points are the [VE] before an ATT token.
    """
    vocab = tokenizer.get_vocab()
    concept_ids = ["year:2010", "age:40", "8507", "8527"]
    epoch = [0.0] * 4
    for i, (day, tokens) in enumerate(visits):
        concept_ids += ["[VS]", "9201"] + list(tokens) + ["[VE]"]
        epoch += [day * DAY] * (len(tokens) + 3)
        if i + 1 < len(visits):
            gap = visits[i + 1][0] - day
            concept_ids.append(f"D{gap}")
            epoch.append(visits[i + 1][0] * DAY)
    return {
        "concept_ids": concept_ids,
        "epoch_times": epoch,
        "input_ids": [vocab[c] for c in concept_ids],
        "ages": [40] * len(concept_ids),
    }


class HandComputedBatchTest(unittest.TestCase):
    """A batch small enough to check by hand, with the expected labels written out."""

    def setUp(self):
        self.tokenizer = CollatorTokenizer()
        # three clinical tokens: 1000 -> task0, 1001 -> task1, 1002 -> no task (the token id of taskN is 5000 + N)
        self.tokenizer.parents = {c: [] for c in self.tokenizer.clinical}
        self.tokenizer.parents.update({"1000": ["task0"], "1001": ["task1"], "1002": []})

    def collator(self, packed):
        kwargs = dict(
            tokenizer=self.tokenizer, max_length=100_000, include_motor_time_to_event=True,
            motor_sampling_probability=0.0,
        )
        if packed:
            return SamplePackingCehrGptDataCollator(100_000, 100_000, **kwargs)
        return CehrGptDataCollator(**kwargs)

    def patients(self):
        # P1: task1 happens on day 5, nothing after day 5 has a task; so its row 1 (day 5) has no pair
        p1 = hand_made(self.tokenizer, [(0, ["1000"]), (5, ["1001"]), (10, ["1002"])])
        # P2: rows at days 0 and 1; task1 happens on day 1 and task0 on day 2
        p2 = hand_made(self.tokenizer, [(0, ["1000"]), (1, ["1001"]), (2, ["1000"])])
        return [p1, p2]

    def check(self, packed):
        batch = self.collator(packed)([dict(p) for p in self.patients()])
        rows = int(batch["motor_tte_task_indicators"].sum())
        pairs = int(batch["motor_end_index"].sum())
        got = list(zip(
            batch["motor_row_indices"].view(-1)[:pairs].tolist(),
            batch["motor_col_indices"].view(-1)[:pairs].tolist(),
            batch["motor_values"].view(-1)[:pairs].tolist(),
        ))
        # P1: 2 prediction rows. Row 0 (day 0): task1 at day 5, 5 days away. Row 1 (day 5): nothing, no pair.
        # P2: 2 prediction rows, so its rows are 2 and 3 in the batch.
        #     Row 2 (day 0): task0 at day 2 (2 days), task1 at day 1 (1 day). Row 3 (day 1): task0 at day 2 (1 day).
        expected = [(0, 5001, 5.0), (2, 5000, 2.0), (2, 5001, 1.0), (3, 5000, 1.0)]
        self.assertEqual(rows, 4)
        self.assertEqual(sorted(got), expected)

    def test_packed(self):
        self.check(packed=True)

    def test_padded(self):
        self.check(packed=False)


N_EMBD = 16
MOTOR_VOCAB = 5100  # the fake task ids are 5000 + N


def motor_head_model():
    torch.manual_seed(0)
    config = CEHRGPTConfig(
        vocab_size=64, n_positions=64, n_embd=N_EMBD, n_layer=1, n_head=2, n_inner=32,
        bos_token_id=0, eos_token_id=1, pad_token_id=0, include_values=False,
        use_sub_time_tokenization=False, include_ttv_prediction=False,
        include_motor_time_to_event=True, motor_tte_vocab_size=MOTOR_VOCAB,
        motor_num_time_pieces=4, motor_time_bins=[0.0, 1.0, 7.0, 365.0, float("inf")], linear_prob_n_layer=1,
        linear_prob_token_id=8, ve_token_id=5, vs_token_id=4, att_token_ids=[6, 7],
    )
    model = CEHRGPT2LMHeadModel(config).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.motor_tte.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
        model.motor_tte.task_layer.bias.copy_(-4.0 + 0.5 * torch.randn(MOTOR_VOCAB, generator=generator))
    return model


class MotorLossOfAPackedBatchTest(unittest.TestCase):
    """
    The MOTOR loss averages over rows and tasks, so the loss of a batch has to be the average of the losses of
    its patients weighted by their rows, when every row sees the same hidden state. It fails if a pair is
    attached to the hidden state of another row.
    """

    def setUp(self):
        self.tokenizer = CollatorTokenizer()
        self.model = motor_head_model()

    def loss(self, hidden, labels):
        return self.model.motor_nll_loss(
            hidden_states=hidden,
            motor_row_indices=labels["motor_row_indices"],
            motor_col_indices=labels["motor_col_indices"],
            motor_values=labels["motor_values"],
            motor_censor_times=labels["motor_censor_times"],
            motor_tte_task_indicators=labels["motor_tte_task_indicators"],
            motor_end_index=labels["motor_end_index"],
        )

    def single_labels(self, processor, record):
        labels = processor.transform(dict(record))
        n = len(labels["input_ids"])
        pairs = len(labels["motor_row_indices"])
        return {
            "motor_row_indices": torch.tensor(labels["motor_row_indices"], dtype=torch.long),
            "motor_col_indices": torch.tensor(labels["motor_col_indices"], dtype=torch.long),
            "motor_values": torch.tensor(labels["motor_values"], dtype=torch.float32),
            "motor_censor_times": torch.tensor(np.asarray(labels["motor_censor_times"]), dtype=torch.float32).view(1, n),
            "motor_tte_task_indicators": torch.tensor(labels["motor_tte_task_indicators"], dtype=torch.bool).view(1, n),
            "motor_end_index": torch.tensor([pairs]),
        }, n

    def check(self, packed, seeds=range(15), patients=4):
        weighted_rows_without_pairs = 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            records = [example(self.tokenizer, rng) for _ in range(patients)]
            kwargs = dict(tokenizer=self.tokenizer, max_length=100_000, include_motor_time_to_event=True, motor_sampling_probability=0.5)
            collator = SamplePackingCehrGptDataCollator(100_000, 100_000, **kwargs) if packed else CehrGptDataCollator(**kwargs)
            np.random.seed(seed)
            singles = [self.single_labels(collator.cehrgpt_data_processor, r) for r in records]
            torch.manual_seed(seed)
            hidden = [torch.randn(1, n, N_EMBD) for _, n in singles]  # the hidden state of every position of a patient
            expected_rows = sum(int(l["motor_tte_task_indicators"].sum()) for l, _ in singles)
            expected = sum(
                self.loss(h, l) * int(l["motor_tte_task_indicators"].sum()) for (l, _), h in zip(singles, hidden)
            ) / expected_rows
            weighted_rows_without_pairs += sum(
                int(l["motor_tte_task_indicators"].sum()) - len(set(l["motor_row_indices"].tolist())) for l, _ in singles
            )

            np.random.seed(seed)
            batch = collator([dict(r) for r in records])
            length = batch["input_ids"].shape[1]
            packed_hidden = torch.zeros(batch["input_ids"].shape[0], length, N_EMBD)
            start = 0
            for i, ((_, n), h) in enumerate(zip(singles, hidden)):
                if packed:
                    packed_hidden[0, start : start + n] = h[0]
                    start += n + 1  # the separator
                else:
                    packed_hidden[i, :n] = h[0]
            got = self.loss(packed_hidden, batch)
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-6, msg=f"seed {seed}")
        self.assertGreater(weighted_rows_without_pairs, 0)  # some rows have no pair, or the test proves little

    def test_the_loss_of_a_packed_batch_is_the_row_weighted_loss_of_its_patients(self):
        self.check(packed=True)

    def test_the_loss_of_a_padded_batch_is_the_row_weighted_loss_of_its_patients(self):
        self.check(packed=False)


if __name__ == "__main__":
    unittest.main()
