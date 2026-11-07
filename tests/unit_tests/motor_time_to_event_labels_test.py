import logging
import unittest
from typing import Any, Dict

import numpy as np

from cehrgpt.data.cehrgpt_data_processor import CehrGptDataProcessor

LOG = logging.getLogger(__name__)


# The implementation before it was vectorized, kept as the reference the new one has to equal
def reference_time_to_event_labels(self, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates time-to-event (TTE) labels and censoring indicators for each visit in a patient's timeline.

    Processes the input sequence in reverse to compute the number of days from each visit (marked by [VE])
    to the occurrence of future motor-related events.

    Args:
        record (Dict[str, Any]): A dictionary containing the encoded patient sequence with the key "input_ids".
            This sequence includes [VS], [VE], time delta tokens, and motor TTE concept codes.

    Returns:
        Dict[str, Any]: The updated input record with added keys:
            - "time_to_event_vectors": np.ndarray of shape [num_visits, motor_vocab_size], containing time-to-event values
            - "event_indicators": np.ndarray of shape [num_visits, motor_vocab_size], where 0 = event occurred, 1 = censored
    """

    """Highly optimized vectorized version using pre-computed token type arrays."""
    concept_ids = record["concept_ids"]
    # Convert concept_ids to indices for vectorized operations
    concept_indices = np.array([self.vocab_to_idx[cid] for cid in concept_ids])
    # Vectorized token type detection
    is_att_tokens = self.is_att_token_array[concept_indices]
    is_clinical_events = self.is_clinical_event_array[concept_indices]
    time_intervals = self.time_intervals_array[concept_indices]

    # Find valid time tokens (att tokens with positive intervals)
    valid_time_tokens = is_att_tokens & (time_intervals > 0)
    n_concepts = len(concept_ids)

    # We need to make sure event_times is monotonic
    event_times = np.zeros(n_concepts, dtype=float)
    previous_time_stamp = record["epoch_times"][0]
    for i, time_stamp in enumerate(record["epoch_times"]):
        if time_stamp < previous_time_stamp:
            time_stamp = previous_time_stamp
        else:
            previous_time_stamp = time_stamp
        event_times[i] = time_stamp

    # Convert epoch times to days
    event_times = event_times / (3600 * 24)

    # Determine prediction positions
    before_valid_time_tokens = np.roll(valid_time_tokens, -1)
    # We randomly make predictions at 50% of the sequence positions
    random_prediction_positions = (
        np.random.rand(n_concepts) < self.motor_sampling_probability
    )
    # We make the predictions at the end of the group of events that share the time stamp
    prediction_positions = np.roll(event_times, -1) != event_times
    # We don't predict at the att time tokens
    prediction_positions &= ~is_att_tokens & random_prediction_positions
    # We disable TTE predictions using the demographics alone
    prediction_positions[:4] = False
    # We take the union of the random prediction positions and the positions right before time token
    prediction_positions = prediction_positions | before_valid_time_tokens
    # We exclude the events that occur at the last time stamp
    prediction_positions &= event_times != event_times[-1]

    prediction_indices = np.where(prediction_positions)[0]
    if len(prediction_indices) == 0:
        return {
            "motor_censor_times": [0.0] * n_concepts,
            "motor_row_indices": [],
            "motor_col_indices": [],
            "motor_values": [],
            "motor_tte_task_indicators": [False] * n_concepts,
        }

    # Pre-compute all motor codes for clinical events to avoid repeated lookups
    clinical_positions = np.where(is_clinical_events)[0]
    motor_codes_cache = {}  # position -> list of (motor_code, motor_token_id)

    for pos in clinical_positions:
        concept_id = concept_ids[pos]
        if concept_id in self.motor_code_cache:
            motor_codes = self.motor_code_cache[concept_id]
        else:
            motor_codes = self.tokenizer.get_motor_parents(concept_id)
            self.motor_code_cache[concept_id] = motor_codes

        if motor_codes:
            motor_codes_cache[pos] = [
                (motor_code, self.tokenizer.get_motor_token_id(motor_code))
                for motor_code in motor_codes
            ]

    # Process sections in REVERSE order but build results in FORWARD order
    section_boundaries = np.concatenate([prediction_indices, [n_concepts]])

    # Pre-allocate arrays with exact size needed
    motor_censor_times = event_times[-1] - event_times
    motor_tte_task_indicators = np.zeros(n_concepts, dtype=bool)

    # Store sparse matrix data grouped by row for efficient construction
    sparse_data_by_row = {}  # row_idx -> [(col_idx, value), ...]

    # Global motor event state that accumulates as we go backwards
    global_motor_events = {}  # motor_code -> earliest_future_time

    # Process in reverse order but assign to forward row indices
    for i in range(len(prediction_indices) - 1, -1, -1):
        start_index = prediction_indices[i]
        end_index = section_boundaries[i + 1]
        current_event_time = event_times[start_index]

        # Add new motor events from this section to global state
        section_start = start_index + 1
        section_end = end_index + 1 if end_index < n_concepts else n_concepts

        # Process clinical events in this section (in reverse order within section)
        section_clinical_positions = clinical_positions[
            (clinical_positions >= section_start)
            & (clinical_positions < section_end)
        ]

        for pos in reversed(section_clinical_positions):
            if pos in motor_codes_cache:
                concept_time = event_times[pos]
                if concept_time > current_event_time:
                    for motor_code, motor_token_id in motor_codes_cache[pos]:
                        global_motor_events[motor_code] = (
                            concept_time,
                            motor_token_id,
                        )

        # Store sparse matrix data for current prediction position
        # Even if global_motor_events is empty, we still need to record this position
        # because it indicates all motor tasks are censored at this time point
        sparse_data_by_row[i] = [
            (motor_token_id, motor_time - current_event_time)
            for motor_code, (
                motor_time,
                motor_token_id,
            ) in global_motor_events.items()
        ]
        motor_tte_task_indicators[start_index] = True

    # Build final sparse matrix lists in forward order (no reversals needed)
    motor_row_indices = []
    motor_col_indices = []
    motor_values = []

    for row_idx in sorted(sparse_data_by_row.keys()):
        for col_idx, value in sparse_data_by_row[row_idx]:
            motor_row_indices.append(row_idx)
            motor_col_indices.append(col_idx)
            motor_values.append(value)

    if len(motor_row_indices) == 0:
        LOG.debug(
            "No MOTOR tasks detected for this sample. "
            "Length: %s, last 10 concepts: %s",
            len(concept_ids),
            concept_ids[-10:] if len(concept_ids) >= 10 else concept_ids,
        )

    return {
        "motor_censor_times": motor_censor_times,
        "motor_row_indices": motor_row_indices,
        "motor_col_indices": motor_col_indices,
        "motor_values": motor_values,
        "motor_tte_task_indicators": motor_tte_task_indicators.tolist(),
    }


NUM_CLINICAL = 200
NUM_MOTOR_CODES = 60


class FakeTokenizer:
    """The parts of CehrGptTokenizer the processor needs, with random motor parents per concept."""

    pad_token_id = 0
    vs_token_id = 1
    ve_token_id = 2

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self._vocab = {t: i for i, t in enumerate(["[PAD]", "[VS]", "[VE]"])}
        for token in ["year:2010", "age:40", "8507", "8527", "9201", "9202"]:
            self._vocab[token] = len(self._vocab)
        for token in ["D1", "D5", "D30", "W2", "M1", "M6"]:
            self._vocab[token] = len(self._vocab)
        self.clinical = [str(1000 + i) for i in range(NUM_CLINICAL)]
        for token in self.clinical:
            self._vocab[token] = len(self._vocab)
        self.parents = {
            c: [f"task{k}" for k in rng.choice(NUM_MOTOR_CODES, rng.integers(0, 5), replace=False)]
            for c in self.clinical
        }
        self.calls = 0

    def get_vocab(self):
        return self._vocab

    def get_motor_parents(self, concept_id):
        self.calls += 1
        return self.parents.get(concept_id, [])

    def get_motor_token_id(self, motor_code):
        return 5000 + int(motor_code[4:])


def random_record(tokenizer, rng, tied=False, shuffled=False):
    concept_ids = ["year:2010", "age:40", "8507", "8527"]
    epoch = [1.2e9] * 4
    ts = 1.2e9
    for _ in range(int(rng.integers(1, 25))):
        concept_ids += ["[VS]", str(rng.choice(["9201", "9202"]))]
        epoch += [ts, ts]
        for token in rng.choice(tokenizer.clinical, int(rng.integers(0, 10))):
            concept_ids.append(str(token))
            epoch.append(ts)
            if tied and rng.random() < 0.3:  # another event at the same time
                concept_ids.append(str(rng.choice(tokenizer.clinical)))
                epoch.append(ts)
        concept_ids.append("[VE]")
        epoch.append(ts)
        att = str(rng.choice(["D1", "D5", "D30", "W2", "M1", "M6"]))
        days = {"D": 1, "W": 7, "M": 30}[att[0]] * int(att[1:])
        ts += days * 86400 if rng.random() < 0.9 else 0  # sometimes the time does not move
        concept_ids.append(att)
        epoch.append(ts)
    epoch = np.asarray(epoch, dtype=float)
    if shuffled:  # epoch times that go backwards, which the labels have to flatten
        epoch[4:] += rng.integers(-3, 3, len(epoch) - 4) * 3600
    return {"concept_ids": concept_ids, "epoch_times": epoch.tolist()}


def sorted_pairs(output):
    return sorted(zip(output["motor_row_indices"], output["motor_col_indices"], np.round(output["motor_values"], 9)))


class MotorLabelsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.processor = CehrGptDataProcessor(
            self.tokenizer, max_length=10_000, include_motor_time_to_event=True, pretraining=True
        )

    def compare(self, record, seed):
        np.random.seed(seed)
        expected = reference_time_to_event_labels(self.processor, record)
        np.random.seed(seed)
        got = self.processor.create_time_to_event_labels(record)
        self.assertEqual(set(got), set(expected))
        np.testing.assert_allclose(got["motor_censor_times"], expected["motor_censor_times"])
        self.assertEqual(got["motor_tte_task_indicators"], expected["motor_tte_task_indicators"])
        self.assertEqual(sorted_pairs(got), sorted_pairs(expected))
        return got

    def test_equals_the_reference_on_random_patients(self):
        rng = np.random.default_rng(1)
        total = 0
        for i in range(150):
            record = random_record(self.tokenizer, rng, tied=i % 2 == 0, shuffled=i % 5 == 0)
            total += len(self.compare(record, i)["motor_values"])
        self.assertGreater(total, 10_000)  # the comparison is not trivially empty

    def test_the_rows_are_the_prediction_points_in_order(self):
        rng = np.random.default_rng(2)
        record = random_record(self.tokenizer, rng)
        got = self.compare(record, 0)
        rows = got["motor_row_indices"]
        self.assertEqual(rows, sorted(rows))
        # a prediction point with nothing after it has no pairs, so the rows can end before the points do
        self.assertLessEqual(max(rows) + 1, sum(got["motor_tte_task_indicators"]))

    def test_a_task_is_listed_once_per_row(self):
        rng = np.random.default_rng(3)
        got = self.compare(random_record(self.tokenizer, rng), 0)
        pairs = list(zip(got["motor_row_indices"], got["motor_col_indices"]))
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_a_patient_without_prediction_points(self):
        record = {"concept_ids": ["year:2010", "age:40", "8507", "8527", "1000"], "epoch_times": [1.2e9] * 5}
        got = self.compare(record, 0)
        self.assertEqual(got["motor_row_indices"], [])

    def test_a_patient_with_no_motor_events(self):
        tokenizer = FakeTokenizer()
        tokenizer.parents = {c: [] for c in tokenizer.clinical}
        processor = CehrGptDataProcessor(tokenizer, max_length=10_000, include_motor_time_to_event=True, pretraining=True)
        record = random_record(tokenizer, np.random.default_rng(4))
        np.random.seed(0)
        got = processor.create_time_to_event_labels(record)
        np.random.seed(0)
        expected = reference_time_to_event_labels(processor, record)
        self.assertEqual(got["motor_row_indices"], [])
        self.assertEqual(got["motor_tte_task_indicators"], expected["motor_tte_task_indicators"])

    def test_the_parents_of_a_concept_are_looked_up_once(self):
        rng = np.random.default_rng(5)
        record = random_record(self.tokenizer, rng)
        self.processor.create_time_to_event_labels(record)
        calls = self.tokenizer.calls
        self.processor.create_time_to_event_labels(record)
        self.assertEqual(self.tokenizer.calls, calls)


if __name__ == "__main__":
    unittest.main()
