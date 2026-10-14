import datetime
import pickle
import time
import unittest

import numpy as np

from cehrgpt.data.hf_cehrgpt_dataset_mapping import (
    ExtractTokenizedSequenceDataMapping,
    _PredictionTimeIndex,
)

UTC = datetime.timezone.utc


def _reference(entry):
    # What the extraction computed before the prediction times were stored as arrays
    return entry["index_date"].replace(tzinfo=UTC).timestamp(), entry["label"]


class PredictionTimeIndexTest(unittest.TestCase):
    def setUp(self):
        self.person_map = {
            30: [{"index_date": datetime.datetime(2020, 5, 1, 10, 30), "label": True}],
            7: [
                # several prediction times for one person keep their order
                {"index_date": datetime.datetime(2019, 1, 2, 3, 4, 5, 678901), "label": False},
                {"index_date": datetime.datetime(2018, 1, 1, tzinfo=UTC), "label": True},
            ],
            12: [{"index_date": datetime.datetime(2021, 12, 31, 23, 59), "label": False}],
        }
        self.index = _PredictionTimeIndex(self.person_map)

    def test_lookup_matches_the_dict(self):
        for person_id, entries in self.person_map.items():
            timestamps, labels = self.index.get(person_id)
            expected = [_reference(entry) for entry in entries]
            self.assertEqual(list(zip(timestamps, labels)), expected)

    def test_labels_come_back_as_python_objects(self):
        _, labels = self.index.get(7)
        self.assertTrue(all(isinstance(label, bool) for label in labels))
        int_index = _PredictionTimeIndex(
            {1: [{"index_date": datetime.datetime(2020, 1, 1), "label": 1}]}
        )
        self.assertIsInstance(int_index.get(1)[1][0], int)

    def test_unknown_person_raises_key_error(self):
        for person_id in (0, 8, 13, 31):
            with self.assertRaises(KeyError):
                self.index.get(person_id)

    def test_length(self):
        self.assertEqual(len(self.index), 3)

    def test_survives_pickling(self):
        restored = pickle.loads(pickle.dumps(self.index))
        for person_id in self.person_map:
            self.assertEqual(restored.get(person_id), self.index.get(person_id))

    def test_mapping_no_longer_carries_the_dict(self):
        mapping = ExtractTokenizedSequenceDataMapping(self.person_map, 5)
        self.assertFalse(hasattr(mapping, "person_index_date_map"))
        restored = pickle.loads(pickle.dumps(mapping))
        self.assertEqual(restored._prediction_times.get(7), self.index.get(7))

    def test_large_map_pickles_quickly(self):
        # The dict version needed ~25 s for 900k persons; the arrays are a few MB
        n = 200_000
        base = datetime.datetime(2015, 1, 1)
        big = {
            person_id: [
                {"index_date": base + datetime.timedelta(minutes=person_id), "label": person_id % 2 == 0}
            ]
            for person_id in range(n)
        }
        mapping = ExtractTokenizedSequenceDataMapping(big, 0)
        start = time.time()
        blob = pickle.dumps(mapping)
        self.assertLess(time.time() - start, 1.0)
        self.assertLess(len(blob), 10 * 2**20)
        timestamps, labels = mapping._prediction_times.get(123_456)
        self.assertEqual((timestamps[0], labels[0]), _reference(big[123_456][0]))


if __name__ == "__main__":
    unittest.main()
