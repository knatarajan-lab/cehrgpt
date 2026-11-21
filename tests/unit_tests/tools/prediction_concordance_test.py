import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from cehrgpt.tools.prediction_concordance import concordance, read_predictions


def write(path: Path, rows: slice, probabilities: np.ndarray, hours: int = 0) -> None:
    n = len(probabilities)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": np.arange(n) // 2,
            "prediction_time": [datetime(2020, 1, 1) + timedelta(days=i, hours=hours) for i in range(n)],
            "predicted_boolean_probability": probabilities,
            "boolean_value": np.arange(n) % 3 == 0,
        }
    )[rows].write_parquet(path)


class PredictionConcordanceTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.risk = np.random.default_rng(0).uniform(size=400)

    def test_only_the_samples_of_both_inputs_are_compared(self):
        write(self.folder / "ref.parquet", slice(0, 300), self.risk)
        # the candidate is a folder of shards, in another order of magnitude but the same ranking
        write(self.folder / "cand" / "shard_0" / "a.parquet", slice(100, 250), self.risk**3)
        write(self.folder / "cand" / "shard_1" / "b.parquet", slice(250, 400), self.risk**3)
        result = concordance(
            read_predictions(self.folder / "ref.parquet"), read_predictions(self.folder / "cand")
        )
        self.assertEqual(result["num_intersection"], 200)
        self.assertEqual(result["num_labels_that_differ"], 0)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertAlmostEqual(result["spearman_cases"], 1.0)
        self.assertEqual(result["share_of_reference_top_10pct_in_candidate_top_10pct"], 1.0)
        self.assertAlmostEqual(result["reference_roc_auc"], result["candidate_roc_auc"])
        self.assertGreater(result["mean_absolute_difference"], 0.05)

    def test_times_that_differ_within_the_day_match_on_the_date(self):
        write(self.folder / "ref.parquet", slice(None), self.risk)
        write(self.folder / "cand.parquet", slice(None), self.risk, hours=5)
        paths = self.folder / "ref.parquet", self.folder / "cand.parquet"
        exact = concordance(*(read_predictions(p) for p in paths))
        self.assertEqual(exact["num_intersection"], 0)
        by_date = concordance(*(read_predictions(p, "date") for p in paths))
        self.assertEqual(by_date["num_intersection"], 400)


if __name__ == "__main__":
    unittest.main()
