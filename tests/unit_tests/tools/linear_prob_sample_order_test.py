import argparse
import datetime
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from cehrgpt.tools.linear_prob import train_with_cehrgpt_features as train
from cehrgpt.tools.linear_prob.train_with_cehrgpt_features import order_samples


def samples(n: int, seed: int = 0, width: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    features = rng.normal(size=(n, width))
    features[:, :3] += label[:, None] * 0.8
    return pd.DataFrame(
        {
            "subject_id": rng.integers(0, n // 2, n),  # a subject can have several samples
            "prediction_time": [datetime.datetime(2020, 1, 1) + datetime.timedelta(days=int(d)) for d in rng.integers(0, 4000, n)],
            "age_at_index": rng.integers(20, 80, n).astype(float),
            "gender_concept_id": rng.choice([8507, 8532], n),
            "race_concept_id": rng.choice([8527, 8516], n),
            "boolean_value": label.astype(bool),
            "features": list(features),
        }
    )


def keys(df: pd.DataFrame):
    return list(zip(df["subject_id"], df["prediction_time"], df["boolean_value"], [round(float(f[0]), 9) for f in df["features"]]))


def write_files(root: Path, split: str, df: pd.DataFrame, parts: int):
    """The samples in `parts` files with random names, as the extraction writes them."""
    folder = root / "features_with_label" / f"{split}_features"
    folder.mkdir(parents=True, exist_ok=True)
    for chunk in np.array_split(np.arange(len(df)), parts):
        df.iloc[chunk].to_parquet(folder / f"{uuid.uuid4()}.parquet")


def make_args(output_dir, **kwargs):
    defaults = dict(features_data_dir=str(output_dir), output_dir=str(output_dir), report_to="none", wandb_project=None,
                    wandb_group=None, wandb_run_name=None, feature_sets=["concatenated"], hidden_size=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class OrderSamplesTest(unittest.TestCase):
    def test_the_order_does_not_depend_on_the_order_of_the_input(self):
        df = samples(500)
        expected = keys(order_samples(df, 7))
        for permutation_seed in range(5):
            shuffled = df.sample(frac=1.0, random_state=100 + permutation_seed)
            self.assertEqual(keys(order_samples(shuffled, 7)), expected)

    def test_the_order_does_not_depend_on_the_files(self):
        df = samples(500)
        results = []
        for parts in (1, 3, 11):
            with tempfile.TemporaryDirectory() as tmp:
                write_files(Path(tmp), "train", df, parts)
                read = pd.read_parquet(Path(tmp) / "features_with_label" / "train_features")
                results.append(keys(order_samples(read, 42)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_samples_of_the_same_subject_and_time_have_an_order_of_their_own(self):
        # the same sample extracted twice: same subject and time, features that differ in the last digits
        df = samples(200)
        copy = df.copy()
        copy["features"] = [f + 1e-6 for f in copy["features"]]
        both = pd.concat([df, copy], ignore_index=True)
        expected = keys(order_samples(both, 3))
        for permutation_seed in range(5):
            self.assertEqual(keys(order_samples(both.sample(frac=1.0, random_state=permutation_seed), 3)), expected)

    def test_it_is_a_permutation_and_the_seed_changes_the_shuffle(self):
        df = samples(300)
        a, b, a_again = order_samples(df, 1), order_samples(df, 2), order_samples(df, 1)
        self.assertEqual(sorted(keys(a)), sorted(keys(df)))
        self.assertEqual(keys(a), keys(a_again))
        self.assertNotEqual(keys(a), keys(b))
        # the shuffle is not the sorted order
        self.assertNotEqual(keys(a), sorted(keys(df))[: len(df)])

    def test_the_default_seed_gives_the_order_of_before(self):
        # the training samples used to be sorted by subject and time and then shuffled with the seed 42
        df = samples(400)
        df = df.drop_duplicates(["subject_id", "prediction_time"])
        before = df.sort_values(["subject_id", "prediction_time"]).sample(frac=1.0, random_state=42, replace=False)
        self.assertEqual(keys(order_samples(df)), keys(before))


class TrainDeterminismTest(unittest.TestCase):
    def run_training(self, parts: int, seed=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_files(root, "train", samples(600, seed=0), parts)
            write_files(root, "test", samples(300, seed=1), parts)
            extra = {} if seed is None else {"seed": seed}
            train.main(make_args(root, **extra))
            metrics = json.loads((root / "logistic" / "metrics.json").read_text())
            predictions = pd.read_parquet(root / "logistic" / "test_predictions" / "predictions.parquet")
            return metrics, predictions

    def test_the_results_do_not_depend_on_how_the_files_were_written(self):
        first_metrics, first_predictions = self.run_training(parts=1)
        second_metrics, second_predictions = self.run_training(parts=7)
        self.assertEqual(first_metrics, second_metrics)
        # the predictions are in the same order too, so they compare row by row
        pd.testing.assert_frame_equal(first_predictions, second_predictions)

    def test_the_test_samples_are_ordered_by_subject_and_time_before_the_shuffle(self):
        _, predictions = self.run_training(parts=3)
        self.assertEqual(len(predictions), 300)
        # shuffled, not left in the order of the files or sorted
        self.assertFalse(predictions["subject_id"].is_monotonic_increasing)

    def test_a_seed_is_an_option_and_the_same_seed_gives_the_same_results(self):
        a, _ = self.run_training(parts=2, seed=5)
        b, _ = self.run_training(parts=4, seed=5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
