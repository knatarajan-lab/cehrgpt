import argparse
import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from cehrgpt.tools.linear_prob import train_with_cehrgpt_features as train


def write_features(root: Path, split: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    features = rng.normal(size=(n, 4)) + label[:, None] * 1.5
    folder = root / "features_with_label" / f"{split}_features"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "prediction_time": [datetime.datetime(2020, 1, 1)] * n,
            "age_at_index": rng.integers(20, 80, n).astype(float),
            "gender_concept_id": rng.choice([8507, 8532], n),
            "race_concept_id": rng.choice([8527, 8516], n),
            "boolean_value": label.astype(bool),
            "features": list(features),
        }
    ).to_parquet(folder / "part-0.parquet")


def make_args(output_dir, **kwargs):
    defaults = dict(
        features_data_dir=str(output_dir),
        output_dir=str(output_dir),
        report_to="none",
        wandb_project=None,
        wandb_group=None,
        wandb_run_name=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestLinearProbingWandb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Mimic run_linear_prob.sh: <output>/<cohort>/<model_name>
        self.output_dir = Path(self.tmp.name) / "readmission" / "my_model"
        write_features(self.output_dir, "train", 300, 0)
        write_features(self.output_dir, "test", 150, 1)
        self.wandb = mock.MagicMock()
        patcher = mock.patch.dict(sys.modules, {"wandb": self.wandb})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_nothing_is_sent_to_wandb_by_default(self):
        train.main(make_args(self.output_dir))
        self.wandb.init.assert_not_called()
        self.assertTrue((self.output_dir / "logistic" / "metrics.json").exists())

    def test_metrics_are_logged_to_wandb(self):
        train.main(make_args(self.output_dir, report_to="wandb"))
        metrics = json.loads((self.output_dir / "logistic" / "metrics.json").read_text())
        kwargs = self.wandb.init.call_args.kwargs
        self.assertEqual(kwargs["project"], "cehrgpt-linear-probing")
        self.assertEqual(kwargs["group"], "my_model")
        self.assertEqual(kwargs["name"], "readmission/my_model")
        self.assertEqual(kwargs["resume"], "allow")
        self.assertEqual(kwargs["config"]["num_train_samples"], 300)
        self.assertEqual(kwargs["config"]["num_test_samples"], 150)
        self.assertEqual(kwargs["config"]["num_features"], 4)
        run = self.wandb.init.return_value
        run.log.assert_called_once_with(metrics)
        run.summary.update.assert_called_once_with(metrics)
        run.finish.assert_called_once()
        self.assertGreater(metrics["roc_auc"], 0.9)

    def test_project_group_and_name_can_be_set(self):
        with mock.patch.dict("os.environ", {"WANDB_PROJECT": "from-env"}):
            train.main(make_args(self.output_dir, report_to="wandb"))
            self.assertEqual(self.wandb.init.call_args.kwargs["project"], "from-env")
        self.wandb.init.reset_mock()
        (self.output_dir / "logistic" / "metrics.json").unlink()
        train.main(
            make_args(
                self.output_dir,
                report_to="wandb",
                wandb_project="p",
                wandb_group="g",
                wandb_run_name="n",
            )
        )
        kwargs = self.wandb.init.call_args.kwargs
        self.assertEqual((kwargs["project"], kwargs["group"], kwargs["name"]), ("p", "g", "n"))

    def test_rerunning_a_finished_cohort_reports_its_results_to_the_same_run(self):
        train.main(make_args(self.output_dir, report_to="wandb"))
        first = self.wandb.init.call_args.kwargs
        metrics = json.loads((self.output_dir / "logistic" / "metrics.json").read_text())
        self.wandb.reset_mock()
        train.main(make_args(self.output_dir, report_to="wandb"))
        again = self.wandb.init.call_args.kwargs
        self.assertEqual(again["id"], first["id"])
        self.wandb.init.return_value.summary.update.assert_called_once_with(metrics)

    def test_missing_wandb_package_is_reported(self):
        with mock.patch.dict(sys.modules, {"wandb": None}):
            with self.assertRaises(ImportError) as ctx:
                train.main(make_args(self.output_dir, report_to="wandb"))
        self.assertIn("pip install wandb", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
