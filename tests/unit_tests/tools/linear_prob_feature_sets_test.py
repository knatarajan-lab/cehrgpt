import argparse
import datetime
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cehrgpt.tools.linear_prob import train_with_cehrgpt_features as train
from cehrgpt.tools.linear_prob.feature_io import (
    FEATURE_INFO_FILE,
    read_feature_info,
    resolve_feature_sets,
    write_feature_info,
)

HIDDEN = 6


def write_split(root: Path, split: str, n: int, seed: int, width: int, informative: slice):
    """Features where only the columns of `informative` carry the label; the others are noise."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    features = rng.normal(size=(n, width))
    features[:, informative] += label[:, None] * 2.0
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
        feature_sets=None,
        hidden_size=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def metrics_of(root: Path, folder: str) -> dict:
    return json.loads((root / folder / "metrics.json").read_text())


COMBINED = {"layout": ["global", "local"], "hidden_size": HIDDEN, "num_features": 2 * HIDDEN,
            "global_is_last_hidden_state": False}


class ResolveFeatureSetsTest(unittest.TestCase):
    def test_combined_features_allow_the_three_sets_with_the_saved_vectors_first(self):
        sets = resolve_feature_sets(None, 2 * HIDDEN, COMBINED)
        self.assertEqual(list(sets), ["concatenated", "global", "local"])
        self.assertIsNone(sets["concatenated"])
        self.assertEqual(sets["global"], slice(0, HIDDEN))
        self.assertEqual(sets["local"], slice(HIDDEN, 2 * HIDDEN))
        self.assertEqual(list(resolve_feature_sets(["all"], 2 * HIDDEN, COMBINED)), list(sets))

    def test_a_set_can_be_asked_for_by_name(self):
        sets = resolve_feature_sets(["local", "global", "local"], 2 * HIDDEN, COMBINED)
        self.assertEqual(list(sets), ["local", "global"])

    def test_features_that_were_not_combined_are_the_global_features(self):
        info = {"layout": ["global"], "hidden_size": HIDDEN, "num_features": HIDDEN, "global_is_last_hidden_state": False}
        self.assertEqual(resolve_feature_sets(None, HIDDEN, info), {"global": None})
        self.assertEqual(resolve_feature_sets(["global"], HIDDEN, info), {"global": None})
        with self.assertRaisesRegex(ValueError, "saved combined"):
            resolve_feature_sets(["local"], HIDDEN, info)
        with self.assertRaisesRegex(ValueError, "saved combined"):
            resolve_feature_sets(["concatenated"], HIDDEN, info)

    def test_without_the_info_file_the_hidden_size_tells(self):
        self.assertEqual(list(resolve_feature_sets(None, 2 * HIDDEN, None, HIDDEN)), ["concatenated", "global", "local"])
        # nothing tells: the vectors are used as they are, and the parts cannot be asked for
        self.assertEqual(resolve_feature_sets(None, 2 * HIDDEN, None), {"concatenated": None})
        self.assertEqual(resolve_feature_sets(["concatenated"], 2 * HIDDEN, None), {"concatenated": None})
        with self.assertRaisesRegex(ValueError, "--hidden_size"):
            resolve_feature_sets(["global"], 2 * HIDDEN, None)

    def test_a_hidden_size_that_does_not_fit_the_width_is_an_error(self):
        # neither the width nor half of it: the vectors would be trained as global features without being them
        for wrong in (HIDDEN + 1, 3 * HIDDEN, 2 * HIDDEN + 2):
            with self.assertRaisesRegex(ValueError, "neither the hidden size"):
                resolve_feature_sets(["global"], 2 * HIDDEN, None, wrong)
        # the width itself is features that were not combined; half of it is combined
        self.assertEqual(resolve_feature_sets(["global"], 2 * HIDDEN, None, 2 * HIDDEN), {"global": None})
        self.assertEqual(resolve_feature_sets(["global"], 2 * HIDDEN, None, HIDDEN)["global"], slice(0, HIDDEN))

    def test_a_model_without_a_motor_head_has_the_same_global_and_local_features(self):
        info = dict(COMBINED, global_is_last_hidden_state=True)
        self.assertEqual(list(resolve_feature_sets(None, 2 * HIDDEN, info)), ["concatenated", "global"])
        # asked for by name, the local features are still trained
        self.assertIn("local", resolve_feature_sets(["local"], 2 * HIDDEN, info))

    def test_unknown_names_and_inconsistent_info(self):
        with self.assertRaisesRegex(ValueError, "Unknown feature sets"):
            resolve_feature_sets(["both"], 2 * HIDDEN, COMBINED)
        with self.assertRaisesRegex(ValueError, "describes"):
            resolve_feature_sets(None, 5, COMBINED)

    def test_the_info_file_round_trips(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(read_feature_info(Path(folder)))
            info = write_feature_info(Path(folder), 2 * HIDDEN, combined=True, global_is_last_hidden_state=False)
            self.assertEqual(read_feature_info(Path(folder)), info)
            self.assertEqual(info["layout"], ["global", "local"])
            self.assertEqual(info["hidden_size"], HIDDEN)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), [FEATURE_INFO_FILE])
            single = write_feature_info(Path(folder), HIDDEN, combined=False, global_is_last_hidden_state=True)
            self.assertEqual((single["layout"], single["hidden_size"]), (["global"], HIDDEN))
            with self.assertRaises(ValueError):
                write_feature_info(Path(folder), 7, combined=True, global_is_last_hidden_state=False)


class TrainFeatureSetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def combined(self, informative: slice):
        write_split(self.root, "train", 400, 0, 2 * HIDDEN, informative)
        write_split(self.root, "test", 300, 1, 2 * HIDDEN, informative)
        write_feature_info(self.root, 2 * HIDDEN, combined=True, global_is_last_hidden_state=False)

    def test_three_models_are_trained_and_the_saved_vectors_keep_the_legacy_paths(self):
        self.combined(informative=slice(0, HIDDEN))  # only the global half knows the label
        train.main(make_args(self.root))
        for folder in ("logistic", "logistic_global", "logistic_local"):
            self.assertTrue((self.root / folder / "test_predictions" / "predictions.parquet").exists(), folder)
        self.assertTrue((self.root / "feature_processor.pickle").exists())
        self.assertTrue((self.root / "feature_processor_global.pickle").exists())
        self.assertTrue((self.root / "feature_processor_local.pickle").exists())
        concatenated, global_, local = (metrics_of(self.root, f) for f in ("logistic", "logistic_global", "logistic_local"))
        self.assertGreater(global_["roc_auc"], 0.9)
        self.assertLess(local["roc_auc"], 0.65)  # noise only
        self.assertGreater(concatenated["roc_auc"], 0.9)

    def test_the_local_features_can_be_the_informative_ones(self):
        self.combined(informative=slice(HIDDEN, 2 * HIDDEN))
        train.main(make_args(self.root))
        self.assertGreater(metrics_of(self.root, "logistic_local")["roc_auc"], 0.9)
        self.assertLess(metrics_of(self.root, "logistic_global")["roc_auc"], 0.65)

    def test_the_models_use_the_columns_of_their_set(self):
        self.combined(informative=slice(0, 2 * HIDDEN))
        train.main(make_args(self.root))
        import pickle

        for folder, width in (("logistic", 2 * HIDDEN), ("logistic_global", HIDDEN), ("logistic_local", HIDDEN)):
            with open(self.root / folder / "model.pickle", "rb") as f:
                self.assertEqual(pickle.load(f).coef_.shape[1], width, folder)

    def test_one_set_can_be_asked_for_alone(self):
        self.combined(informative=slice(0, HIDDEN))
        train.main(make_args(self.root, feature_sets=["local"]))
        self.assertTrue((self.root / "logistic_local" / "metrics.json").exists())
        self.assertFalse((self.root / "logistic").exists())
        self.assertFalse((self.root / "logistic_global").exists())

    def test_finished_sets_are_not_trained_again_and_the_others_are(self):
        self.combined(informative=slice(0, HIDDEN))
        train.main(make_args(self.root, feature_sets=["global"]))
        model = self.root / "logistic_global" / "model.pickle"
        before = model.stat().st_mtime_ns
        train.main(make_args(self.root))  # all: global is done, the other two are not
        self.assertEqual(model.stat().st_mtime_ns, before)
        self.assertTrue((self.root / "logistic" / "metrics.json").exists())
        self.assertTrue((self.root / "logistic_local" / "metrics.json").exists())

    def test_features_that_were_not_combined_train_one_model_at_the_legacy_paths(self):
        write_split(self.root, "train", 400, 0, HIDDEN, slice(0, HIDDEN))
        write_split(self.root, "test", 300, 1, HIDDEN, slice(0, HIDDEN))
        write_feature_info(self.root, HIDDEN, combined=False, global_is_last_hidden_state=False)
        train.main(make_args(self.root))
        self.assertGreater(metrics_of(self.root, "logistic")["roc_auc"], 0.9)
        self.assertEqual(sorted(p.name for p in self.root.glob("logistic*")), ["logistic"])
        with self.assertRaisesRegex(ValueError, "saved combined"):
            train.main(make_args(self.root, feature_sets=["local"]))

    def test_features_from_before_the_info_file_train_as_they_did(self):
        write_split(self.root, "train", 400, 0, 2 * HIDDEN, slice(0, HIDDEN))
        write_split(self.root, "test", 300, 1, 2 * HIDDEN, slice(0, HIDDEN))
        train.main(make_args(self.root))
        self.assertEqual(sorted(p.name for p in self.root.glob("logistic*")), ["logistic"])
        # the hidden size is what lets the parts be told apart
        train.main(make_args(self.root, hidden_size=HIDDEN))
        self.assertGreater(metrics_of(self.root, "logistic_global")["roc_auc"], 0.9)

    def test_every_feature_set_is_a_run_of_its_own_in_wandb(self):
        import sys
        from unittest import mock

        self.combined(informative=slice(0, HIDDEN))
        wandb = mock.MagicMock()
        runs = []
        wandb.init.side_effect = lambda **kwargs: runs.append(kwargs) or mock.MagicMock()
        with mock.patch.dict(sys.modules, {"wandb": wandb}):
            train.main(make_args(self.root, report_to="wandb"))
        self.assertEqual(len(runs), 3)
        self.assertEqual(len({run["id"] for run in runs}), 3)
        names = sorted(run["name"] for run in runs)
        cohort = f"{self.root.parent.name}/{self.root.name}"
        self.assertEqual(names, sorted([cohort, f"{cohort}/global", f"{cohort}/local"]))
        self.assertEqual(sorted(run["config"]["feature_set"] for run in runs), ["concatenated", "global", "local"])
        # the group is the model, so the three sets are compared side by side
        self.assertEqual(len({run["group"] for run in runs}), 1)

    def test_a_namespace_from_before_the_option_still_works(self):
        write_split(self.root, "train", 300, 0, 4, slice(0, 4))
        write_split(self.root, "test", 200, 1, 4, slice(0, 4))
        old = argparse.Namespace(
            features_data_dir=str(self.root), output_dir=str(self.root), report_to="none",
            wandb_project=None, wandb_group=None, wandb_run_name=None,
        )
        train.main(old)
        self.assertTrue((self.root / "logistic" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
