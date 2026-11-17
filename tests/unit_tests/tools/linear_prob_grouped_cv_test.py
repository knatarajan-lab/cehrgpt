import argparse
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cehrgpt.tools.linear_prob import train_with_cehrgpt_features as train
from cehrgpt.tools.linear_prob.feature_io import write_feature_info

from .linear_prob_feature_sets_test import HIDDEN, make_args, metrics_of, write_split


class CvFoldsTest(unittest.TestCase):
    def test_by_default_the_folds_are_a_number_as_before(self):
        args = argparse.Namespace(seed=42)
        self.assertEqual(train.cv_folds(np.zeros(10), np.arange(10), args), train.DEFAULT_CV_FOLDS)

    def test_grouped_folds_never_split_a_subject(self):
        rng = np.random.default_rng(0)
        subjects = rng.integers(0, 60, 600)
        labels = rng.integers(0, 2, 600)
        args = argparse.Namespace(seed=42, group_cv_by_subject=True, cv_folds=5)
        folds = train.cv_folds(labels, subjects, args)
        self.assertEqual(len(folds), 5)
        for fit, held_out in folds:
            self.assertFalse(set(subjects[fit]) & set(subjects[held_out]))
        all_held_out = np.concatenate([held_out for _, held_out in folds])
        self.assertEqual(sorted(all_held_out), list(range(600)))


class BlockwiseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def combined(self, informative: slice):
        write_split(self.root, "train", 400, 0, 2 * HIDDEN, informative)
        write_split(self.root, "test", 300, 1, 2 * HIDDEN, informative)
        write_feature_info(self.root, 2 * HIDDEN, combined=True, global_is_last_hidden_state=False)

    def test_the_metrics_tell_how_c_was_picked(self):
        self.combined(informative=slice(0, HIDDEN))
        train.main(make_args(self.root, feature_sets=["global"], group_cv_by_subject=True))
        metrics = metrics_of(self.root, "logistic_global")
        self.assertEqual(metrics["cv"], "grouped by subject")
        self.assertGreater(metrics["cv_roc_auc"], 0.9)
        self.assertGreater(metrics["C"], 0)

    def test_the_blockwise_model_keeps_the_scale_with_the_best_cv(self):
        self.combined(informative=slice(0, HIDDEN))
        scales = [0.0, 0.5, 1.0]
        train.main(make_args(self.root, feature_sets=["global"], local_block_scales=scales))
        metrics = metrics_of(self.root, "logistic_blockwise")
        self.assertIn(metrics["local_scale"], scales)
        by_scale = metrics["cv_roc_auc_by_local_scale"]
        self.assertEqual(sorted(by_scale), sorted(str(s) for s in scales))
        self.assertAlmostEqual(by_scale[str(metrics["local_scale"])], max(by_scale.values()))
        self.assertAlmostEqual(metrics["cv_roc_auc"], max(by_scale.values()))
        self.assertGreater(metrics["roc_auc"], 0.9)
        with open(self.root / "logistic_blockwise" / "model.pickle", "rb") as f:
            model = pickle.load(f)
        self.assertEqual(model.coef_.shape[1], 2 * HIDDEN)
        self.assertEqual(model.local_scale_, metrics["local_scale"])
        self.assertTrue((self.root / "feature_processor_blockwise.pickle").exists())

    def test_a_scale_of_zero_is_the_global_features_alone(self):
        self.combined(informative=slice(0, HIDDEN))
        train.main(make_args(self.root, feature_sets=["global"], local_block_scales=[0.0]))
        with open(self.root / "logistic_blockwise" / "model.pickle", "rb") as f:
            model = pickle.load(f)
        np.testing.assert_allclose(model.coef_[0, HIDDEN:], 0.0)
        self.assertAlmostEqual(
            metrics_of(self.root, "logistic_blockwise")["roc_auc"],
            metrics_of(self.root, "logistic_global")["roc_auc"],
            places=6,
        )

    def test_blockwise_needs_combined_features(self):
        write_split(self.root, "train", 100, 0, HIDDEN, slice(0, HIDDEN))
        write_split(self.root, "test", 100, 1, HIDDEN, slice(0, HIDDEN))
        write_feature_info(self.root, HIDDEN, combined=False, global_is_last_hidden_state=False)
        with self.assertRaisesRegex(ValueError, "saved combined"):
            train.main(make_args(self.root, local_block_scales=[0.5]))


if __name__ == "__main__":
    unittest.main()
