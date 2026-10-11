import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from cehrgpt.tools.bootstrap_ci import (
    bootstrap_confidence_intervals,
    compute_metrics,
    read_predictions,
)


def make_predictions(num_patients=150, labels_per_patient=3, seed=1):
    rng = np.random.default_rng(seed)
    subject_ids = np.repeat(np.arange(num_patients), labels_per_patient)
    # The risk of a patient is shared by their labels, which makes the labels correlated
    patient_risk = np.repeat(rng.normal(size=num_patients), labels_per_patient)
    y = (patient_risk + rng.normal(size=len(subject_ids)) > 0.5).astype(int)
    score = 1 / (1 + np.exp(-(0.8 * y + patient_risk + rng.normal(size=len(y)))))
    return subject_ids, y, score


class TestBootstrapConfidenceIntervals(unittest.TestCase):
    def setUp(self):
        self.subject_ids, self.y, self.score = make_predictions()

    def test_point_estimates_match_the_linear_probing_metrics(self):
        result = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_boot=20, n_jobs=1
        )
        precision, recall, _ = precision_recall_curve(self.y, self.score)
        self.assertAlmostEqual(result["roc_auc"], roc_auc_score(self.y, self.score))
        self.assertAlmostEqual(result["pr_auc"], auc(recall, precision))
        self.assertEqual(result["n_labels"], len(self.y))
        self.assertEqual(result["n_patients"], 150)
        self.assertAlmostEqual(result["prevalence"], self.y.mean())

    def test_interval_surrounds_the_estimate(self):
        result = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_boot=200, n_jobs=1
        )
        for name in ("roc_auc", "pr_auc"):
            self.assertLess(result[f"{name}_low"], result[name])
            self.assertLess(result[name], result[f"{name}_high"])
            self.assertGreater(result[f"{name}_boot_std"], 0)
        self.assertEqual(result["n_boot"], 200)

    def test_resampling_patients_is_wider_than_resampling_labels(self):
        # With correlated labels the patient-level interval has to be wider than the one that
        # treats every label as a patient
        patient_level = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_boot=300, n_jobs=1
        )
        label_level = bootstrap_confidence_intervals(
            np.arange(len(self.y)), self.y, self.score, n_boot=300, n_jobs=1
        )
        self.assertGreater(
            patient_level["roc_auc_high"] - patient_level["roc_auc_low"],
            label_level["roc_auc_high"] - label_level["roc_auc_low"],
        )

    def test_result_depends_on_the_seed_but_not_on_n_jobs(self):
        kwargs = dict(n_boot=40)
        a = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_jobs=1, seed=3, **kwargs
        )
        b = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_jobs=2, seed=3, **kwargs
        )
        c = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_jobs=1, seed=4, **kwargs
        )
        self.assertEqual(a, b)
        self.assertNotEqual(a["roc_auc_low"], c["roc_auc_low"])

    def test_text_and_plus_minus(self):
        ci = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_boot=50, n_jobs=1
        )
        std = bootstrap_confidence_intervals(
            self.subject_ids, self.y, self.score, n_boot=50, n_jobs=1, plus_minus="std"
        )
        half = 100 * (ci["roc_auc_high"] - ci["roc_auc_low"]) / 2
        self.assertEqual(
            ci["roc_auc_text"], f"{100 * ci['roc_auc']:.1f}±{half:.1f}%"
        )
        self.assertEqual(
            std["roc_auc_text"],
            f"{100 * std['roc_auc']:.1f}±{100 * std['roc_auc_boot_std']:.1f}%",
        )

    def test_average_precision(self):
        result = bootstrap_confidence_intervals(
            self.subject_ids,
            self.y,
            self.score,
            n_boot=20,
            n_jobs=1,
            pr_auc="average_precision",
        )
        self.assertAlmostEqual(
            result["pr_auc"], compute_metrics(self.y, self.score, None, "average_precision")[1]
        )
        self.assertEqual(result["pr_auc_kind"], "average_precision")

    def test_single_class_raises(self):
        with self.assertRaises(ValueError):
            bootstrap_confidence_intervals(
                self.subject_ids, np.zeros_like(self.y), self.score, n_boot=5, n_jobs=1
            )


class TestReadPredictions(unittest.TestCase):
    def test_reads_folders_recursively_and_skips_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, subjects in (("a.parquet", [1, 2]), ("shard_0/b.parquet", [3, 4])):
                path = Path(tmp) / name
                path.parent.mkdir(exist_ok=True)
                pl.DataFrame(
                    {
                        "subject_id": subjects,
                        "boolean_value": [True, False],
                        "predicted_boolean_probability": [0.9, None],
                        "extra": ["x", "y"],
                    }
                ).write_parquet(path)
            subject_ids, y, score = read_predictions([tmp])
        self.assertEqual(sorted(subject_ids.tolist()), [1, 3])
        self.assertEqual(y.tolist(), [1, 1])
        self.assertEqual(score.tolist(), [0.9, 0.9])


if __name__ == "__main__":
    unittest.main()
