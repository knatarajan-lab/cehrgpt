import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cehrgpt.time_to_event import time_to_event_prediction as tte


def make_predictions(num_patients=60, seed=1):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, num_patients).astype(bool)
    return pd.DataFrame(
        {
            "subject_id": np.arange(num_patients),
            "boolean_value": label,
            "predicted_boolean_probability": np.clip(
                0.5 * label + rng.uniform(0, 0.6, num_patients), 0, 1
            ),
        }
    )


class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.task_config = os.path.join(self.tmp, "task.yaml")
        with open(self.task_config, "w") as f:
            f.write('task_name: "toy_task"\noutcome_events: ["1"]\n')
        self.output_folder = os.path.join(self.tmp, "out")
        self.prediction_folder = os.path.join(
            self.output_folder, "top_p10000", "toy_task"
        )
        self.metrics_file = f"{self.prediction_folder}_metrics.json"
        os.makedirs(self.prediction_folder)

    def parse_args(self, *extra):
        return tte.create_arg_parser().parse_args(
            [
                "--tokenizer_folder", "tok",
                "--model_folder", "model",
                "--output_folder", self.output_folder,
                "--dataset_folder", "unused",
                "--num_return_sequences", "2",
                "--task_config", self.task_config,
                "--min_num_of_concepts", "1",
                "--batch_size", "1",
                "--context_window", "128",
                "--sampling_strategy", "TopPStrategy",
                "--top_p", "1.0",
                *extra,
            ]
        )  # fmt: skip


class TestWritePredictionMetrics(MetricsTestCase):
    def test_covers_all_saved_predictions_of_a_resumed_run(self):
        predictions = make_predictions()
        # An earlier session without prefix_values, and a --gpu_ids shard with it
        predictions.iloc[:30].to_parquet(
            os.path.join(self.prediction_folder, "earlier.parquet")
        )
        shard_folder = os.path.join(self.prediction_folder, "shard_0")
        os.makedirs(shard_folder)
        predictions.iloc[30:].assign(
            prefix_values=[["BIN:1"]] * 30
        ).to_parquet(os.path.join(shard_folder, "later.parquet"))

        tte.write_prediction_metrics(
            self.parse_args("--n_boot", "50"), self.prediction_folder
        )

        with open(self.metrics_file) as f:
            result = json.load(f)
        self.assertEqual(result["n_labels"], 60)
        self.assertEqual(result["n_boot"], 50)
        self.assertAlmostEqual(
            result["roc_auc"],
            roc_auc_score(
                predictions["boolean_value"],
                predictions["predicted_boolean_probability"],
            ),
        )
        self.assertLessEqual(result["roc_auc_low"], result["roc_auc_high"])
        self.assertLessEqual(result["pr_auc_low"], result["pr_auc_high"])
        # The prediction folder keeps only parquet files
        self.assertEqual(
            sorted(os.listdir(self.prediction_folder)), ["earlier.parquet", "shard_0"]
        )

    def test_alpha_sets_the_width_of_the_interval(self):
        make_predictions().to_parquet(
            os.path.join(self.prediction_folder, "a.parquet")
        )
        widths = []
        for alpha in ("0.05", "0.5"):
            tte.write_prediction_metrics(
                self.parse_args("--n_boot", "100", "--alpha", alpha),
                self.prediction_folder,
            )
            with open(self.metrics_file) as f:
                result = json.load(f)
            widths.append(result["roc_auc_high"] - result["roc_auc_low"])
        self.assertGreater(widths[0], widths[1])

    def test_zero_replicates_skips_the_metrics(self):
        make_predictions().to_parquet(
            os.path.join(self.prediction_folder, "a.parquet")
        )
        tte.write_prediction_metrics(
            self.parse_args("--n_boot", "0"), self.prediction_folder
        )
        self.assertFalse(os.path.exists(self.metrics_file))

    def test_a_single_class_does_not_fail_the_run(self):
        make_predictions().assign(boolean_value=False).to_parquet(
            os.path.join(self.prediction_folder, "a.parquet")
        )
        tte.write_prediction_metrics(
            self.parse_args("--n_boot", "10"), self.prediction_folder
        )
        self.assertFalse(os.path.exists(self.metrics_file))

    def test_no_predictions_does_not_fail_the_run(self):
        tte.write_prediction_metrics(
            self.parse_args("--n_boot", "10"), self.prediction_folder
        )
        self.assertFalse(os.path.exists(self.metrics_file))


class TestMainComputesMetricsOnce(MetricsTestCase):
    def run_main(self, args):
        tokenizer = mock.MagicMock(vocab_size=7)
        tokenizer.get_vocab.return_value = {}
        tokenizer.get_value_vocab.return_value = {}
        generation_config = mock.Mock(max_length=1024, max_new_tokens=128)
        with mock.patch.object(
            tte.CehrGptTokenizer, "from_pretrained", return_value=tokenizer
        ), mock.patch.object(
            tte.CEHRGPT2LMHeadModel, "from_pretrained"
        ), mock.patch.object(
            tte, "read_backbone"
        ), mock.patch.object(
            tte, "resolve_attn_implementation"
        ), mock.patch.object(
            tte, "TimeToEventModel"
        ) as model, mock.patch.object(
            tte, "prepare_test_dataset", return_value=[]
        ), mock.patch.object(
            tte, "load_from_disk", return_value=[]
        ), mock.patch.object(
            tte, "resolve_gpu_ids", return_value=["0", "1"]
        ), mock.patch.object(
            tte, "run_on_multiple_gpus", return_value=0
        ), mock.patch.object(
            tte, "write_prediction_metrics"
        ) as write_metrics:
            model.get_generation_config.return_value = generation_config
            tte.main(args)
        return write_metrics

    def test_single_process_run(self):
        args = self.parse_args()
        self.run_main(args).assert_called_once_with(args, self.prediction_folder)

    def test_multi_gpu_parent(self):
        args = self.parse_args("--gpu_ids", "0,1")
        self.run_main(args).assert_called_once_with(args, self.prediction_folder)

    def test_multi_gpu_worker_leaves_it_to_the_parent(self):
        shard = os.path.join(self.tmp, "shards", "shard_0")
        args = self.parse_args("--prepared_dataset_path", shard)
        self.run_main(args).assert_not_called()


if __name__ == "__main__":
    unittest.main()
