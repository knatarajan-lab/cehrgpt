import datetime
import os
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd
from datasets import Dataset, load_from_disk

from cehrgpt.time_to_event import time_to_event_prediction as tte

DAY = 24 * 3600
START = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
NUM_PERSONS = 10


class TestResolveGpuIds(unittest.TestCase):
    def test_explicit_ids_map_through_cuda_visible_devices(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3,5"}):
            self.assertEqual(tte.resolve_gpu_ids("0,2"), ["1", "5"])
            self.assertEqual(tte.resolve_gpu_ids("all"), ["1", "3", "5"])

    def test_ids_without_cuda_visible_devices(self):
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            tte.torch.cuda, "device_count", return_value=4
        ):
            self.assertEqual(tte.resolve_gpu_ids("all"), ["0", "1", "2", "3"])
            self.assertEqual(tte.resolve_gpu_ids("3,1"), ["3", "1"])

    def test_invalid_ids_are_rejected(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}):
            with self.assertRaises(ValueError):
                tte.resolve_gpu_ids("0,2")
            with self.assertRaises(ValueError):
                tte.resolve_gpu_ids("0,0")

    def test_no_visible_gpu(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), mock.patch.object(
            tte.torch.cuda, "device_count", return_value=0
        ):
            with self.assertRaises(RuntimeError):
                tte.resolve_gpu_ids("all")


class TestStripCliOption(unittest.TestCase):
    def test_strips_both_spellings(self):
        argv = ["--a", "1", "--gpu_ids", "0,1", "--b", "2", "--gpu_ids=all", "--c"]
        self.assertEqual(
            tte.strip_cli_option(argv, "--gpu_ids"), ["--a", "1", "--b", "2", "--c"]
        )


class TestRunOnMultipleGpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.task_config = os.path.join(self.tmp, "task.yaml")
        with open(self.task_config, "w") as f:
            f.write('task_name: "toy_task"\noutcome_events: ["1"]\n')
        self.output_folder = os.path.join(self.tmp, "out")
        self.args = tte.create_arg_parser().parse_args(
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
                "--gpu_ids", "0,1",
            ]
        )  # fmt: skip
        self.prediction_folder = os.path.join(
            self.output_folder, "top_p10000", "toy_task"
        )
        self.dataset = Dataset.from_dict(
            {
                "person_id": list(range(NUM_PERSONS)),
                "index_date": [START] * NUM_PERSONS,
                "num_of_concepts": [5] * NUM_PERSONS,
                "label": [0] * NUM_PERSONS,
            }
        )

    def _run(self, gpu_ids, exit_code=0):
        popen_calls = []
        shard_contents = []

        def fake_popen(cmd, env, stdout, stderr):
            shard_path = cmd[cmd.index("--prepared_dataset_path") + 1]
            shard_contents.append(sorted(load_from_disk(shard_path)["person_id"]))
            popen_calls.append((cmd, env))
            process = mock.Mock()
            process.wait.return_value = exit_code
            process.poll.return_value = exit_code
            return process

        argv = ["prog", "--task_config", self.task_config, "--gpu_ids", "0,1"]
        with mock.patch.object(
            tte.CehrGptTokenizer, "from_pretrained", return_value=mock.Mock(vocab_size=7)
        ), mock.patch.object(
            tte, "load_time_to_event_dataset", return_value=self.dataset
        ), mock.patch.object(
            tte.subprocess, "Popen", side_effect=fake_popen
        ), mock.patch.object(
            tte.sys, "argv", argv
        ):
            code = tte.run_on_multiple_gpus(self.args, gpu_ids)
        return code, popen_calls, shard_contents

    def test_pending_samples_are_split_into_disjoint_shards_one_per_gpu(self):
        code, calls, shards = self._run(["2", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for _, env in calls], ["2", "3"])
        self.assertEqual(sorted(sum(shards, [])), list(range(NUM_PERSONS)))
        self.assertFalse(set(shards[0]) & set(shards[1]))
        for cmd, _ in calls:
            self.assertNotIn("--gpu_ids", cmd)
        # Shards are cleaned up after a successful run
        self.assertFalse(
            os.path.exists(os.path.join(self.output_folder, "top_p10000", "temp"))
        )

    def test_samples_predicted_in_a_previous_session_are_skipped(self):
        shard_folder = os.path.join(self.prediction_folder, "shard_0")
        os.makedirs(shard_folder)
        prediction_time = datetime.datetime(2020, 1, 1)
        pd.DataFrame(
            {
                "subject_id": [0, 1, 2],
                "prediction_time": [prediction_time] * 3,
            }
        ).to_parquet(os.path.join(shard_folder, "done.parquet"))
        code, _, shards = self._run(["0", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(sum(shards, [])), list(range(3, NUM_PERSONS)))

    def test_nothing_is_launched_when_everything_is_done(self):
        os.makedirs(self.prediction_folder)
        pd.DataFrame(
            {
                "subject_id": list(range(NUM_PERSONS)),
                "prediction_time": [datetime.datetime(2020, 1, 1)] * NUM_PERSONS,
            }
        ).to_parquet(os.path.join(self.prediction_folder, "done.parquet"))
        code, calls, _ = self._run(["0", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])

    def test_failed_worker_returns_non_zero_and_keeps_the_shards(self):
        code, _, _ = self._run(["0", "1"], exit_code=1)
        self.assertEqual(code, 1)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_folder, "top_p10000", "temp", "shards")
            )
        )


if __name__ == "__main__":
    unittest.main()
