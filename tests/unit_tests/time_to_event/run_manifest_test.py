import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from cehrgpt.time_to_event import run_manifest
from cehrgpt.time_to_event import time_to_event_prediction as tte


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


class RunManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.model = os.path.join(self.tmp, "model")
        self.tokenizer = os.path.join(self.tmp, "tokenizer")
        self.dataset = os.path.join(self.tmp, "tokenized")
        self.cohort = os.path.join(self.tmp, "cohort")
        write(os.path.join(self.model, "model.safetensors"), b"weights")
        write(os.path.join(self.model, "config.json"), b"{}")
        write(os.path.join(self.tokenizer, "cehrgpt_tokenizer.json"), b"tokens")
        # The prepared datasets live in the tokenizer folder and are not the tokenizer
        write(os.path.join(self.tokenizer, "dataset_prepared", "big.arrow"), b"x" * 50)
        write(os.path.join(self.dataset, "dataset_dict.json"), b"{}")
        write(os.path.join(self.dataset, "test", "data-00000.arrow"), b"rows")
        write(
            os.path.join(self.dataset, "test", "state.json"),
            json.dumps({"_fingerprint": "abc123"}).encode(),
        )
        write(os.path.join(self.cohort, "part", "data.parquet"), b"cohort")
        self.task_config = os.path.join(self.tmp, "task.yaml")
        with open(self.task_config, "w") as f:
            f.write('task_name: "toy_task"\noutcome_events: ["1"]\nmax_new_tokens: 128\n')
        self.output_folder = os.path.join(self.tmp, "out")
        self.manifest_file = os.path.join(
            self.output_folder, "top_p10000", "toy_task_manifest.json"
        )

    def parse_args(self, *extra):
        return tte.create_arg_parser().parse_args(
            [
                "--tokenizer_folder", self.tokenizer,
                "--model_folder", self.model,
                "--output_folder", self.output_folder,
                "--tokenized_full_dataset_path", self.dataset,
                "--cohort_folder", self.cohort,
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

    def write_manifest(self, *extra):
        with mock.patch.object(tte.CehrGptTokenizer, "from_pretrained"):
            tte.write_prediction_run_manifest(self.parse_args(*extra))
        with open(self.manifest_file) as f:
            return json.load(f)["sessions"]

    def test_the_manifest_says_what_the_run_read(self):
        (session,) = self.write_manifest()
        self.assertEqual(session["model"]["path"], self.model)
        self.assertEqual(
            session["model"]["files"]["model.safetensors"]["sha256"],
            hashlib.sha256(b"weights").hexdigest(),
        )
        self.assertEqual(session["tokenizer"]["path"], self.tokenizer)
        self.assertEqual(
            list(session["tokenizer"]["files"]), ["cehrgpt_tokenizer.json"]
        )
        self.assertEqual(session["tokenized_dataset"]["path"], self.dataset)
        test_split = session["tokenized_dataset"]["splits"]["test"]
        self.assertEqual(test_split["fingerprint"], "abc123")
        self.assertEqual(test_split["num_files"], 2)
        # A tokenized dataset is too large to hash
        self.assertNotIn("sha256", json.dumps(session["tokenized_dataset"]))
        self.assertEqual(session["cohort"]["path"], self.cohort)
        self.assertEqual(
            session["cohort"]["files"][os.path.join("part", "data.parquet")]["sha256"],
            hashlib.sha256(b"cohort").hexdigest(),
        )
        self.assertIsNone(session["dataset"])
        self.assertEqual(session["arguments"]["batch_size"], 1)
        self.assertIn("cehrgpt_folder", session["code"])
        self.assertIn("torch", session["code"]["versions"])

    def test_the_task_config_is_the_one_after_the_overrides(self):
        (session,) = self.write_manifest(
            "--task_config_override", "max_new_tokens=192"
        )
        self.assertEqual(session["task_config"]["max_new_tokens"], 192)
        self.assertEqual(session["task_config"]["task_name"], "toy_task")

    def test_a_resumed_run_adds_a_session(self):
        self.write_manifest()
        write(os.path.join(self.model, "model.safetensors"), b"other weights")
        first, second = self.write_manifest()
        self.assertNotEqual(
            first["model"]["files"]["model.safetensors"]["sha256"],
            second["model"]["files"]["model.safetensors"]["sha256"],
        )

    def test_a_missing_folder_is_recorded_as_missing(self):
        description = run_manifest.describe_folder(os.path.join(self.tmp, "nowhere"))
        self.assertEqual(description["exists"], False)
        self.assertIsNone(run_manifest.describe_folder(None))

    def test_the_commit_is_only_of_a_checkout_the_code_is_tracked_in(self):
        code = run_manifest.describe_code()
        if "git_commit" in code:
            self.assertEqual(len(code["git_commit"]), 40)
            self.assertIn("git_uncommitted_changes", code)

    def test_a_worker_of_a_multi_gpu_run_writes_no_manifest(self):
        args = self.parse_args("--prepared_dataset_path", "shard")
        with mock.patch.object(tte, "write_prediction_run_manifest") as write_manifest:
            with mock.patch.object(
                tte.CehrGptTokenizer, "from_pretrained", side_effect=RuntimeError("stop")
            ):
                with self.assertRaises(RuntimeError):
                    tte.main(args)
        write_manifest.assert_not_called()

    def test_a_run_writes_the_manifest_before_anything_else(self):
        args = self.parse_args()
        with mock.patch.object(
            tte, "write_prediction_run_manifest", side_effect=RuntimeError("stop")
        ) as write_manifest:
            with self.assertRaises(RuntimeError):
                tte.main(args)
        write_manifest.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
