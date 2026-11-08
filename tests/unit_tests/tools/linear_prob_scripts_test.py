import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
EVALUATION = SCRIPTS / "run_meds_evaluation.sh"
LINEAR_PROB = SCRIPTS / "run_linear_prob.sh"

# Stands for meds-evaluation-cli: records the call and writes a results file where output_dir says
STUB = """#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    predictions_path=*) predictions="${arg#*=}" ;;
    output_dir=*) output="${arg#*=}" ;;
  esac
done
echo "$predictions $output" >> "$STUB_LOG"
case "$predictions" in *fail*) exit 3 ;; esac
mkdir -p "$output"
echo '{"roc_auc": 0.5}' > "$output/results.json"
"""


class EvaluationScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "meds-evaluation-cli"
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.log = self.root / "calls.log"
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_LOG": str(self.log)}

    def predictions(self, *parts: str) -> Path:
        folder = self.root.joinpath(*parts)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "predictions.parquet").write_bytes(b"x")
        return folder

    def run_script(self, *args):
        return subprocess.run(["sh", str(EVALUATION), *args], capture_output=True, text=True, env=self.env)

    def calls(self):
        return sorted(self.log.read_text().splitlines()) if self.log.exists() else []

    def test_a_result_folder_with_test_predictions(self):
        self.predictions("cohort", "model", "logistic_global", "test_predictions")
        folder = self.root / "cohort" / "model" / "logistic_global"
        result = self.run_script(str(folder))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [f"{folder}/test_predictions {folder}/"])
        self.assertTrue((folder / "results.json").exists())
        self.assertIn('"roc_auc": 0.5', result.stdout)

    def test_any_folder_of_predictions(self):
        folder = self.predictions("zero_shot", "predictions")
        result = self.run_script(str(folder) + "/")
        self.assertEqual(result.returncode, 0, result.stderr)
        # not named test_predictions: the results go in a folder next to it
        self.assertEqual(self.calls(), [f"{folder} {folder}_evaluation/"])

    def test_several_folders_and_an_explicit_output_dir(self):
        first = self.predictions("a", "test_predictions")
        second = self.predictions("b", "preds")
        self.assertEqual(self.run_script(str(first), str(second)).returncode, 0)
        self.assertEqual(len(self.calls()), 2)
        again = self.run_script("--output_dir=" + str(self.root / "out"), str(first.parent))
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn(f"{first} {self.root / 'out'}/", self.calls())
        two = self.run_script("--output_dir=" + str(self.root / "out"), str(first), str(second))
        self.assertNotEqual(two.returncode, 0)
        self.assertIn("single predictions folder", two.stderr)

    def test_recursive_finds_every_predictions_folder(self):
        for cohort in ("c1", "c2"):
            for variant in ("logistic", "logistic_global", "logistic_local"):
                self.predictions("outputs", cohort, "model", variant, "test_predictions")
        self.predictions("outputs", "other", "elsewhere")  # a folder with predictions.parquet in it
        result = self.run_script("--recursive", str(self.root / "outputs"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 7)
        for cohort in ("c1", "c2"):
            for variant in ("logistic", "logistic_global", "logistic_local"):
                self.assertTrue((self.root / "outputs" / cohort / "model" / variant / "results.json").exists())

    def test_a_failure_is_reported_and_the_others_still_run(self):
        good = self.predictions("good", "test_predictions")
        bad = self.predictions("fail_me", "test_predictions")
        result = self.run_script(str(bad), str(good))
        self.assertEqual(result.returncode, 1)
        self.assertIn("meds-evaluation failed", result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue((good.parent / "results.json").exists())

    def test_a_folder_without_predictions_is_an_error(self):
        empty = self.root / "empty"
        empty.mkdir()
        result = self.run_script(str(empty))
        self.assertEqual(result.returncode, 1)
        self.assertIn("no predictions", result.stderr)
        self.assertFalse(self.log.exists())
        self.assertEqual(self.run_script(str(self.root / "missing")).returncode, 1)
        self.assertNotEqual(self.run_script().returncode, 0)  # usage


class LinearProbScriptTest(unittest.TestCase):
    """The functions of run_linear_prob.sh that decide which results a cohort needs."""

    @classmethod
    def setUpClass(cls):
        text = LINEAR_PROB.read_text()
        start = text.index("# The feature sets, as the words that train_with_cehrgpt_features takes")
        end = text.index("# Log file setup")
        cls.functions = text[start:end]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def sh(self, body: str, feature_sets: str = "", disable_combine: str = "false"):
        script = (
            f'FEATURE_SETS="{feature_sets}"\nDISABLE_COMBINE_GLOBAL_LOCAL="{disable_combine}"\n'
            f"{self.functions}\n{body}"
        )
        return subprocess.run(["sh", "-c", script], capture_output=True, text=True)

    def info(self, layout, global_is_last=False):
        (self.out / "feature_info.json").write_text(
            json.dumps({"layout": layout, "hidden_size": 4, "num_features": 4 * len(layout),
                        "global_is_last_hidden_state": global_is_last}, indent=2)
        )

    def required(self, **kwargs):
        result = self.sh(f'required_result_dirs "{self.out}"', **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.split()

    def test_the_default_asks_for_all_three_when_the_features_were_combined(self):
        self.info(["global", "local"])
        self.assertEqual(self.required(), ["logistic", "logistic_global", "logistic_local"])
        self.assertEqual(self.required(feature_sets="all"), ["logistic", "logistic_global", "logistic_local"])

    def test_without_a_motor_head_the_local_features_are_not_trained_again(self):
        self.info(["global", "local"], global_is_last=True)
        self.assertEqual(self.required(), ["logistic", "logistic_global"])

    def test_features_that_were_not_combined_or_not_extracted_yet_are_one_model(self):
        self.assertEqual(self.required(), ["logistic"])  # nothing extracted yet
        self.info(["global"])
        self.assertEqual(self.required(), ["logistic"])
        self.assertEqual(self.required(feature_sets="global"), ["logistic"])

    def test_the_sets_asked_for_by_name(self):
        self.info(["global", "local"])
        self.assertEqual(self.required(feature_sets="local"), ["logistic_local"])
        self.assertEqual(self.required(feature_sets="global,concatenated"), ["logistic_global", "logistic"])
        self.assertEqual(self.required(feature_sets="local global"), ["logistic_local", "logistic_global"])

    def test_results_are_complete_only_when_every_required_folder_has_its_metrics(self):
        self.info(["global", "local"])
        check = f'results_complete "{self.out}"'
        self.assertEqual(self.sh(check).returncode, 1)
        for folder in ("logistic", "logistic_global"):
            (self.out / folder).mkdir()
            (self.out / folder / "metrics.json").write_text("{}")
        self.assertEqual(self.sh(check).returncode, 1)  # local is missing
        self.assertEqual(self.sh(check, feature_sets="concatenated,global").returncode, 0)
        (self.out / "logistic_local").mkdir()
        (self.out / "logistic_local" / "metrics.json").write_text("{}")
        self.assertEqual(self.sh(check).returncode, 0)

    def test_results_of_before_the_option_are_complete_with_just_logistic(self):
        (self.out / "logistic").mkdir()
        (self.out / "logistic" / "metrics.json").write_text("{}")
        self.assertEqual(self.sh(f'results_complete "{self.out}"').returncode, 0)

    def test_unknown_sets_and_sets_that_need_the_combined_features_are_rejected_up_front(self):
        bad = self.sh("echo reached", feature_sets="global,both")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("unknown feature set 'both'", bad.stdout)
        self.assertNotIn("reached", bad.stdout)
        for name in ("local", "concatenated"):
            rejected = self.sh("echo reached", feature_sets=name, disable_combine="true")
            self.assertEqual(rejected.returncode, 1, name)
            self.assertIn("need the global and local features combined", rejected.stdout)
        self.assertIn("reached", self.sh("echo reached", feature_sets="global", disable_combine="true").stdout)

    def test_the_option_is_documented_and_parsed(self):
        text = LINEAR_PROB.read_text()
        self.assertIn("--feature_sets=LIST", text)
        self.assertIsNotNone(re.search(r"--feature_sets=\*\)\s+FEATURE_SETS=", text))
        self.assertIn("--feature_sets ${FEATURE_SETS_ARGS:-all}", text)


if __name__ == "__main__":
    unittest.main()


# Stands for python: records the training call and writes what the training writes
PYTHON_STUB = """#!/bin/sh
echo "$@" >> "$STUB_LOG"
case "$*" in
  *compute_cehrgpt_features*) echo "EXTRACTION" >> "$STUB_LOG"; exit 0 ;;
esac
while [ $# -gt 0 ]; do
  case "$1" in --output_dir) out="$2"; shift ;; esac
  shift
done
mkdir -p "$out/logistic/test_predictions"
echo '{"roc_auc": 0.7, "pr_auc": 0.6}' > "$out/logistic/metrics.json"
touch "$out/logistic/test_predictions/predictions.parquet"
"""


class SkipFeatureExtractionTest(unittest.TestCase):
    """run_linear_prob.sh --skip_feature_extraction trains on the features that are there."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name, text in (("python", PYTHON_STUB), ("meds-evaluation-cli", STUB)):
            path = bin_dir / name
            path.write_text(text)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.log = self.root / "calls.log"
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_LOG": str(self.log)}
        self.out = self.root / "results"
        self.out.mkdir()

    def features(self, cohort: str, model: str = "my_model"):
        folder = self.out / cohort / model / "features_with_label"
        for split in ("train", "test"):
            (folder / f"{split}_features").mkdir(parents=True)
            (folder / f"{split}_features" / "part.parquet").write_bytes(b"x")
        return self.out / cohort / model

    def run_script(self, *args):
        return subprocess.run(
            ["sh", str(LINEAR_PROB), "--skip_feature_extraction", f"--output_dir={self.out}", *args],
            capture_output=True, text=True, env=self.env,
        )

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_only_the_output_directory_is_needed_and_nothing_is_extracted(self):
        readmission = self.features("readmission")
        mortality = self.features("mortality")
        result = self.run_script("--model_name=my_model")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertNotIn("EXTRACTION", calls)
        self.assertFalse(any("compute_cehrgpt_features" in c for c in calls))
        training = [c for c in calls if "train_with_cehrgpt_features" in c]
        self.assertEqual(len(training), 2)
        for cohort in (readmission, mortality):
            self.assertTrue(any(f"--features_data_dir {cohort}" in c and f"--output_dir {cohort}" in c for c in training))
            self.assertTrue((cohort / "logistic" / "metrics.json").exists())
            self.assertTrue((cohort / "logistic" / "results.json").exists())  # evaluated
        self.assertTrue(all("--feature_sets all" in c for c in training))
        self.assertIn("Skipping the feature extraction", result.stdout)

    def test_cohorts_without_the_features_of_the_model_are_left_out(self):
        self.features("readmission")
        self.features("other_model_only", model="another_model")
        (self.out / "empty").mkdir()
        result = self.run_script("--model_name=my_model")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        training = [c for c in self.calls() if "train_with_cehrgpt_features" in c]
        self.assertEqual(len(training), 1)
        self.assertIn("readmission", training[0])
        self.assertIn("Skipping other_model_only: no features of my_model", result.stdout)
        self.assertNotIn("Found 0 cohorts", result.stdout)

    def test_the_feature_sets_and_finished_cohorts(self):
        done = self.features("done")
        (done / "logistic").mkdir()
        (done / "logistic" / "metrics.json").write_text("{}")
        self.features("todo")
        result = self.run_script("--model_name=my_model", "--feature_sets=concatenated")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        training = [c for c in self.calls() if "train_with_cehrgpt_features" in c]
        self.assertEqual(len(training), 1)  # the finished cohort is not trained again
        self.assertIn("--feature_sets concatenated", training[0])
        self.assertIn("Skipping cohort done", result.stdout)

    def test_no_features_at_all_is_an_error(self):
        result = self.run_script("--model_name=my_model")
        self.assertEqual(result.returncode, 1)
        self.assertIn("No valid cohorts", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_without_the_option_the_other_arguments_are_still_required(self):
        result = subprocess.run(
            ["sh", str(LINEAR_PROB), f"--output_dir={self.out}"], capture_output=True, text=True, env=self.env
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required arguments", result.stdout)
        missing_output = subprocess.run(
            ["sh", str(LINEAR_PROB), "--skip_feature_extraction"], capture_output=True, text=True, env=self.env
        )
        self.assertNotEqual(missing_output.returncode, 0)
        self.assertIn("--output_dir is required", missing_output.stdout)
