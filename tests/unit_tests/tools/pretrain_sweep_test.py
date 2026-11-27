import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn
from transformers import Trainer, TrainingArguments

from cehrgpt.runners.hf_cehrgpt_pretrain_runner import StopAtStepCallback
from cehrgpt.tools import pretrain_sweep
from cehrgpt.tools.pretrain_sweep import (
    Progress,
    build_trials,
    done_step,
    main,
    run_trial,
    run_trials,
    summarize,
    write_trial_config,
)

BASE_CONFIG = {
    "tokenizer_name_or_path": "/models/tokenizer",
    "model_name_or_path": "/models/base",
    "output_dir": "/models/base",
    "learning_rate": 0.00005,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "use_early_stopping": True,
    "num_train_epochs": 20,
}


def write_trainer_state(sweep_dir, trial_name, global_step, evals=()):
    model_dir = sweep_dir / trial_name / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    history = [{"loss": 3.0, "step": 10}] + [
        dict(entry, step=step) for step, entry in evals
    ]
    with open(model_dir / "trainer_state.json", "w") as f:
        json.dump({"global_step": global_step, "log_history": history}, f)


class TestTrials(unittest.TestCase):
    def test_the_grid_is_every_combination(self):
        trials = build_trials(
            ["learning_rate=1e-4,2e-4", "max_grad_norm=1.0,2,null"]
        )
        self.assertEqual(len(trials), 6)
        self.assertEqual(
            trials["learning_rate-0.0001__max_grad_norm-2"],
            {"learning_rate": 1e-4, "max_grad_norm": 2},
        )
        # 1e-4 is a number, not the string YAML would read it as
        self.assertIsInstance(
            trials["learning_rate-0.0002__max_grad_norm-1.0"]["learning_rate"], float
        )
        self.assertIsNone(
            trials["learning_rate-0.0001__max_grad_norm-None"]["max_grad_norm"]
        )

    def test_no_params_is_the_base_config(self):
        self.assertEqual(build_trials([]), {"base": {}})

    def test_a_param_without_values_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "name=v1,v2"):
            build_trials(["learning_rate"])


class TestTrialConfig(unittest.TestCase):
    def setUp(self):
        self.sweep_dir = Path(tempfile.mkdtemp())

    def write(self, overrides):
        path = write_trial_config(
            BASE_CONFIG, overrides, self.sweep_dir, "trial", 28000, 14000
        )
        with open(path) as f:
            return yaml.safe_load(f)

    def test_the_overrides_and_the_trial_folders_are_set(self):
        config = self.write({"learning_rate": 2e-4})
        model_dir = str(self.sweep_dir / "trial" / "model")
        self.assertEqual(config["learning_rate"], 2e-4)
        self.assertEqual(config["output_dir"], model_dir)
        # the trial starts from scratch, not from the base model
        self.assertEqual(config["model_name_or_path"], model_dir)
        self.assertEqual(config["tokenizer_name_or_path"], "/models/tokenizer")
        self.assertEqual(config["weight_decay"], 0.01)

    def test_the_trials_stop_and_evaluate_at_the_same_steps(self):
        config = self.write({})
        self.assertEqual(config["stop_at_step"], 28000)
        self.assertEqual((config["eval_steps"], config["save_steps"]), (14000, 14000))
        self.assertEqual(config["evaluation_strategy"], "steps")
        self.assertEqual(config["save_strategy"], "steps")
        self.assertFalse(config["use_early_stopping"])
        # the schedule of the full run is kept
        self.assertEqual(config["num_train_epochs"], 20)

    def test_an_unknown_field_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "learning_rates"):
            self.write({"learning_rates": 2e-4})

    def test_a_trainer_default_can_be_swept(self):
        self.assertEqual(self.write({"max_grad_norm": 2.0})["max_grad_norm"], 2.0)


class TestRunTrials(unittest.TestCase):
    def test_each_gpu_runs_one_trial_at_a_time(self):
        lock = threading.Lock()
        running = {"0": 0, "1": 0}
        max_running = {"0": 0, "1": 0}

        def run(trial_name, gpu_id):
            with lock:
                running[gpu_id] += 1
                max_running[gpu_id] = max(max_running[gpu_id], running[gpu_id])
            time.sleep(0.05)
            with lock:
                running[gpu_id] -= 1
            return f"done on {gpu_id}"

        statuses = run_trials(["a", "b", "c", "d", "e"], ["0", "1"], run)
        self.assertEqual(set(statuses), {"a", "b", "c", "d", "e"})
        self.assertEqual(max_running, {"0": 1, "1": 1})
        self.assertEqual({s for s in statuses.values()}, {"done on 0", "done on 1"})

    def test_a_failing_trial_does_not_stop_the_others(self):
        def run(trial_name, gpu_id):
            if trial_name == "b":
                raise RuntimeError("out of memory")
            return "done"

        statuses = run_trials(["a", "b", "c"], ["0"], run)
        self.assertEqual(
            statuses, {"a": "done", "b": "error: out of memory", "c": "done"}
        )


class FakeTraining(unittest.TestCase):
    """Replaces the pretrain runner with one that writes the trainer state at a step."""

    def setUp(self):
        self.sweep_dir = Path(tempfile.mkdtemp())
        self.calls = []
        self.reaches = None  # the step a training reaches, the stop step if None
        self.return_code = 0
        self.original = pretrain_sweep.run_logged

        def run_logged(command, log_file, gpu_id):
            config_path = Path(command[-1])
            with open(config_path) as f:
                config = yaml.safe_load(f)
            trial_name = config_path.parent.name
            self.calls.append((trial_name, gpu_id))
            step = self.reaches if self.reaches is not None else config["stop_at_step"]
            write_trainer_state(self.sweep_dir, trial_name, step)
            return self.return_code

        pretrain_sweep.run_logged = run_logged

    def tearDown(self):
        pretrain_sweep.run_logged = self.original

    def run_trial(self, stop_at_step, trial_name="trial"):
        config_path = write_trial_config(
            BASE_CONFIG, {"learning_rate": 2e-4}, self.sweep_dir, trial_name,
            stop_at_step, stop_at_step,
        )
        return run_trial(
            self.sweep_dir, trial_name, {"learning_rate": 2e-4}, config_path,
            stop_at_step, "1", Progress(self.sweep_dir),
        )

    def progress(self):
        return Progress(self.sweep_dir).read()


class TestRunTrial(FakeTraining):
    def test_a_trained_trial_is_marked_done(self):
        self.assertEqual(self.run_trial(28000), "done")
        self.assertEqual(self.calls, [("trial", "1")])
        self.assertEqual(done_step(self.sweep_dir, "trial"), 28000)
        with open(self.sweep_dir / "trial" / "DONE") as f:
            self.assertEqual(json.load(f)["overrides"], {"learning_rate": 2e-4})
        progress = self.progress()["trial"]
        self.assertEqual(progress["status"], "done")
        self.assertEqual(progress["done_step"], 28000)
        self.assertEqual(progress["overrides"], {"learning_rate": 2e-4})

    def test_a_done_trial_is_skipped(self):
        self.run_trial(28000)
        self.assertEqual(self.run_trial(28000), "done already")
        self.assertEqual(self.run_trial(14000), "done already")
        self.assertEqual(len(self.calls), 1)

    def test_a_done_trial_is_trained_further_with_a_later_stop(self):
        self.run_trial(28000)
        self.assertEqual(self.run_trial(56000), "done")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(done_step(self.sweep_dir, "trial"), 56000)

    def test_a_failed_training_is_not_marked_done(self):
        self.return_code = 1
        self.reaches = 5000
        status = self.run_trial(28000)
        self.assertTrue(status.startswith("failed at step 5000 (exit code 1)"))
        self.assertEqual(done_step(self.sweep_dir, "trial"), 0)
        self.assertEqual(self.progress()["trial"]["status"], "failed")
        # it is trained again on the next sweep
        self.return_code, self.reaches = 0, None
        self.assertEqual(self.run_trial(28000), "done")
        self.assertEqual(self.progress()["trial"]["status"], "done")
        self.assertIsNone(self.progress()["trial"]["error"])

    def test_a_training_that_stops_early_is_not_marked_done(self):
        # e.g. the runner stopped for another reason and exited cleanly
        self.reaches = 20000
        self.assertTrue(self.run_trial(28000).startswith("failed at step 20000"))
        self.assertEqual(done_step(self.sweep_dir, "trial"), 0)


class TestIncrementalSweep(FakeTraining):
    def sweep(self, *params, stop_at_step=28000, extra=()):
        base_config = self.sweep_dir / "base.yaml"
        with open(base_config, "w") as f:
            yaml.safe_dump(BASE_CONFIG, f)
        argv = ["--base_config", str(base_config), "--sweep_dir", str(self.sweep_dir)]
        argv += ["--stop_at_step", str(stop_at_step), "--gpu_ids", "0,1"]
        for param in params:
            argv += ["--param", param]
        main(argv + list(extra))

    def test_an_interrupted_trial_picks_up_where_it_left_off(self):
        # killed at step 14000: a checkpoint, no DONE, and still "running" in the progress
        trial = self.sweep_dir / "learning_rate-0.0001"
        (trial / "model" / "checkpoint-14000").mkdir(parents=True)
        Progress(self.sweep_dir).update("learning_rate-0.0001", status="running")

        self.sweep("learning_rate=1e-4")
        self.assertEqual([t for t, _ in self.calls], ["learning_rate-0.0001"])
        # the checkpoint is kept for the runner to resume from
        self.assertTrue((trial / "model" / "checkpoint-14000").is_dir())
        with open(trial / "model.yaml") as f:
            self.assertFalse(yaml.safe_load(f)["overwrite_output_dir"])
        self.assertEqual(self.progress()["learning_rate-0.0001"]["status"], "done")

    def test_overwrite_trains_the_selected_trials_from_scratch(self):
        self.sweep("learning_rate=1e-4,2e-4")
        stale = self.sweep_dir / "learning_rate-0.0002" / "model" / "checkpoint-28000"
        stale.mkdir()
        kept = self.sweep_dir / "learning_rate-0.0001" / "model" / "checkpoint-28000"
        kept.mkdir()

        self.sweep(
            "learning_rate=1e-4,2e-4",
            extra=["--overwrite", "--trials", "learning_rate-0.0002"],
        )
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.calls[-1][0], "learning_rate-0.0002")
        self.assertFalse(stale.exists())
        self.assertTrue(kept.exists())
        self.assertEqual(done_step(self.sweep_dir, "learning_rate-0.0002"), 28000)

    def test_overwrite_and_summarize_only_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.sweep("learning_rate=1e-4", extra=["--overwrite", "--summarize_only"])

    def test_the_sweep_folder_is_never_deleted(self):
        progress = Progress(self.sweep_dir)
        with self.assertRaisesRegex(ValueError, "not a trial folder"):
            pretrain_sweep.reset_trial(self.sweep_dir, "", progress)
        with self.assertRaisesRegex(ValueError, "not a trial folder"):
            pretrain_sweep.reset_trial(self.sweep_dir, "../other", progress)

    def test_only_the_new_trials_are_trained(self):
        self.sweep("learning_rate=1e-4,2e-4")
        self.assertEqual(
            sorted(t for t, _ in self.calls),
            ["learning_rate-0.0001", "learning_rate-0.0002"],
        )
        self.sweep("learning_rate=1e-4,2e-4,4e-4")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.calls[-1][0], "learning_rate-0.0004")
        self.assertEqual(
            {t: p["status"] for t, p in self.progress().items()},
            {
                "learning_rate-0.0001": "done",
                "learning_rate-0.0002": "done",
                "learning_rate-0.0004": "done",
            },
        )


class TestSummarize(unittest.TestCase):
    def test_the_trials_are_ranked_by_the_eval_loss_at_the_latest_step(self):
        sweep_dir = Path(tempfile.mkdtemp())
        evals = lambda loss: {"eval_loss": loss, "token_loss": loss - 1}
        write_trainer_state(sweep_dir, "lr1", 28000, [(14000, evals(3.1)), (28000, evals(3.0))])
        write_trainer_state(sweep_dir, "lr2", 28000, [(14000, evals(3.2)), (28000, evals(2.9))])

        summary = summarize(sweep_dir, ["lr1", "lr2"])
        self.assertEqual(list(summary.step), [28000, 28000, 14000, 14000])
        self.assertEqual(list(summary.trial), ["lr2", "lr1", "lr1", "lr2"])
        self.assertAlmostEqual(summary.iloc[0]["token_loss"], 1.9)

    def test_nothing_to_summarize(self):
        self.assertTrue(summarize(Path(tempfile.mkdtemp()), ["a"]).empty)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x, labels):
        return {"loss": ((self.linear(x).squeeze(-1) - labels) ** 2).mean()}


class TestStopAtStep(unittest.TestCase):
    def train(self, output_dir, stop_at_step, resume, evaluate=True):
        torch.manual_seed(0)
        data = [{"x": torch.randn(2), "labels": torch.randn(())} for _ in range(64)]
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=100,
            per_device_train_batch_size=4,
            learning_rate=0.1,
            lr_scheduler_type="linear",
            logging_steps=1,
            # neither evaluates nor saves before the stop step on its own
            save_strategy="steps",
            save_steps=1000,
            eval_strategy="steps" if evaluate else "no",
            eval_steps=1000,
            report_to="none",
            use_cpu=True,
        )
        trainer = Trainer(
            model=TinyModel(),
            args=args,
            train_dataset=data,
            eval_dataset=data[:8] if evaluate else None,
            callbacks=[StopAtStepCallback(stop_at_step)],
        )
        trainer.train(resume_from_checkpoint=resume)
        return trainer.state

    def test_the_run_stops_with_the_schedule_of_the_full_run_and_continues(self):
        output_dir = tempfile.mkdtemp()
        state = self.train(output_dir, 10, None)
        self.assertEqual(state.global_step, 10)
        self.assertTrue((Path(output_dir) / "checkpoint-10").is_dir())
        # the learning rate decays over 100 steps (it would be 0 at the end of a 10-step schedule)
        last_lr = [h["learning_rate"] for h in state.log_history if "learning_rate" in h][-1]
        self.assertAlmostEqual(last_lr, 0.1 * 90 / 100, places=6)
        self.assertEqual([h["step"] for h in state.log_history if "eval_loss" in h], [10])

        state = self.train(output_dir, 20, True)
        self.assertEqual(state.global_step, 20)
        steps = [h["step"] for h in state.log_history if "loss" in h]
        self.assertEqual(steps, list(range(1, 21)))
        self.assertEqual(
            [h["step"] for h in state.log_history if "eval_loss" in h], [10, 20]
        )

    def test_a_run_without_an_eval_set_stops_without_evaluating(self):
        output_dir = tempfile.mkdtemp()
        state = self.train(output_dir, 10, None, evaluate=False)
        self.assertEqual(state.global_step, 10)
        self.assertTrue((Path(output_dir) / "checkpoint-10").is_dir())


if __name__ == "__main__":
    unittest.main()
