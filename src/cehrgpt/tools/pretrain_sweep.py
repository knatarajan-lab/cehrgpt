"""
Grid sweep of pretraining hyperparameters on short runs, ranked by the eval loss.

Every trial is the pretraining config with some fields overridden. It is trained with
the pretrain runner up to --stop_at_step, keeping the learning rate schedule of the
full run (see StopAtStepCallback), so a short trial is the start of the full run it
stands for. The eval losses of all trials are written to <sweep_dir>/summary.csv.
The trials share the loss terms and weights of the base config, so their eval losses
can be compared; sweeping a loss weight would make the total eval loss incomparable.

Trials are queued over the GPUs, one trial per GPU at a time. The sweep picks up
where it left off: a trial that trains to --stop_at_step gets a DONE file in its
folder holding that step and is skipped when the sweep is run again, and a trial that
was interrupted resumes from its last checkpoint. So the sweep can be rerun after a
failure or with more values added to the grid; --overwrite deletes the trials and
trains them from scratch instead. <sweep_dir>/progress.json holds the status of every
trial (running, done or failed). Rerunning some trials with a larger --stop_at_step
continues them from their checkpoints (successive halving):

    python -m cehrgpt.tools.pretrain_sweep \
        --base_config model.yaml --sweep_dir sweep \
        --param learning_rate=1e-4,2e-4,4e-4 --param max_grad_norm=1.0,2.0 \
        --stop_at_step 28000 --eval_steps 14000 --gpu_ids 0,1

    python -m cehrgpt.tools.pretrain_sweep ... --stop_at_step 56000 \
        --trials learning_rate-0.0002__max_grad_norm-1.0
"""

import argparse
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

EVAL_LOSSES = [
    "eval_loss",
    "token_loss",
    "token_value_loss",
    "time_token_loss",
    "time_to_visit_loss",
    "motor_tte_loss",
    "age_at_vs_loss",
    "year_at_vs_loss",
]

DONE_FILE = "DONE"
PROGRESS_FILE = "progress.json"

# Trainer arguments that are commonly swept but often left at their default in the config
ALLOWED_NEW_FIELDS = {
    "max_grad_norm",
    "warmup_steps",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "lr_scheduler_type",
}


def parse_value(value: str):
    """A number if it is one (YAML reads 1e-4 as a string), otherwise the YAML value."""
    for number_type in (int, float):
        try:
            return number_type(value)
        except ValueError:
            pass
    return yaml.safe_load(value)


def parse_param(spec: str) -> (str, List):
    """Parse ``name=v1,v2`` into the field name and its values."""
    if "=" not in spec:
        raise ValueError(f"--param must be name=v1,v2,..., got {spec!r}")
    name, values = spec.split("=", 1)
    return name, [parse_value(v) for v in values.split(",")]


def build_trials(params: List[str]) -> Dict[str, Dict]:
    """The grid of all value combinations, keyed by a trial name built from the values."""
    grid = [parse_param(spec) for spec in params]
    names = [name for name, _ in grid]
    trials = {}
    for values in itertools.product(*[values for _, values in grid]):
        overrides = dict(zip(names, values))
        # without params, the one trial is the base config
        trial_name = "__".join(f"{k}-{v}" for k, v in overrides.items()) or "base"
        trials[trial_name] = overrides
    return trials


def trial_dir(sweep_dir: Path, trial_name: str) -> Path:
    return sweep_dir / trial_name


def write_trial_config(
    base_config: Dict,
    overrides: Dict,
    sweep_dir: Path,
    trial_name: str,
    stop_at_step: int,
    eval_steps: int,
) -> Path:
    """Write the pretraining config of a trial and return its path."""
    # A typo would otherwise sweep a field that nothing reads
    unknown = set(overrides) - set(base_config) - ALLOWED_NEW_FIELDS
    if unknown:
        raise ValueError(
            f"The fields {sorted(unknown)} are not in the base config. Add them to the "
            "base config with their current value to sweep them."
        )
    directory = trial_dir(sweep_dir, trial_name)
    model_dir = directory / "model"
    config = dict(base_config)
    config.update(overrides)
    config.update(
        {
            # the trial starts from scratch and saves its own model
            "model_name_or_path": str(model_dir),
            "output_dir": str(model_dir),
            "logging_dir": str(directory / "logs"),
            "overwrite_output_dir": False,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "eval_steps": eval_steps,
            "save_steps": eval_steps,
            "stop_at_step": stop_at_step,
            # the trials are compared at the same steps, so none of them stops early
            "use_early_stopping": False,
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "model.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


def trained_steps(sweep_dir: Path, trial_name: str) -> int:
    """The step the trainer saved its state at, 0 if it has not."""
    state_file = trial_dir(sweep_dir, trial_name) / "model" / "trainer_state.json"
    if not state_file.exists():
        return 0
    with open(state_file) as f:
        return json.load(f).get("global_step", 0)


def done_step(sweep_dir: Path, trial_name: str) -> int:
    """The step the trial is done training to, from its DONE file, 0 if it has none."""
    done_file = trial_dir(sweep_dir, trial_name) / DONE_FILE
    if not done_file.exists():
        return 0
    with open(done_file) as f:
        return json.load(f)["stop_at_step"]


def mark_done(sweep_dir: Path, trial_name: str, stop_at_step: int, overrides: Dict):
    with open(trial_dir(sweep_dir, trial_name) / DONE_FILE, "w") as f:
        json.dump({"stop_at_step": stop_at_step, "overrides": overrides}, f, indent=2)


class Progress:
    """The status of every trial in <sweep_dir>/progress.json, kept across sweeps."""

    def __init__(self, sweep_dir: Path):
        self.path = sweep_dir / PROGRESS_FILE
        self.lock = threading.Lock()

    def read(self) -> Dict[str, Dict]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            return json.load(f)

    def update(self, trial_name: str, **fields):
        with self.lock:
            progress = self.read()
            progress.setdefault(trial_name, {}).update(fields)
            self.write(progress)

    def remove(self, trial_name: str):
        with self.lock:
            progress = self.read()
            if progress.pop(trial_name, None) is not None:
                self.write(progress)

    def write(self, progress: Dict[str, Dict]):
        # written to a temporary file first, so an interrupted write leaves the old file
        temporary = self.path.with_suffix(".tmp")
        with open(temporary, "w") as f:
            json.dump(progress, f, indent=2, sort_keys=True)
        temporary.replace(self.path)


def reset_trial(sweep_dir: Path, trial_name: str, progress: Progress):
    """Delete the folder (model, checkpoints, logs, DONE) and the progress of a trial."""
    directory = trial_dir(sweep_dir, trial_name)
    # a trial is a direct subfolder of the sweep, never the sweep folder itself
    if directory.parent != sweep_dir or directory == sweep_dir:
        raise ValueError(f"{directory} is not a trial folder of {sweep_dir}")
    if directory.exists():
        shutil.rmtree(directory)
    progress.remove(trial_name)


def train_command(config_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "cehrgpt.runners.hf_cehrgpt_pretrain_runner",
        str(config_path),
    ]


def run_logged(command: List[str], log_file: Path, gpu_id: str) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id)
    with open(log_file, "a") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        return subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env
        ).returncode


def run_trial(
    sweep_dir: Path,
    trial_name: str,
    overrides: Dict,
    config_path: Path,
    stop_at_step: int,
    gpu_id: str,
    progress: Progress,
) -> str:
    """Train one trial on one GPU up to the stop step, return its status."""
    if done_step(sweep_dir, trial_name) >= stop_at_step:
        return "done already"
    print(f"[gpu {gpu_id}] training {trial_name}", flush=True)
    progress.update(
        trial_name,
        status="running",
        gpu=gpu_id,
        stop_at_step=stop_at_step,
        overrides=overrides,
    )
    log_file = trial_dir(sweep_dir, trial_name) / "train.log"
    return_code = run_logged(train_command(config_path), log_file, gpu_id)
    # the runner exits cleanly when it stops early for another reason, so the step is checked
    reached = trained_steps(sweep_dir, trial_name)
    if return_code or reached < stop_at_step:
        status = f"failed at step {reached} (exit code {return_code}), see {log_file}"
        progress.update(trial_name, status="failed", error=status)
        return status
    mark_done(sweep_dir, trial_name, stop_at_step, overrides)
    progress.update(trial_name, status="done", done_step=stop_at_step, error=None)
    return "done"


def run_trials(trial_names: List[str], gpu_ids: List[str], run) -> Dict[str, str]:
    """Run the trials over the GPUs, one trial per GPU at a time, return their statuses."""
    pending = queue.Queue()
    for trial_name in trial_names:
        pending.put(trial_name)
    statuses = {}

    def worker(gpu_id):
        while True:
            try:
                trial_name = pending.get_nowait()
            except queue.Empty:
                return
            try:
                statuses[trial_name] = run(trial_name, gpu_id)
            except Exception as e:  # keep the other trials going
                statuses[trial_name] = f"error: {e}"

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return statuses


def eval_rows(sweep_dir: Path, trial_name: str) -> List[Dict]:
    """The eval losses of a trial at every eval step."""
    state_file = trial_dir(sweep_dir, trial_name) / "model" / "trainer_state.json"
    if not state_file.exists():
        return []
    with open(state_file) as f:
        history = json.load(f)["log_history"]
    rows = []
    for entry in history:
        if "eval_loss" not in entry:
            continue
        row = {"trial": trial_name, "step": entry["step"]}
        for loss in EVAL_LOSSES:
            # the auxiliary losses may be logged with or without the eval_ prefix
            value = entry.get(loss, entry.get(f"eval_{loss}"))
            if value is not None:
                row[loss] = value
        rows.append(row)
    return rows


def summarize(sweep_dir: Path, trial_names: List[str]) -> pd.DataFrame:
    """One row per trial and eval step, the latest step first and then the lowest eval loss."""
    summary = pd.DataFrame([r for t in trial_names for r in eval_rows(sweep_dir, t)])
    if summary.empty:
        return summary
    return summary.sort_values(
        ["step", "eval_loss"], ascending=[False, True]
    ).reset_index(drop=True)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base_config", required=True, type=Path)
    parser.add_argument("--sweep_dir", required=True, type=Path)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="A field of the config and its values to sweep, name=v1,v2,... (repeatable). "
        "The trials are all combinations of the values",
    )
    parser.add_argument(
        "--trials",
        default=None,
        help="Comma separated trial names to run, out of the grid (default: all of them)",
    )
    parser.add_argument("--stop_at_step", required=True, type=int)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluate and save every this many steps (default: --stop_at_step)",
    )
    parser.add_argument(
        "--gpu_ids", default="0", help="Comma separated GPUs to run the trials on"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the trials of this sweep (the ones in --trials, or the whole grid) and "
        "train them from scratch, instead of picking up where they left off",
    )
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Do not train, only write the summary of what is there",
    )
    return parser


def main(argv: Optional[List[str]] = None):
    args = create_arg_parser().parse_args(argv)
    sweep_dir = args.sweep_dir.resolve()
    eval_steps = args.eval_steps or args.stop_at_step
    trials = build_trials(args.param)
    if args.trials:
        selected = args.trials.split(",")
        unknown = set(selected) - set(trials)
        if unknown:
            raise ValueError(
                f"The trials {sorted(unknown)} are not in the grid {sorted(trials)}"
            )
        trials = {name: trials[name] for name in selected}

    if args.overwrite and args.summarize_only:
        raise ValueError("--overwrite would delete the trials that --summarize_only reads")

    if not args.summarize_only:
        with open(args.base_config) as f:
            base_config = yaml.safe_load(f)
        progress = Progress(sweep_dir)
        if args.overwrite:
            for name in trials:
                reset_trial(sweep_dir, name, progress)
        config_paths = {
            name: write_trial_config(
                base_config, overrides, sweep_dir, name, args.stop_at_step, eval_steps
            )
            for name, overrides in trials.items()
        }
        statuses = run_trials(
            list(trials),
            args.gpu_ids.split(","),
            lambda name, gpu_id: run_trial(
                sweep_dir,
                name,
                trials[name],
                config_paths[name],
                args.stop_at_step,
                gpu_id,
                progress,
            ),
        )
        for name, status in statuses.items():
            print(f"{name}: {status}")

    summary = summarize(sweep_dir, list(trials))
    if summary.empty:
        print("No eval results yet")
        return
    summary.to_csv(sweep_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Written to {sweep_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
