import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import polars as pl
import torch

from cehrgpt.analysis.counterfactual import generate_counterfactual_sequences as gen

NUM_TRAJECTORIES = 2
BATCH_SIZE = 3


def write_context(path, person_ids):
    pl.DataFrame(
        {
            "person_id": person_ids,
            "drug_epoch_time": [0.0] * len(person_ids),
            "drug_concept_id": ["100"] * len(person_ids),
            "concept_ids": [["9202", "100"]] * len(person_ids),
        }
    ).write_parquet(path)


class FakeCollator:
    def __init__(self, **kwargs):
        pass

    def __call__(self, examples):
        return {}


class GenerationTestCase(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.output = self.root / "trajectories"
        write_context(self.root / "a.parquet", list(range(1, 8)))  # 3 batches of 3, 3, 1
        write_context(self.root / "b.parquet", list(range(11, 15)))  # 2 batches of 3, 1

    def person_ids(self, arm, traj_id, batch_i):
        path = gen.batch_output_path(self.output, arm, traj_id, batch_i)
        return pl.read_parquet(path)["subject_id"].to_list()


class BatchTasksTest(GenerationTestCase):
    def test_every_batch_of_every_trajectory_is_pending_at_first(self):
        tasks = gen.pending_tasks("a", 7, BATCH_SIZE, NUM_TRAJECTORIES, self.output)
        self.assertEqual(tasks, [(t, b) for t in range(2) for b in range(3)])

    def test_a_batch_with_a_parquet_is_not_pending(self):
        path = gen.batch_output_path(self.output, "a", 1, 2)
        path.parent.mkdir(parents=True)
        pl.DataFrame({"subject_id": [7]}).write_parquet(path)
        tasks = gen.pending_tasks("a", 7, BATCH_SIZE, NUM_TRAJECTORIES, self.output)
        self.assertEqual(len(tasks), 5)
        self.assertNotIn((1, 2), tasks)

    def test_a_partial_file_does_not_count_as_a_finished_batch(self):
        path = gen.batch_output_path(self.output, "a", 0, 0)
        path.parent.mkdir(parents=True)
        path.with_name(path.name + ".tmp").write_bytes(b"partial")
        self.assertIn((0, 0), gen.pending_tasks("a", 7, BATCH_SIZE, NUM_TRAJECTORIES, self.output))

    def test_the_rows_of_a_context_directory_are_counted(self):
        shards = self.root / "shards"
        shards.mkdir()
        write_context(shards / "1.parquet", [1, 2])
        write_context(shards / "2.parquet", [3])
        self.assertEqual(gen.context_num_rows(str(shards)), 3)
        self.assertEqual(gen.context_num_rows(str(self.root / "a.parquet")), 7)


class RunGenerationTest(GenerationTestCase):
    def run_generation(self, **kwargs):
        calls = []

        def fake_generate(batch, trajectory_id, arm, **_):
            calls.append((arm, trajectory_id, batch["person_id"].tolist()))
            return pl.DataFrame(
                {
                    "subject_id": batch["person_id"].tolist(),
                    "trajectory_id": [trajectory_id] * len(batch["person_id"]),
                }
            )

        with mock.patch.object(gen, "CehrGptDataCollator", FakeCollator), mock.patch.object(
            gen, "generate_trajectories_for_batch", side_effect=fake_generate
        ):
            gen.run_generation(
                context_parquet=str(self.root / "a.parquet"),
                arm="a",
                cehrgpt_tokenizer=None,
                cehrgpt_model=None,
                device=torch.device("cpu"),
                output_dir=self.output,
                num_trajectories=NUM_TRAJECTORIES,
                batch_size=BATCH_SIZE,
                context_length=100,
                max_length=200,
                include_values=False,
                num_workers=0,
                **kwargs,
            )
        return calls

    def test_the_contexts_are_cut_into_consecutive_batches_for_every_trajectory(self):
        calls = self.run_generation()
        self.assertEqual(len(calls), 6)
        for traj_id in range(2):
            self.assertEqual(self.person_ids("a", traj_id, 0), [1, 2, 3])
            self.assertEqual(self.person_ids("a", traj_id, 1), [4, 5, 6])
            self.assertEqual(self.person_ids("a", traj_id, 2), [7])

    def test_only_the_given_tasks_are_generated(self):
        calls = self.run_generation(tasks=[(1, 2), (0, 1)])
        self.assertEqual(sorted(c[1:] for c in calls), [(0, [4, 5, 6]), (1, [7])])
        written = sorted(p.name for p in self.output.rglob("*.parquet"))
        self.assertEqual(written, ["batch_00001.parquet", "batch_00002.parquet"])

    def test_finished_batches_are_not_generated_again(self):
        self.run_generation()
        self.assertEqual(self.run_generation(), [])
        self.assertEqual(len(self.run_generation(tasks=[(0, 0), (1, 1)])), 0)

    def test_no_partial_files_are_left_behind(self):
        self.run_generation()
        self.assertEqual(list(self.output.rglob("*.tmp")), [])


class MultiGpuTest(GenerationTestCase):
    def parse(self, *extra):
        argv = [
            "prog",
            "--arm_context", f"a:{self.root / 'a.parquet'}",
            "--arm_context", f"b:{self.root / 'b.parquet'}",
            "--model_name_or_path", "model",
            "--tokenizer_path", "tokenizer",
            "--output_dir", str(self.output),
            "--num_trajectories", str(NUM_TRAJECTORIES),
            "--batch_size", str(BATCH_SIZE),
            "--gpu_ids", "0,1",
            *extra,
        ]  # fmt: skip
        with mock.patch.object(gen.sys, "argv", argv):
            return gen._parse_args(), argv

    def launch(self, gpu_ids, exit_code=0):
        args, argv = self.parse()
        calls = []
        worker_tasks = []

        def fake_popen(command, env, stdout, stderr):
            task_file = command[command.index("--task_file") + 1]
            with open(task_file) as f:
                worker_tasks.append(
                    sorted((arm, t, b) for arm, tasks in json.load(f).items() for t, b in tasks)
                )
            calls.append((command, env))
            process = mock.Mock()
            process.wait.return_value = exit_code
            process.poll.return_value = exit_code
            return process

        with mock.patch.object(gen.subprocess, "Popen", side_effect=fake_popen), mock.patch.object(
            gen.sys, "argv", argv
        ), mock.patch.object(gen.signal, "signal"):
            code = gen.run_on_multiple_gpus(args, gpu_ids)
        return code, calls, worker_tasks

    def all_tasks(self):
        return sorted(
            [("a", t, b) for t in range(2) for b in range(3)]
            + [("b", t, b) for t in range(2) for b in range(2)]
        )

    def test_the_batches_are_dealt_out_without_overlap_one_worker_per_gpu_and_round(self):
        code, calls, worker_tasks = self.launch(["2", "3"])
        self.assertEqual(code, 0)
        # two trajectories, so two rounds of one worker per GPU
        self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for _, env in calls], ["2", "3", "2", "3"])
        self.assertEqual(sorted(sum(worker_tasks, [])), self.all_tasks())
        for first, second in ((0, 1), (2, 3)):
            self.assertFalse(set(worker_tasks[first]) & set(worker_tasks[second]))
            self.assertLessEqual(abs(len(worker_tasks[first]) - len(worker_tasks[second])), 1)
        for command, _ in calls:
            self.assertNotIn("--gpu_ids", command)
            self.assertIn("--arm_context", command)
        self.assertFalse((self.output / "temp").exists())

    def test_a_round_is_one_trajectory_of_all_the_arms(self):
        _, _, worker_tasks = self.launch(["0", "1"])
        for round_index in (0, 1):
            tasks = worker_tasks[2 * round_index] + worker_tasks[2 * round_index + 1]
            self.assertEqual({t for _, t, _ in tasks}, {round_index})
            self.assertEqual({arm for arm, _, _ in tasks}, {"a", "b"})

    def test_the_arms_are_generated_at_the_same_time_on_different_gpus(self):
        _, _, worker_tasks = self.launch(["0", "1"])
        # the batches go out alternately (a0, b0, a1, b1, a2), so each GPU gets one arm
        self.assertEqual({arm for arm, _, _ in worker_tasks[0]}, {"a"})
        self.assertEqual({arm for arm, _, _ in worker_tasks[1]}, {"b"})

    def test_finished_batches_are_left_out_whatever_the_number_of_gpus(self):
        write_batch = gen.batch_output_path(self.output, "a", 0, 0)
        write_batch.parent.mkdir(parents=True)
        pl.DataFrame({"subject_id": [1]}).write_parquet(write_batch)
        code, _, worker_tasks = self.launch(["0", "1", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(
            sorted(sum(worker_tasks, [])), [t for t in self.all_tasks() if t != ("a", 0, 0)]
        )

    def test_there_are_no_more_workers_than_batches(self):
        for traj_id in range(2):
            for arm, count in (("a", 3), ("b", 2)):
                for batch_i in range(count):
                    path = gen.batch_output_path(self.output, arm, traj_id, batch_i)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if (arm, traj_id, batch_i) not in (("a", 1, 2), ("b", 0, 0)):
                        pl.DataFrame({"subject_id": [1]}).write_parquet(path)
        code, calls, worker_tasks = self.launch(["0", "1", "2", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(sum(worker_tasks, [])), [("a", 1, 2), ("b", 0, 0)])

    def test_nothing_is_launched_when_everything_is_done(self):
        for arm, tasks in (("a", 3), ("b", 2)):
            for traj_id in range(2):
                for batch_i in range(tasks):
                    path = gen.batch_output_path(self.output, arm, traj_id, batch_i)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pl.DataFrame({"subject_id": [1]}).write_parquet(path)
        code, calls, _ = self.launch(["0", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])

    def test_a_failed_worker_returns_non_zero_stops_the_next_round_and_keeps_the_task_files(self):
        code, calls, _ = self.launch(["0", "1"], exit_code=1)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)  # the workers of the first round only
        self.assertTrue((self.output / "temp" / "tasks" / "worker_0.json").exists())
        self.assertTrue((self.output / "worker_logs").is_dir())

    def test_a_worker_generates_only_its_own_tasks(self):
        task_file = self.root / "tasks.json"
        task_file.write_text(json.dumps({"b": [[1, 1]]}))
        argv = [
            "prog",
            "--arm_context", f"a:{self.root / 'a.parquet'}",
            "--arm_context", f"b:{self.root / 'b.parquet'}",
            "--model_name_or_path", "model",
            "--tokenizer_path", "tokenizer",
            "--output_dir", str(self.output),
            "--task_file", str(task_file),
            "--gpu_ids", "0,1",
        ]  # fmt: skip
        generated = []
        model = mock.Mock()
        model.config.include_values = False
        with mock.patch.object(gen.sys, "argv", argv), mock.patch.object(
            gen.CehrGptTokenizer, "from_pretrained", return_value=mock.Mock()
        ), mock.patch.object(
            gen.CEHRGPT2LMHeadModel, "from_pretrained"
        ) as from_pretrained, mock.patch.object(
            gen, "run_generation", side_effect=lambda **kw: generated.append((kw["arm"], kw["tasks"]))
        ), mock.patch.object(
            gen, "run_on_multiple_gpus"
        ) as launcher:
            from_pretrained.return_value.eval.return_value.to.return_value = model
            gen.main()
        launcher.assert_not_called()
        self.assertEqual(generated, [("b", [(1, 1)])])


if __name__ == "__main__":
    unittest.main()
