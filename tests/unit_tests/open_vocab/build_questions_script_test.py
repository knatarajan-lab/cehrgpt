import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_open_vocab_questions.sh"

# Stands for python: records the call, and the universe step writes the file of each domain it is given
STUB = """#!/bin/sh
echo "$@" >> "$STUB_LOG"
case "$2" in
  cehrgpt.open_vocab.question_universe)
    shift 2
    while [ $# -gt 0 ]; do
      case "$1" in
        --output_dir) out="$2"; shift ;;
        --domains) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do doms="$doms $1"; shift; done; continue ;;
      esac
      shift
    done
    mkdir -p "$out"
    for d in $doms; do touch "$out/$(echo "$d" | tr 'A-Z' 'a-z')_universe.parquet"; done
    ;;
esac
exit ${STUB_EXIT:-0}
"""


class BuildQuestionsScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stub = self.root / "fakepython"
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.log = self.root / "calls.log"
        self.out = self.root / "open_vocab"

    def run_script(self, *args, exit_code=0):
        env = {**os.environ, "STUB_LOG": str(self.log), "STUB_EXIT": str(exit_code)}
        return subprocess.run(
            ["sh", str(SCRIPT), f"--python={self.root / 'fakepython'}", *args], capture_output=True, text=True, env=env
        )

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_both_steps_with_the_defaults(self):
        result = self.run_script(f"--vocab_path={self.root}/ohdsi", f"--output_dir={self.out}/")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        universe, embeddings = self.calls()
        universe_dir, embeddings_dir = f"{self.out}/question_universe", f"{self.out}/question_embeddings"
        self.assertEqual(
            universe,
            f"-m cehrgpt.open_vocab.question_universe --vocab_path {self.root}/ohdsi --output_dir {universe_dir} "
            "--domains Condition Procedure Drug Visit",
        )
        files = " ".join(f"{universe_dir}/{d}_universe.parquet" for d in ("condition", "procedure", "drug", "visit"))
        self.assertEqual(
            embeddings,
            f"-m cehrgpt.open_vocab.question_embeddings --universe_files {files} --output_dir {embeddings_dir}",
        )
        # the lines for the config of the training
        self.assertIn(f'open_vocab_universe_dir: "{universe_dir}"', result.stdout)
        self.assertIn(f'open_vocab_embeddings_dir: "{embeddings_dir}"', result.stdout)

    def test_the_options_are_passed_to_the_steps(self):
        result = self.run_script(
            f"--vocab_path={self.root}/ohdsi", f"--output_dir={self.out}", "--domains=Condition,Drug", "--max_depth=4",
            "--model=my/encoder", "--revision=abc123", "--template=Medical: {text}", "--batch_size=64", "--device=cuda",
            "--fp16", "--overwrite",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        universe, embeddings = self.calls()
        self.assertTrue(universe.endswith("--domains Condition Drug --max_depth 4"), universe)
        self.assertIn("condition_universe.parquet", embeddings)
        self.assertIn("drug_universe.parquet", embeddings)
        self.assertNotIn("visit_universe.parquet", embeddings)
        self.assertTrue(
            embeddings.endswith(
                "--model my/encoder --revision abc123 --template Medical: {text} --batch_size 64 --device cuda "
                "--fp16 --overwrite"
            ),
            embeddings,
        )

    def test_an_existing_universe_is_kept_unless_overwrite(self):
        args = (f"--vocab_path={self.root}/ohdsi", f"--output_dir={self.out}")
        self.run_script(*args)
        self.log.unlink()
        again = self.run_script(*args)
        self.assertEqual(again.returncode, 0)
        self.assertIn("keeping it", again.stdout)
        calls = self.calls()
        self.assertEqual(len(calls), 1)
        self.assertIn("question_embeddings", calls[0])  # the embeddings are always asked for; they have a cache
        self.log.unlink()
        self.run_script(*args, "--overwrite")
        self.assertEqual(len(self.calls()), 2)

    def test_one_step_alone(self):
        only_universe = self.run_script(f"--vocab_path={self.root}/ohdsi", f"--output_dir={self.out}", "--skip_embeddings")
        self.assertEqual(only_universe.returncode, 0)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn("question_universe", self.calls()[0])
        self.log.unlink()
        # embedding needs no vocabulary, only the universe that is there
        only_embeddings = self.run_script(f"--output_dir={self.out}", "--skip_universe")
        self.assertEqual(only_embeddings.returncode, 0, only_embeddings.stdout)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn("question_embeddings", self.calls()[0])

    def test_embedding_without_a_universe_is_an_error(self):
        result = self.run_script(f"--output_dir={self.out}", "--skip_universe")
        self.assertEqual(result.returncode, 1)
        self.assertIn("build the universe first", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_wrong_arguments(self):
        self.assertNotEqual(self.run_script().returncode, 0)  # no output_dir: usage
        missing_vocab = self.run_script(f"--output_dir={self.out}")
        self.assertEqual(missing_vocab.returncode, 1)
        self.assertIn("--vocab_path is required", missing_vocab.stdout)
        unknown = self.run_script(f"--vocab_path=x", f"--output_dir={self.out}", "--domains=Measurement")
        self.assertEqual(unknown.returncode, 1)
        self.assertIn("unknown domain 'Measurement'", unknown.stdout)
        both = self.run_script(f"--output_dir={self.out}", "--skip_universe", "--skip_embeddings")
        self.assertEqual(both.returncode, 1)
        self.assertEqual(self.calls(), [])

    def test_a_failing_step_stops_the_script(self):
        result = self.run_script(f"--vocab_path={self.root}/ohdsi", f"--output_dir={self.out}", exit_code=3)
        self.assertEqual(result.returncode, 1)
        self.assertIn("building the question universe failed", result.stdout)
        self.assertEqual(len(self.calls()), 1)  # the embeddings were not started


if __name__ == "__main__":
    unittest.main()
