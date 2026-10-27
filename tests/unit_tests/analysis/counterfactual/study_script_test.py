import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "run_legend_htn_study.sh"


@unittest.skipUnless(shutil.which("bash"), "bash is not available")
class StudyScriptTest(unittest.TestCase):
    """
    Runs the script with a stand-in for ``python`` that records the commands, so the flow of each
    design is checked without data or GPUs.  The study tool runs for real.
    """

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.log = self.root / "commands.log"
        shim = self.root / "bin" / "python"
        shim.parent.mkdir()
        shim.write_text(
            "#!/usr/bin/env bash\n"
            f'if [[ "$1" == *legend_study.py ]]; then exec "{sys.executable}" "$@"; fi\n'
            'echo "$*" >> "$SHIM_LOG"\n'
            'if [[ "$1" == "-" ]]; then cat > /dev/null; fi\n'
            "exit 0\n"
        )
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)

    def run_script(self, **env):
        environment = {
            **os.environ,
            "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(ROOT / "src"),
            "SHIM_LOG": str(self.log),
            "OUTPUT_ROOT": str(self.root / "out"),
            # the stand-in finishes at once, so the arms never run on a GPU
            "SUPPRESS_CONCEPTS": "true",
            **env,
        }
        result = subprocess.run(
            ["bash", str(SCRIPT)], cwd=ROOT, env=environment, capture_output=True, text=True
        )
        commands = self.log.read_text().splitlines() if self.log.exists() else []
        return result, commands

    @staticmethod
    def calls(commands, script):
        return [c for c in commands if script in c.split()[0]]

    def test_the_symmetric_design_extracts_and_swaps_in_both_directions(self):
        result, commands = self.run_script(DESIGN="symmetric")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls(commands, "legend_outcomes.py")), 1)
        extractions = self.calls(commands, "extract_drug_initiation_sequences.py")
        swaps = self.calls(commands, "create_drug_swap_context.py")
        self.assertEqual((len(extractions), len(swaps)), (2, 2))
        # the second extraction has the comparator class as the source
        self.assertIn("--source_concept_ids 1395058,974166,978555,907013", extractions[1])
        # the arms are the combination of the two directions, both overlap scores are used
        self.assertIn("swap_contexts/ace_inhibitors_arm_ctx.parquet", " ".join(commands))
        generated = self.calls(commands, "hazard_ratio_estimation.py")[0]
        self.assertIn("--drug_info_path " + str(self.root / "out" / "swap_contexts" / "drug_info"), generated)
        self.assertIn("target_pts_overlap_scores.parquet", generated)
        self.assertIn("comparator_pts_overlap_scores.parquet", generated)
        self.assertIn("comparator_contexts/observed_outcomes", generated)

    def test_the_target_initiators_design_uses_one_extraction_and_one_swap(self):
        result, commands = self.run_script(DESIGN="target_initiators")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls(commands, "extract_drug_initiation_sequences.py")), 1)
        self.assertEqual(len(self.calls(commands, "create_drug_swap_context.py")), 1)
        arms = " ".join(self.calls(commands, "generate_counterfactual_sequences.py"))
        self.assertIn("ace_inhibitors:" + str(self.root / "out" / "swap_contexts" / "target_pts_target_ctx.parquet"), arms)
        self.assertIn(
            "thiazide_or_thiazide_like_diuretics:" + str(self.root / "out" / "swap_contexts" / "target_pts_comparator_ctx.parquet"),
            arms,
        )
        generated = self.calls(commands, "hazard_ratio_estimation.py")[0]
        self.assertIn("--drug_info_path " + str(self.root / "out" / "target_contexts" / "drug_info"), generated)
        self.assertNotIn("comparator_pts_overlap_scores", generated)

    def test_both_designs_compute_the_generated_and_the_observed_hazard_ratios(self):
        for design in ("symmetric", "target_initiators"):
            self.log.unlink(missing_ok=True)
            result, commands = self.run_script(DESIGN=design, OUTPUT_ROOT=str(self.root / design))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            generated, observed = self.calls(commands, "hazard_ratio_estimation.py")
            self.assertIn("--trajectories_dir", generated)
            self.assertNotIn("--trajectories_dir", observed)
            self.assertIn("--observed_outcomes_path " + str(self.root / design / "target_contexts" / "observed_outcomes"), observed)

    def test_the_legend_rules_reach_the_extraction(self):
        _, commands = self.run_script()
        extraction = self.calls(commands, "extract_drug_initiation_sequences.py")[0]
        for expected in (
            "--monotherapy_window_days 7",
            "--population_concept_ids 316866",
            "--population_window_days 365",
            "--min_prior_observation_days 365",
            "--era_gap_days 30",
            "--outcome_definitions_path",
        ):
            self.assertIn(expected, extraction)
        self.assertNotIn("--censor_at_switch", extraction)

    def test_another_pair_of_classes_changes_the_ingredients(self):
        result, commands = self.run_script(
            TARGET_CLASS="Angiotensin receptor blockers (ARBs)",
            COMPARATOR_CLASS="Dihydropyridine calcium channel blockers (dCCB)",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        extraction = self.calls(commands, "extract_drug_initiation_sequences.py")[0]
        self.assertIn("--source_concept_ids 40235485,1351557,1346686,1347384,1367500,40226742,1317640,1308842", extraction)
        self.assertIn("angiotensin_receptor_blockers_arbs", " ".join(commands))

    def test_an_unknown_design_or_class_stops_the_script_before_any_step(self):
        for env in ({"DESIGN": "both"}, {"TARGET_CLASS": "statins"}):
            self.log.unlink(missing_ok=True)
            result, commands = self.run_script(**env)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
