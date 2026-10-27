import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "run_causal_study.sh"


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
            f'if [[ "$1" == *study_definition.py ]]; then exec "{sys.executable}" "$@"; fi\n'
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
        self.assertEqual(len(self.calls(commands, "outcome_definitions.py")), 1)
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
            TARGET="Angiotensin receptor blockers (ARBs)",
            COMPARATOR="Dihydropyridine calcium channel blockers (dCCB)",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        extraction = self.calls(commands, "extract_drug_initiation_sequences.py")[0]
        self.assertIn("--source_concept_ids 40235485,1351557,1346686,1347384,1367500,40226742,1317640,1308842", extraction)
        self.assertIn("angiotensin_receptor_blockers_arbs", " ".join(commands))

    def test_an_unknown_design_or_class_stops_the_script_before_any_step(self):
        for env in ({"DESIGN": "both"}, {"TARGET": "statins"}, {"STUDY": "no_such_study"}):
            self.log.unlink(missing_ok=True)
            result, commands = self.run_script(**env)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(commands, [])

    def test_the_inputs_are_set_from_the_command_line_and_the_tokenizer_defaults_to_the_model(self):
        _, commands = self.run_script(
            PATIENT_SEQUENCE_PATH="/my/patient_sequence", VOCAB_PATH="/my/vocab", MODEL_PATH="/my/model"
        )
        extraction = self.calls(commands, "extract_drug_initiation_sequences.py")[0]
        self.assertIn("--patient_sequence_path /my/patient_sequence", extraction)
        self.assertIn("--vocab_path /my/vocab", extraction)
        self.assertIn("--tokenizer_path /my/model", extraction)
        generation = self.calls(commands, "generate_counterfactual_sequences.py")[0]
        self.assertIn("--model_name_or_path /my/model", generation)
        self.log.unlink()
        _, commands = self.run_script(MODEL_PATH="/my/model", TOKENIZER_PATH="/my/tokenizer")
        self.assertIn("--tokenizer_path /my/tokenizer", self.calls(commands, "extract_drug_initiation_sequences.py")[0])

    def test_the_outcomes_of_the_study_are_resolved_in_step_zero(self):
        _, commands = self.run_script()
        outcomes = self.calls(commands, "outcome_definitions.py")[0]
        self.assertIn("--study " + str(ROOT / "src/cehrgpt/analysis/counterfactual/studies/legend_hypertension"), outcomes)

    def test_another_study_drives_the_whole_run(self):
        study = self.root / "mini_study"
        study.mkdir()
        (study / "exposures.csv").write_text(
            "type,name,cohort_id,concept_id,included_concept_ids\n"
            + "".join(f"Drug,Drug {i},{i},{i},\n" for i in range(1, 7))
            + "Group,Class A,101,,1;2\nGroup,Class B,102,,3\n"
        )
        (study / "study.json").write_text(
            '{"name": "Mini Study", "exposures": {"levels": ["Group", "Drug"], "default_level": "Group", '
            '"drug_level": "Drug"}, "default_pair": {"target": "Class A", "comparator": "Class B"}, '
            '"indication": {"name": "Condition", "concept_ids": [777], "window_days": 90}, '
            '"washout_days": 180, "era_gap_days": 45, "monotherapy_window_days": null, '
            '"time_at_risk": {"min_days_at_risk": 2}}'
        )
        (study / "outcomes.json").write_text("[]")
        result, commands = self.run_script(STUDY=str(study), OUTPUT_ROOT=str(self.root / "mini_out"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        joined = "\n".join(commands)
        extraction = self.calls(commands, "extract_drug_initiation_sequences.py")[0]
        self.assertIn("--source_concept_ids 1,2", extraction)
        self.assertIn("--comparator_concept_ids 3", extraction)
        self.assertIn("--exclusion_concept_ids 4,5,6", extraction)
        self.assertIn("--population_concept_ids 777", extraction)
        self.assertIn("--population_window_days 90", extraction)
        self.assertIn("--min_prior_observation_days 180", extraction)
        self.assertIn("--era_gap_days 45", extraction)
        self.assertNotIn("--monotherapy_window_days", extraction)  # the study has none
        self.assertIn("--min_days_at_risk 2", self.calls(commands, "hazard_ratio_estimation.py")[0])
        self.assertIn("--study " + str(study), self.calls(commands, "outcome_definitions.py")[0])
        # nothing of the other study is left in the run
        for leftover in ("316866", "1335471", "1395058", "ace_inhibitors"):
            self.assertNotIn(leftover, joined)
        # the arms are named after the exposures of that study
        self.assertIn("class_a:", joined)
        self.assertIn("class_b:", joined)


if __name__ == "__main__":
    unittest.main()
