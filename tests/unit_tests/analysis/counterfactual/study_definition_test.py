import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cehrgpt.analysis.counterfactual import study_definition as study

ACE_INHIBITORS = {1335471, 1340128, 1341927, 1363749, 1308216, 1310756, 1373225, 1331235, 1334456, 1342439}
THIAZIDES = {1395058, 974166, 978555, 907013}


def write_study(folder: Path, default_pair=True, indication=True):
    """A small study of its own: drugs 1-6, classes A = {1, 2}, B = {3}, C = {4, 5}."""
    (folder / "exposures.csv").write_text(
        "type,name,cohort_id,concept_id,included_concept_ids\n"
        + "".join(f"Drug,Drug {i},{i},{i},\n" for i in range(1, 7))
        + "Group,Class A,101,,1;2\nGroup,Class B,102,,3\nGroup,Class C,103,,4;5\n"
    )
    definition = {
        "name": "Mini Study",
        "exposures": {"levels": ["Group", "Drug"], "default_level": "Group", "drug_level": "Drug"},
        "washout_days": 180,
        "era_gap_days": 45,
        "monotherapy_window_days": None,
        "time_at_risk": {"min_days_at_risk": 2},
        "reference_results": [
            {"target": "Class A", "comparator": "Class B", "label": "Event", "hr": 1.5, "ci_lower": 1.1, "ci_upper": 2.0}
        ],
    }
    if default_pair:
        definition["default_pair"] = {"target": "Class A", "comparator": "Class B"}
    if indication:
        definition["indication"] = {"name": "Condition", "concept_ids": [777], "window_days": 90}
    (folder / "study.json").write_text(json.dumps(definition))
    (folder / "outcomes.json").write_text("[]")


class BundledStudyTest(unittest.TestCase):
    def test_the_rules_of_the_cohorts_are_those_of_legend(self):
        definition = study.load_study()
        self.assertEqual(definition["indication"], {"name": "Hypertension", "concept_ids": [316866], "window_days": 365})
        self.assertEqual(
            (definition["washout_days"], definition["era_gap_days"], definition["monotherapy_window_days"]),
            (365, 30, 7),
        )
        self.assertEqual(definition["time_at_risk"]["min_days_at_risk"], 1)
        self.assertEqual(definition["time_at_risk"]["risk_window_start_day"], 1)

    def test_the_exposures_are_the_ingredients_and_classes_of_the_study(self):
        by_type = {}
        for exposure in study.load_exposures():
            by_type.setdefault(exposure.type, []).append(exposure)
        self.assertEqual({k: len(v) for k, v in by_type.items()}, {"Drug": 58, "Drug class": 15, "Drug major class": 7})
        self.assertEqual(len(study.drug_universe()), 58)
        ingredients = set(study.drug_universe())
        for exposure in by_type["Drug class"] + by_type["Drug major class"]:
            self.assertTrue(set(exposure.ingredients) <= ingredients, exposure.name)

    def test_the_study_is_found_by_name_and_listed(self):
        self.assertIn("legend_hypertension", study.bundled_studies())
        self.assertEqual(study.study_dir("legend_hypertension"), study.STUDIES_DIR / "legend_hypertension")

    def test_ace_inhibitors_against_thiazides(self):
        pair = study.resolve_pair("ACE inhibitors", "Thiazide or thiazide-like diuretics")
        self.assertEqual(set(pair["target"].ingredients), ACE_INHIBITORS)
        self.assertEqual(set(pair["comparator"].ingredients), THIAZIDES)
        # the 44 other ingredients: ARBs, dCCBs, ndCCBs and the second-line agents of the paper
        self.assertEqual(len(pair["exclusion"]), 44)
        self.assertFalse(set(pair["exclusion"]) & (ACE_INHIBITORS | THIAZIDES))
        self.assertEqual(len(set(pair["exclusion"]) | ACE_INHIBITORS | THIAZIDES), 58)

    def test_the_default_pair_is_the_one_of_the_study(self):
        pair = study.resolve_pair()
        self.assertEqual((pair["target"].name, pair["comparator"].name), ("ACE inhibitors", "Thiazide or thiazide-like diuretics"))

    def test_a_name_may_be_a_part_of_one_name_and_is_not_case_sensitive(self):
        self.assertEqual(study.find_exposure("ace inhibitors").name, "ACE inhibitors")
        self.assertEqual(study.find_exposure("thiazide").name, "Thiazide or thiazide-like diuretics")

    def test_an_unknown_or_ambiguous_name_lists_the_choices(self):
        with self.assertRaisesRegex(ValueError, "no 'Drug class' matches.*ACE inhibitors"):
            study.find_exposure("statins")
        with self.assertRaisesRegex(ValueError, "more than one"):
            study.find_exposure("blockers")

    def test_the_levels_are_separate(self):
        self.assertEqual(study.find_exposure("Diuretics", "Drug major class").name, "Diuretics")
        self.assertEqual(study.find_exposure("Lisinopril", "Drug").ingredients, [1308216])

    def test_overlapping_exposures_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "share ingredients"):
            study.resolve_pair("ACE inhibitors", "ACE inhibitors")

    def test_the_reference_results_belong_to_their_pair(self):
        pair = study.resolve_pair()
        lines = study.reference_lines(pair)
        self.assertIn("AMI", lines[1])
        self.assertIn("0.99", lines[1])
        other = study.resolve_pair("Angiotensin receptor blockers (ARBs)", "Loop diuretics")
        self.assertEqual(study.reference_lines(other), [])


class CustomStudyTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        write_study(self.path)

    def test_a_study_folder_drives_the_pair_the_exclusion_and_the_rules(self):
        pair = study.resolve_pair(study=str(self.path))
        self.assertEqual((pair["target"].ingredients, pair["comparator"].ingredients), ([1, 2], [3]))
        # every other drug of the study
        self.assertEqual(pair["exclusion"], [4, 5, 6])
        lines = dict(line.split("=", 1) for line in study.environment(study=str(self.path)))
        self.assertEqual(lines["TARGET_CONCEPT_IDS"], "1,2")
        self.assertEqual(lines["EXCLUSION_CONCEPT_IDS"], '"${EXCLUSION_CONCEPT_IDS-4,5,6}"')
        self.assertEqual(lines["POPULATION_CONCEPT_IDS"], '"${POPULATION_CONCEPT_IDS-777}"')
        self.assertEqual(lines["POPULATION_WINDOW_DAYS"], '"${POPULATION_WINDOW_DAYS:-90}"')
        self.assertEqual(lines["MIN_PRIOR_OBSERVATION_DAYS"], '"${MIN_PRIOR_OBSERVATION_DAYS:-180}"')
        self.assertEqual(lines["ERA_GAP_DAYS"], '"${ERA_GAP_DAYS:-45}"')
        self.assertEqual(lines["MONOTHERAPY_WINDOW_DAYS"], '"${MONOTHERAPY_WINDOW_DAYS-}"')
        self.assertEqual(lines["MIN_DAYS_AT_RISK"], '"${MIN_DAYS_AT_RISK:-2}"')

    def test_another_pair_of_the_same_study(self):
        pair = study.resolve_pair("Class C", "Class B", study=str(self.path))
        self.assertEqual(pair["exclusion"], [1, 2, 6])

    def test_a_study_without_an_indication_has_no_population_filter(self):
        folder = self.path / "no_indication"
        folder.mkdir()
        write_study(folder, indication=False)
        lines = dict(line.split("=", 1) for line in study.environment(study=str(folder)))
        self.assertEqual(lines["POPULATION_CONCEPT_IDS"], '"${POPULATION_CONCEPT_IDS-}"')

    def test_a_study_without_a_default_pair_needs_one(self):
        folder = self.path / "no_pair"
        folder.mkdir()
        write_study(folder, default_pair=False)
        with self.assertRaisesRegex(ValueError, "no default pair"):
            study.resolve_pair(study=str(folder))
        self.assertEqual(study.resolve_pair("Class A", "Class C", study=str(folder))["exclusion"], [3, 6])

    def test_the_reference_results_of_the_study_are_listed_for_its_pair(self):
        lines = study.reference_lines(study.resolve_pair(study=str(self.path)), str(self.path))
        self.assertIn("Mini Study", lines[0])
        self.assertIn("1.50", lines[1])

    def test_a_folder_that_is_not_a_study_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no study"):
            study.study_dir(str(self.path / "missing"))
        (self.path / "outcomes.json").unlink()
        with self.assertRaisesRegex(ValueError, "lacks outcomes.json"):
            study.study_dir(str(self.path))


class CommandLineTest(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "cehrgpt.analysis.counterfactual.study_definition", *args],
            capture_output=True, text=True, check=True,
        ).stdout

    def evaluate(self, assignments, extra=""):
        """Run the assignments in bash, as the analysis script does, and print some variables."""
        out = subprocess.run(
            ["bash", "-c", f'set -euo pipefail\n{extra}\n{assignments}\necho "$TARGET_LABEL|$COMPARATOR_LABEL|$TARGET_CONCEPT_IDS|$POPULATION_CONCEPT_IDS|$POPULATION_WINDOW_DAYS|$MONOTHERAPY_WINDOW_DAYS|$ERA_GAP_DAYS"'],
            capture_output=True, text=True, check=True,
        ).stdout
        return out.strip().split("|")

    def test_the_assignments_define_what_the_scripts_use(self):
        label, other, target_ids, population, window, mono, gap = self.evaluate(self.run_tool())
        self.assertEqual((label, other), ("ace_inhibitors", "thiazide_or_thiazide_like_diuretics"))
        self.assertEqual(set(target_ids.split(",")), {str(i) for i in ACE_INHIBITORS})
        self.assertEqual((population, window, mono, gap), ("316866", "365", "7", "30"))

    def test_a_value_that_is_set_beforehand_wins(self):
        fields = self.evaluate(
            self.run_tool(), extra='export MONOTHERAPY_WINDOW_DAYS="" POPULATION_WINDOW_DAYS=180 TARGET_LABEL=acei'
        )
        self.assertEqual(fields[0], "acei")
        self.assertEqual((fields[4], fields[5]), ("180", ""))

    def test_a_wrong_exposure_name_fails_the_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "cehrgpt.analysis.counterfactual.study_definition", "--target", "x", "--comparator", "y"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_the_lists(self):
        self.assertIn("ACE inhibitors  (10 ingredient(s))", self.run_tool("--list"))
        self.assertIn("legend_hypertension", self.run_tool("--studies"))
        self.assertIn("AMI", self.run_tool("--reference"))


if __name__ == "__main__":
    unittest.main()
