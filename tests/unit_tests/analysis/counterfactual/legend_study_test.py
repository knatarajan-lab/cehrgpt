import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cehrgpt.analysis.counterfactual import legend_outcomes as lo
from cehrgpt.analysis.counterfactual import legend_study as study

ACE_INHIBITORS = {1335471, 1340128, 1341927, 1363749, 1308216, 1310756, 1373225, 1331235, 1334456, 1342439}
THIAZIDES = {1395058, 974166, 978555, 907013}


class StudyDefinitionTest(unittest.TestCase):
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
        self.assertEqual(len(study.antihypertensive_ingredients()), 58)
        # every class holds ingredients of the study
        ingredients = set(study.antihypertensive_ingredients())
        for exposure in by_type["Drug class"] + by_type["Drug major class"]:
            self.assertTrue(set(exposure.ingredients) <= ingredients, exposure.name)


class ResolvePairTest(unittest.TestCase):
    def test_ace_inhibitors_against_thiazides(self):
        pair = study.resolve_pair("ACE inhibitors", "Thiazide or thiazide-like diuretics")
        self.assertEqual(set(pair["target"].ingredients), ACE_INHIBITORS)
        self.assertEqual(set(pair["comparator"].ingredients), THIAZIDES)
        # the 44 other ingredients: ARBs, dCCBs, ndCCBs and the second-line agents of the paper
        self.assertEqual(len(pair["exclusion"]), 44)
        self.assertFalse(set(pair["exclusion"]) & (ACE_INHIBITORS | THIAZIDES))
        self.assertEqual(
            len(set(pair["exclusion"]) | ACE_INHIBITORS | THIAZIDES), 58
        )

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
        lisinopril = study.find_exposure("Lisinopril", "Drug")
        self.assertEqual(lisinopril.ingredients, [1308216])

    def test_overlapping_exposures_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "share ingredients"):
            study.resolve_pair("ACE inhibitors", "ACE inhibitors")


class EnvironmentTest(unittest.TestCase):
    def assignments(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "cehrgpt.analysis.counterfactual.legend_study", *args],
            capture_output=True, text=True, check=True,
        ).stdout

    def evaluate(self, script_text, extra=""):
        """Run the assignments in bash, as the analysis scripts do, and print some variables."""
        out = subprocess.run(
            ["bash", "-c", f'set -euo pipefail\n{extra}\n{script_text}\necho "$TARGET_LABEL|$COMPARATOR_LABEL|$TARGET_CONCEPT_IDS|$POPULATION_CONCEPT_IDS|$POPULATION_WINDOW_DAYS|$MONOTHERAPY_WINDOW_DAYS|$ERA_GAP_DAYS"'],
            capture_output=True, text=True, check=True,
        ).stdout
        return out.strip().split("|")

    def test_the_assignments_define_what_the_scripts_use(self):
        text = self.assignments("--target", "ACE inhibitors", "--comparator", "Thiazide or thiazide-like diuretics")
        label, other, target_ids, population, window, mono, gap = self.evaluate(text)
        self.assertEqual((label, other), ("ace_inhibitors", "thiazide_or_thiazide_like_diuretics"))
        self.assertEqual(set(target_ids.split(",")), {str(i) for i in ACE_INHIBITORS})
        self.assertEqual((population, window, mono, gap), ("316866", "365", "7", "30"))

    def test_a_value_that_is_set_beforehand_wins(self):
        text = self.assignments("--target", "ACE inhibitors", "--comparator", "Thiazide or thiazide-like diuretics")
        fields = self.evaluate(
            text, extra='export MONOTHERAPY_WINDOW_DAYS="" POPULATION_WINDOW_DAYS=180 TARGET_LABEL=acei'
        )
        self.assertEqual(fields[0], "acei")
        self.assertEqual((fields[4], fields[5]), ("180", ""))

    def test_a_wrong_class_name_fails_the_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "cehrgpt.analysis.counterfactual.legend_study", "--target", "x", "--comparator", "y"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_the_list_shows_the_classes(self):
        listing = self.assignments("--list")
        self.assertIn("ACE inhibitors  (10 ingredient(s))", listing)


if __name__ == "__main__":
    unittest.main()
