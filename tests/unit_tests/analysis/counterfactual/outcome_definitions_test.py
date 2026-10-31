import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import polars as pl

from cehrgpt.analysis.counterfactual import hazard_ratio_estimation as hr
from cehrgpt.analysis.counterfactual import outcome_definitions as lo
from cehrgpt.analysis.counterfactual.study_definition import study_dir
from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import (
    find_outcomes_after_drug,
    process,
)

COLUMNS = [
    "name", "kind", "concept_ids", "exact_concept_ids", "exclude_concept_ids",
    "exclude_exact_concept_ids", "inpatient_or_er", "note",
]  # fmt: skip


def rows_of(*rows):
    """Rows of outcomes.csv from dicts that leave out the empty columns."""
    return [{c: row.get(c, "") for c in COLUMNS} for row in rows]


def write_vocab(root: Path, pairs):
    (root / "vocab" / "concept_ancestor").mkdir(parents=True)
    pl.DataFrame(
        {"ancestor_concept_id": [a for a, _ in pairs], "descendant_concept_id": [d for _, d in pairs]}
    ).write_parquet(root / "vocab" / "concept_ancestor" / "part.parquet")
    return str(root / "vocab")


class ReadTest(unittest.TestCase):
    def parse(self, *rows):
        return lo.parse_outcomes(rows_of(*rows))

    def test_a_row_is_an_outcome_with_a_concept_set(self):
        (outcome,) = self.parse(
            {"name": "Heart failure", "concept_ids": "316139;4", "exact_concept_ids": "9",
             "exclude_concept_ids": "42", "exclude_exact_concept_ids": "43", "inpatient_or_er": "True", "note": "n"}
        )  # fmt: skip
        self.assertEqual((outcome.name, outcome.kind, outcome.inpatient_or_er, outcome.note), ("Heart failure", "primary", True, "n"))
        (concept_set,) = outcome.concept_sets
        self.assertEqual(concept_set.include_descendants, [316139, 4])
        self.assertEqual((concept_set.include, concept_set.exclude_descendants, concept_set.exclude), ([9], [42], [43]))

    def test_the_kind_and_the_visit_constraint_default_to_primary_and_false(self):
        (outcome,) = lo.parse_outcomes([{"name": "A", "concept_ids": "1"}])
        self.assertEqual((outcome.kind, outcome.inpatient_or_er), ("primary", False))

    def test_rows_of_one_name_are_the_concept_sets_of_one_outcome(self):
        outcomes = self.parse(
            {"name": "A", "concept_ids": "4", "exclude_concept_ids": "41"},
            {"name": "B", "concept_ids": "5"},
            {"name": "A", "concept_ids": "6"},
        )
        self.assertEqual([o.name for o in outcomes], ["A", "B"])
        self.assertEqual(len(outcomes[0].concept_sets), 2)

    def test_death_is_a_concept_id_of_its_own(self):
        (outcome,) = self.parse({"name": "All-cause mortality", "concept_ids": "[DEATH]"})
        self.assertTrue(outcome.death)
        self.assertEqual(outcome.concept_sets, [])

    def test_mistakes_are_reported_with_the_row(self):
        cases = {
            "name is empty": [{"name": "", "concept_ids": "1"}],
            "kind has to be": [{"name": "A", "concept_ids": "1", "kind": "secondary"}],
            "has no concept ids": [{"name": "A", "concept_ids": ""}],
            "concept ids, got 'x'": [{"name": "A", "concept_ids": "x"}],
            "true or false": [{"name": "A", "concept_ids": "1", "inpatient_or_er": "maybe"}],
            "differ in kind": [{"name": "A", "concept_ids": "1"}, {"name": "A", "concept_ids": "2", "inpatient_or_er": "true"}],
            "takes no other column": [{"name": "A", "concept_ids": "[DEATH]", "inpatient_or_er": "true"}],
            "same label": [{"name": "A b", "concept_ids": "1"}, {"name": "A-b", "concept_ids": "2"}],
        }  # fmt: skip
        for message, rows in cases.items():
            with self.assertRaisesRegex(ValueError, message):
                self.parse(*rows)

    def test_a_file_without_the_name_column_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("study.json", "exposures.csv"):
                (Path(folder) / name).write_text((study_dir() / name).read_text())
            (Path(folder) / "outcomes.csv").write_text("label,concept_ids\nA,1\n")
            with self.assertRaisesRegex(ValueError, "lacks the column"):
                lo.read_outcomes(folder)


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        # 4 has the descendants 41 and 42, 5 has the descendant 51, 7 has 71
        self.vocab = write_vocab(
            self.root, [(4, 4), (4, 41), (4, 42), (5, 5), (5, 51), (7, 7), (7, 71)]
        )

    def resolve(self, *rows):
        (resolved,) = lo.resolve_outcomes(lo.parse_outcomes(rows_of(*rows)), self.vocab)
        return resolved

    def test_descendants_are_expanded_and_exclusions_removed(self):
        resolved = self.resolve({"name": "Outcome", "concept_ids": "4", "exact_concept_ids": "9", "exclude_concept_ids": "42"})
        self.assertEqual(resolved.concepts, ["4", "41", "9"])
        self.assertFalse(resolved.inpatient_only)
        self.assertEqual((resolved.label, resolved.role), ("outcome", "primary"))

    def test_an_exact_concept_does_not_bring_its_descendants(self):
        resolved = self.resolve({"name": "Outcome", "exact_concept_ids": "4"})
        self.assertEqual(resolved.concepts, ["4"])

    def test_an_excluded_ancestor_removes_its_descendants_too(self):
        resolved = self.resolve({"name": "Outcome", "concept_ids": "4;5", "exclude_concept_ids": "5"})
        self.assertEqual(resolved.concepts, ["4", "41", "42"])

    def test_an_exact_exclusion_removes_only_that_concept(self):
        resolved = self.resolve({"name": "Outcome", "concept_ids": "4", "exclude_exact_concept_ids": "4"})
        self.assertEqual(resolved.concepts, ["41", "42"])

    def test_every_concept_set_has_its_own_exclusions(self):
        resolved = self.resolve(
            {"name": "Outcome", "concept_ids": "4", "exclude_concept_ids": "41"},
            {"name": "Outcome", "exact_concept_ids": "41"},
        )
        self.assertEqual(resolved.concepts, ["4", "41", "42"])

    def test_the_visit_constraint_is_kept(self):
        resolved = self.resolve({"name": "Outcome", "concept_ids": "5", "inpatient_or_er": "true"})
        self.assertTrue(resolved.inpatient_only)

    def test_death_is_the_death_token(self):
        resolved = self.resolve({"name": "All-cause mortality", "concept_ids": "[DEATH]"})
        self.assertEqual(resolved.concepts, [lo.DEATH_TOKEN])

    def test_a_negative_control_is_the_concept_and_its_descendants(self):
        resolved = self.resolve({"name": "Rash", "kind": "negative_control", "concept_ids": "7"})
        self.assertEqual((resolved.label, resolved.role), ("rash", lo.NEGATIVE_CONTROL))
        self.assertEqual(resolved.concepts, ["7", "71"])

    def test_exclusions_that_leave_nothing_are_refused(self):
        with self.assertRaisesRegex(ValueError, "leave no concept"):
            self.resolve({"name": "Outcome", "concept_ids": "5", "exclude_concept_ids": "5"})

    def test_the_definitions_round_trip(self):
        resolved = self.resolve({"name": "A b", "concept_ids": "4"})
        path = self.root / "definitions.json"
        lo.write_resolved([resolved], path)
        self.assertEqual([e["label"] for e in lo.load_outcome_definitions(str(path))], ["a_b"])
        self.assertEqual(lo.definitions_fingerprint(str(path)), lo.definitions_fingerprint(str(path)))

    def test_duplicate_labels_in_the_definitions_are_refused(self):
        resolved = self.resolve({"name": "A", "concept_ids": "4"})
        path = self.root / "definitions.json"
        lo.write_resolved([resolved, resolved], path)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            lo.load_outcome_definitions(str(path))


class BundledDefinitionsTest(unittest.TestCase):
    """The pipeline reads the bundled definitions, so it does not need a checkout of LEGEND."""

    @classmethod
    def setUpClass(cls):
        cls.outcomes = {o.name: o for o in lo.read_outcomes()}

    def of_kind(self, kind):
        return [o for o in self.outcomes.values() if o.kind == kind]

    def test_the_hypertension_outcomes_and_controls_are_bundled(self):
        self.assertEqual((len(self.of_kind("primary")), len(self.of_kind("negative_control"))), (45, 76))

    def test_the_bundled_files_say_which_version_of_legend_they_come_from(self):
        source = json.loads((study_dir() / "source.json").read_text())
        self.assertEqual(source["repository"], "https://github.com/OHDSI/Legend")
        self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual((source["n_outcomes"], source["n_negative_controls"]), (58, 76))
        self.assertEqual(source["n_outcomes"], len(self.of_kind("primary")) + source["n_outcomes_left_out"])

    def test_acute_myocardial_infarction_keeps_its_exclusion_and_visit_restriction(self):
        mi = self.outcomes["Acute myocardial infarction"]
        (concept_set,) = mi.concept_sets
        self.assertEqual((concept_set.include_descendants, concept_set.exclude_descendants), ([4329847], [314666]))
        self.assertTrue(mi.inpatient_or_er)

    def test_hemorrhagic_stroke_members_do_not_include_descendants(self):
        stroke = self.outcomes["Hemorrhagic stroke"]
        self.assertEqual([c.include for c in stroke.concept_sets], [[376713, 432923, 439847]])
        self.assertEqual([c.include_descendants for c in stroke.concept_sets], [[]])

    def test_cardiovascular_disease_has_a_concept_set_per_branch(self):
        self.assertGreater(len(self.outcomes["Cardiovascular disease"].concept_sets), 1)

    def test_the_outcomes_that_need_an_inpatient_or_er_visit_have_the_constraint(self):
        needs_visit = [
            "Acute myocardial infarction", "Acute pancreatitis", "Acute renal failure",
            "Anaphylactoid reaction", "Angioedema", "Cardiovascular disease",
            "Gastrointestinal bleeding", "Hemorrhagic stroke", "Hospitalization with heart failure",
            "Ischemic stroke", "Rhabdomyolysis", "Stroke", "Sudden cardiac death",
            "Transient ischemic attack", "Hospitalization with preinfarction syndrome",
        ]  # fmt: skip
        self.assertEqual(len(needs_visit), 15)  # the 15 of the paper
        # the SQL of "Coronary heart disease" restricts it to inpatient and ER visits as well
        needs_visit.append("Coronary heart disease")
        self.assertEqual(sorted(o.name for o in self.outcomes.values() if o.inpatient_or_er), sorted(needs_visit))
        self.assertIn("creatine kinase", self.outcomes["Rhabdomyolysis"].note)
        self.assertIn("potassium", self.outcomes["Hyperkalemia"].note)

    def test_the_outcomes_that_cannot_be_matched_on_tokens_are_not_in_the_file(self):
        for name in ("Heart failure", "Measured renal dysfunction", "Cardiovascular-related mortality"):
            self.assertNotIn(name, self.outcomes)

    def test_all_cause_mortality_is_the_death_record(self):
        self.assertTrue(self.outcomes["All-cause mortality"].death)

    def test_a_study_is_read_from_a_folder_too(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("study.json", "exposures.csv", "outcomes.csv"):
                (Path(folder) / name).write_text((study_dir() / name).read_text())
            self.assertEqual(len(lo.read_outcomes(folder)), 121)

    def vocabulary(self, root):
        ids = sorted({c for o in self.outcomes.values() for s in o.concept_sets for c in s.concept_ids()})
        return write_vocab(root, [(i, i) for i in ids])

    def test_the_command_line_resolves_the_bundled_definitions_for_a_vocabulary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = ["prog", "--vocab_path", self.vocabulary(root), "--output_dir", str(root / "out")]
            with mock.patch("sys.argv", argv):
                lo.main()
            entries = json.loads((root / "out" / "outcome_definitions.json").read_text())
        self.assertEqual(len(entries), 121)
        self.assertEqual(sum(e["role"] == lo.NEGATIVE_CONTROL for e in entries), 76)
        self.assertEqual(sum(e["inpatient_only"] for e in entries), 16)
        by_name = {e["name"]: e for e in entries}
        self.assertEqual(by_name["All-cause mortality"]["concepts"], [lo.DEATH_TOKEN])


class FindOutcomesTest(unittest.TestCase):
    TOKENS = ["x", "[VS]", "9201", "a", "[VE]", "[VS]", "9202", "a", "b", "[VE]"]
    TIMES = [float(i) for i in range(10)]

    def test_a_token_in_two_outcomes_counts_for_both(self):
        groups = {"one": {"a"}, "two": {"a", "b"}}
        found = find_outcomes_after_drug(self.TOKENS, self.TIMES, 0, groups)
        self.assertEqual(found, {"one": 3.0, "two": 3.0})

    def test_an_inpatient_outcome_skips_the_outpatient_record(self):
        groups = {"one": {"b"}, "two": {"a"}}
        found = find_outcomes_after_drug(self.TOKENS, self.TIMES, 0, groups, inpatient_outcome_labels={"two", "one"})
        self.assertEqual(found, {"one": None, "two": 3.0})

    def test_the_prebuilt_index_gives_the_same_result(self):
        from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import build_labels_by_token

        groups = {"one": {"a"}, "two": {"b", "x"}}
        index = build_labels_by_token(groups)
        self.assertEqual(
            find_outcomes_after_drug(self.TOKENS, self.TIMES, 0, groups, labels_by_token=index),
            find_outcomes_after_drug(self.TOKENS, self.TIMES, 0, groups),
        )


DAY = 86_400
START = 1_600_000_000.0
SOURCE, COMPARATOR = 100, 200


def patient(person_id, visits):
    """visits: list of (day, visit concept, [tokens])."""
    concept_ids = ["year:2020", "age:50", "8507", "0"]
    epoch_times = [0.0] * 4
    for day, visit_concept, tokens in visits:
        for token in ["[VS]", visit_concept, *tokens, "[VE]"]:
            concept_ids.append(token)
            epoch_times.append(START + day * DAY)
    n = len(concept_ids)
    return {
        "person_id": person_id,
        "concept_ids": concept_ids,
        "epoch_times": epoch_times,
        "ages": [50] * n,
        "input_ids": list(range(n)),
        "value_indicators": [0] * n,
        "values": [0] * n,
    }


class ExtractionWithDefinitionsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "vocab" / "concept_ancestor").mkdir(parents=True)
        ids = [SOURCE, COMPARATOR]
        pl.DataFrame({"ancestor_concept_id": ids, "descendant_concept_id": ids}).write_parquet(
            self.root / "vocab" / "concept_ancestor" / "part.parquet"
        )
        self.definitions = self.root / "definitions.json"
        self.write_definitions(
            [
                {"label": "mi", "concepts": ["600", "601"], "inpatient_only": True},
                {"label": "rash", "concepts": ["700"], "inpatient_only": False},
            ]
        )
        pl.DataFrame(
            [
                patient(1, [(0, "9202", [str(SOURCE)]), (10, "9202", ["600"]), (20, "9201", ["601"]), (30, "9202", ["700"])]),
                patient(2, [(0, "9202", [str(SOURCE)]), (10, "9202", ["600"])]),
            ]
        ).write_parquet(self.root / "patients.parquet")
        self.output = self.root / "out"

    def write_definitions(self, entries):
        full = [
            {
                "name": e["label"],
                "role": "primary",
                "cohort_id": None,
                **e,
            }
            for e in entries
        ]
        self.definitions.write_text(json.dumps(full))

    def run_process(self, **kwargs):
        process(
            patient_sequence_path=str(self.root / "patients.parquet"),
            vocab_path=str(self.root / "vocab"),
            source_ingredient_ids=[SOURCE],
            comparator_ingredient_ids=[COMPARATOR],
            output_dir=self.output,
            min_context_length=1,
            outcome_definitions_path=str(self.definitions),
            **kwargs,
        )

    def observed(self):
        return pl.read_parquet(str(self.output / "observed_outcomes" / "*.parquet")).sort("person_id")

    def test_defined_outcomes_are_extracted_with_their_visit_restriction(self):
        self.run_process()
        observed = self.observed()
        self.assertEqual(observed.columns, ["person_id", "drug_epoch_time", "mi", "rash", "prior__mi", "prior__rash"])
        # the outpatient record on day 10 does not count, the inpatient one on day 20 does
        self.assertEqual(observed["mi"].to_list(), [START + 20 * DAY, None])
        self.assertEqual(observed["rash"].to_list(), [START + 30 * DAY, None])

    def test_a_changed_definitions_file_is_noticed_on_rerun(self):
        self.run_process()
        self.write_definitions([{"label": "rash", "concepts": ["700", "701"], "inpatient_only": False}])
        with self.assertRaisesRegex(RuntimeError, "outcome_definitions"):
            self.run_process()

    def test_the_same_definitions_are_reused(self):
        self.run_process()
        self.run_process()

    def test_a_label_in_both_places_raises(self):
        self.write_definitions([{"label": str(SOURCE), "concepts": ["600"], "inpatient_only": False}])
        with self.assertRaisesRegex(ValueError, "outcome label 100"):
            self.run_process(outcome_concept_ids=[SOURCE])


class HazardRatioWithDefinitionsTest(unittest.TestCase):
    def test_defined_outcomes_and_controls_are_analysed_and_labelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = datetime(2020, 1, 1)
            persons = list(range(1, 61))
            start = datetime(2020, 1, 1).timestamp()
            pl.DataFrame(
                {
                    "person_id": persons,
                    "drug_concept_id": ["100"] * 60,
                    "drug_epoch_time": [start] * 60,
                    "arm": ["source"] * 60,
                    "era_end_epoch_time": [None] * 60,
                    "last_observed_epoch_time": [start + 400 * DAY] * 60,
                },
                schema_overrides={"era_end_epoch_time": pl.Float64},
            ).write_parquet(root / "drug_info.parquet")
            for arm, hit in (("a", 3), ("b", 4)):
                rows = []
                for p in persons:
                    events = [(30, "x"), (200, "x")]
                    if p % hit == 0:
                        events.append((50, "600"))  # the defined primary outcome
                    if p % 5 == 0:
                        events.append((60, "71"))  # a descendant of the negative control
                    rows += [(p, index + timedelta(days=d), c) for d, c in events]
                folder_arm = root / "traj" / arm / "0"
                folder_arm.mkdir(parents=True)
                pl.DataFrame(
                    {
                        "subject_id": [r[0] for r in rows],
                        "trajectory_id": [0] * len(rows),
                        "time": [r[1] for r in rows],
                        "code": [r[2] for r in rows],
                    }
                ).write_parquet(folder_arm / "batch_00000.parquet")
            definitions = [
                {"label": "mi", "name": "MI", "role": "primary", "concepts": ["600"], "inpatient_only": False, "supported": True},
                {"label": "rash", "name": "Rash", "role": "negative_control", "concepts": ["7", "71"], "inpatient_only": False},
            ]
            (root / "definitions.json").write_text(json.dumps(definitions))
            argv = [
                "prog", "--trajectories_dir", str(root / "traj"), "--drug_info_path", str(root / "drug_info.parquet"),
                "--outcome_definitions_path", str(root / "definitions.json"),
                "--arm_a", "a", "--arm_b", "b", "--follow_up_days", "365", "--output_dir", str(root / "out"),
            ]  # fmt: skip
            with mock.patch("sys.argv", argv):
                hr.main()
            summary = pl.read_csv(root / "out" / "hazard_ratio_summary.csv")
            self.assertEqual(summary["outcome_concept_id"].cast(pl.String).to_list(), ["mi", "rash"])
            self.assertEqual(summary["outcome_type"].to_list(), ["primary", "negative_control"])
            self.assertEqual(summary["expected_hr"].to_list(), [None, 1.0])
            self.assertTrue((root / "out" / "control_outcome_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
