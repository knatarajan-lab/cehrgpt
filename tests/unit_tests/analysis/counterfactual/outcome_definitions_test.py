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

def codeset(codeset_id, include=(), descendants=(), exclude=(), exclude_descendants=()):
    return lo.Codeset(codeset_id, list(include), list(descendants), list(exclude), list(exclude_descendants))


def outcome(name, codesets, primary, correlated=(), partial=None):
    """primary / correlated: (domain, codeset id or None) pairs."""
    return lo.OutcomeDefinition(
        name=name,
        codesets={c.codeset_id: c for c in codesets},
        primary=[lo.Criterion(d, [] if i is None else [i]) for d, i in primary],
        correlated=[lo.Criterion(d, [] if i is None else [i]) for d, i in correlated],
        partial=partial,
    )


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "vocab" / "concept_ancestor").mkdir(parents=True)
        # 4 has the descendants 41 and 42, 5 has the descendant 51, 262 has 8717
        pairs = [(4, 4), (4, 41), (4, 42), (5, 5), (5, 51), (262, 262), (262, 8717), (7, 7), (7, 71)]
        pl.DataFrame(
            {"ancestor_concept_id": [a for a, _ in pairs], "descendant_concept_id": [d for _, d in pairs]}
        ).write_parquet(self.root / "vocab" / "concept_ancestor" / "part.parquet")
        self.vocab = str(self.root / "vocab")

    def resolve(self, definition):
        ids = lo.codeset_concept_ids(list(definition.codesets.values()))
        return lo.resolve_outcome(definition, lo.descendants_of(self.vocab, ids))

    def test_descendants_are_expanded_and_exclusions_removed(self):
        resolved = self.resolve(
            outcome(
                "Outcome",
                [codeset(1, include=[9], descendants=[4], exclude=[42])],
                primary=[("Condition Occurrence", 1)],
            )
        )
        self.assertTrue(resolved.supported)
        self.assertEqual(resolved.concepts, ["4", "41", "9"])
        self.assertFalse(resolved.inpatient_only)
        self.assertEqual(resolved.label, "outcome")

    def test_an_excluded_ancestor_removes_its_descendants_too(self):
        resolved = self.resolve(
            outcome(
                "Outcome",
                [codeset(1, descendants=[4, 5], exclude_descendants=[5])],
                primary=[("Condition Occurrence", 1)],
            )
        )
        self.assertEqual(resolved.concepts, ["4", "41", "42"])

    def test_a_visit_criterion_makes_the_outcome_inpatient_only(self):
        resolved = self.resolve(
            outcome(
                "Outcome",
                [codeset(2, descendants=[262]), codeset(3, descendants=[4])],
                primary=[("Condition Occurrence", 3)],
                correlated=[("Visit Occurrence", 2)],
            )
        )
        self.assertTrue(resolved.inpatient_only)
        self.assertEqual(resolved.concepts, ["4", "41", "42"])

    def test_a_hospitalization_is_the_condition_during_an_inpatient_visit(self):
        resolved = self.resolve(
            outcome(
                "Hospitalization",
                [codeset(0, descendants=[262]), codeset(1, descendants=[5])],
                primary=[("Visit Occurrence", 0)],
                correlated=[("Visit Occurrence", 0), ("Condition Occurrence", 1)],
            )
        )
        self.assertTrue(resolved.supported and resolved.inpatient_only)
        self.assertEqual(resolved.concepts, ["5", "51"])

    def test_death_is_the_death_token(self):
        resolved = self.resolve(outcome("All-cause mortality", [], primary=[("Death", None)]))
        self.assertTrue(resolved.supported)
        self.assertEqual(resolved.concepts, [lo.DEATH_TOKEN])

    def test_outcomes_that_cannot_be_matched_on_tokens_say_why(self):
        confirmed = self.resolve(
            outcome(
                "Confirmed",
                [codeset(1, descendants=[4])],
                primary=[("Condition Occurrence", 1)],
                correlated=[("Condition Occurrence", 1)],
            )
        )
        measured = self.resolve(
            outcome("Measured", [codeset(1, descendants=[4])], primary=[("Measurement", 1)])
        )
        self.assertIn("confirming record", confirmed.unsupported_reason)
        self.assertIn("Measurement", measured.unsupported_reason)
        self.assertFalse(measured.supported)

    def test_a_partial_definition_keeps_the_named_concept_sets_and_says_what_is_left_out(self):
        resolved = self.resolve(
            outcome(
                "Two branches",
                [codeset(1, descendants=[4]), codeset(2, descendants=[5])],
                primary=[("Condition Occurrence", 1), ("Condition Occurrence", 2)],
                correlated=[("Measurement", 3)],
                partial={"kept_codeset_ids": [1], "left_out": "the branch with a measured value"},
            )
        )
        self.assertTrue(resolved.supported)
        self.assertEqual(resolved.concepts, ["4", "41", "42"])
        self.assertEqual(resolved.approximation, "left out: the branch with a measured value")

    def test_a_negative_control_is_the_concept_and_its_descendants(self):
        resolved = lo.resolve_negative_control(7, 1001, "Rash", lo.descendants_of(self.vocab, [7]))
        self.assertEqual((resolved.label, resolved.role), ("7", lo.NEGATIVE_CONTROL))
        self.assertEqual(resolved.concepts, ["7", "71"])

    def test_the_definitions_round_trip_and_unsupported_outcomes_are_left_out(self):
        supported = self.resolve(
            outcome("A b", [codeset(1, descendants=[4])], primary=[("Condition Occurrence", 1)])
        )
        measured = self.resolve(
            outcome("C", [codeset(1, descendants=[4])], primary=[("Measurement", 1)])
        )
        path = self.root / "definitions.json"
        lo.write_resolved([supported, measured], path)
        loaded = lo.load_outcome_definitions(str(path))
        self.assertEqual([e["label"] for e in loaded], ["a_b"])
        self.assertEqual(lo.definitions_fingerprint(str(path)), lo.definitions_fingerprint(str(path)))


class BundledDefinitionsTest(unittest.TestCase):
    """The pipeline reads the bundled definitions, so it does not need a checkout of LEGEND."""

    @classmethod
    def setUpClass(cls):
        cls.outcomes = {o.name: o for o in lo.read_outcomes()}

    def test_the_hypertension_outcomes_and_controls_are_bundled(self):
        self.assertEqual((len(self.outcomes), len(lo.negative_controls())), (58, 76))
        self.assertEqual(len({o.cohort_id for o in self.outcomes.values()}), 58)

    def test_the_bundled_files_say_which_version_of_legend_they_come_from(self):
        source = json.loads((study_dir() / "source.json").read_text())
        self.assertEqual(source["repository"], "https://github.com/OHDSI/Legend")
        self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual((source["n_outcomes"], source["n_negative_controls"]), (58, 76))

    def test_acute_myocardial_infarction_keeps_its_exclusion_and_visit_restriction(self):
        mi = self.outcomes["Acute myocardial infarction"]
        self.assertEqual(mi.codesets[3].include_descendants, [4329847])
        self.assertEqual(mi.codesets[3].exclude_descendants, [314666])
        self.assertEqual(mi.visit_codeset_ids, [2])
        self.assertEqual(mi.codesets[2].include_descendants, [262, 9201, 9203])

    def test_hemorrhagic_stroke_members_do_not_include_descendants(self):
        codeset_4 = self.outcomes["Hemorrhagic stroke"].codesets[4]
        self.assertEqual(codeset_4.include, [376713, 432923, 439847])
        self.assertEqual(codeset_4.include_descendants, [])

    def test_a_hospitalization_is_a_visit_with_a_condition_inside(self):
        hospitalization = self.outcomes["Hospitalization with heart failure"]
        self.assertEqual([c.domain for c in hospitalization.primary], ["Visit Occurrence"] * 2)
        self.assertIn("Condition Occurrence", [c.domain for c in hospitalization.correlated])

    def resolve_all(self):
        outcomes = lo.read_outcomes()
        ids = lo.codeset_concept_ids([cs for o in outcomes for cs in o.codesets.values()])
        ids = sorted(set(ids) | {c for c, _, _ in lo.negative_controls()})
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        (root / "vocab" / "concept_ancestor").mkdir(parents=True)
        pl.DataFrame({"ancestor_concept_id": ids, "descendant_concept_id": ids}).write_parquet(
            root / "vocab" / "concept_ancestor" / "part.parquet"
        )
        return {r.name: r for r in lo.resolve_outcomes(outcomes, lo.negative_controls(), str(root / "vocab"))}

    def test_the_outcomes_of_the_paper_that_need_an_inpatient_or_er_visit_have_the_constraint(self):
        needs_visit = [
            "Acute myocardial infarction", "Acute pancreatitis", "Acute renal failure",
            "Anaphylactoid reaction", "Angioedema", "Cardiovascular disease",
            "Gastrointestinal bleeding", "Hemorrhagic stroke", "Hospitalization with heart failure",
            "Ischemic stroke", "Rhabdomyolysis", "Stroke", "Sudden cardiac death",
            "Transient ischemic attack", "Hospitalization with preinfarction syndrome",
        ]  # fmt: skip
        resolved = self.resolve_all()
        self.assertEqual(len(needs_visit), 15)
        for name in needs_visit:
            self.assertTrue(resolved[name].supported, name)
            self.assertTrue(resolved[name].inpatient_only, name)
        self.assertIn("creatine kinase", resolved["Rhabdomyolysis"].approximation)
        self.assertFalse(resolved["Hyperkalemia"].inpatient_only)

    def test_the_two_outcomes_with_a_measured_branch_are_partial_in_the_data(self):
        partial = {name: o.partial for name, o in self.outcomes.items() if o.partial}
        self.assertEqual(sorted(partial), ["Hyperkalemia", "Rhabdomyolysis"])
        self.assertEqual(partial["Rhabdomyolysis"]["kept_codeset_ids"], [10])

    def test_a_study_is_read_from_a_folder_too(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("study.json", "exposures.csv", "outcomes.json"):
                (Path(folder) / name).write_text((study_dir() / name).read_text())
            self.assertEqual(len(lo.read_outcomes(folder)), 58)
            self.assertEqual(lo.negative_controls(folder), [])  # no negative_controls.csv there

    def test_the_numbers_of_supported_and_unsupported_outcomes(self):
        resolved = self.resolve_all()
        self.assertEqual(len(resolved), 134)
        self.assertEqual(sum(r.supported for r in resolved.values()), 121)
        self.assertEqual(resolved["All-cause mortality"].concepts, [lo.DEATH_TOKEN])
        self.assertFalse(resolved["Heart failure"].supported)

    def test_the_command_line_resolves_the_bundled_definitions_for_a_vocabulary(self):
        outcomes = lo.read_outcomes()
        ids = lo.codeset_concept_ids([cs for o in outcomes for cs in o.codesets.values()])
        ids = sorted(set(ids) | {c for c, _, _ in lo.negative_controls()})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "vocab" / "concept_ancestor").mkdir(parents=True)
            pl.DataFrame({"ancestor_concept_id": ids, "descendant_concept_id": ids}).write_parquet(
                root / "vocab" / "concept_ancestor" / "part.parquet"
            )
            argv = ["prog", "--vocab_path", str(root / "vocab"), "--output_dir", str(root / "out")]
            with mock.patch("sys.argv", argv):
                lo.main()
            entries = json.loads((root / "out" / "outcome_definitions.json").read_text())
        self.assertEqual(len(entries), 134)
        self.assertEqual(sum(e["role"] == lo.NEGATIVE_CONTROL for e in entries), 76)


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
                {"label": "unclear", "concepts": ["800"], "inpatient_only": False, "supported": False},
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
                "supported": True,
                "unsupported_reason": "needs a second record",
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
                {"label": "7", "name": "Rash", "role": "negative_control", "concepts": ["7", "71"], "inpatient_only": False, "supported": True},
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
            self.assertEqual(summary["outcome_concept_id"].cast(pl.String).to_list(), ["mi", "7"])
            self.assertEqual(summary["outcome_type"].to_list(), ["primary", "negative_control"])
            self.assertEqual(summary["expected_hr"].to_list(), [None, 1.0])
            self.assertTrue((root / "out" / "control_outcome_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
