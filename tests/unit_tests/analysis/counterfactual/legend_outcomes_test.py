import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import polars as pl

from cehrgpt.analysis.counterfactual import hazard_ratio_estimation as hr
from cehrgpt.analysis.counterfactual import legend_outcomes as lo
from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import (
    find_outcomes_after_drug,
    process,
)

LEGEND = Path("/Users/chaopang/Documents/Legend")


def codeset_sql(codeset_id, include=(), descendants=(), exclude=(), exclude_descendants=()):
    """The shape of the #Codesets inserts that ATLAS generates."""

    def part(plain, with_descendants):
        listed = ",".join(map(str, sorted(set(plain) | set(with_descendants))))
        sql = (
            "( \n  select concept_id from @vocabulary_database_schema.CONCEPT where concept_id in "
            f"({listed})and invalid_reason is null\n"
        )
        if with_descendants:
            sql += (
                "UNION  select c.concept_id\n  from @vocabulary_database_schema.CONCEPT c\n"
                "  join @vocabulary_database_schema.CONCEPT_ANCESTOR ca on c.concept_id = ca.descendant_concept_id\n"
                f"  and ca.ancestor_concept_id in ({','.join(map(str, with_descendants))})\n"
                "  and c.invalid_reason is null\n"
            )
        return sql + "\n) "

    sql = (
        "INSERT INTO #Codesets (codeset_id, concept_id)\n"
        f"SELECT {codeset_id} as codeset_id, c.concept_id FROM (select distinct I.concept_id FROM\n"
        + part(include, descendants)
        + "I\n"
    )
    if exclude or exclude_descendants:
        sql += "LEFT JOIN\n" + part(exclude, exclude_descendants) + "E ON I.concept_id = E.concept_id\nWHERE E.concept_id is null\n"
    return sql + ") C;\n"


def criteria_sql(domain, codeset_id):
    return (
        f"-- Begin {domain} Criteria\nSELECT C.person_id FROM (\n  SELECT x.* FROM T x\n"
        f"  JOIN #Codesets codesets on ((x.concept_id = codesets.concept_id and codesets.codeset_id = {codeset_id}))\n"
        f") C\n-- End {domain} Criteria\n"
    )


def outcome_sql(codesets, primary, correlated=()):
    sql = "".join(codesets)
    sql += "with primary_events as (\n-- Begin Primary Events\n" + "".join(criteria_sql(*p) for p in primary)
    sql += "-- End Primary Events\n)\nSELECT 1\n"
    for domain, codeset_id in correlated:
        sql += "-- Begin Correlated Criteria\n" + criteria_sql(domain, codeset_id) + "-- End Correlated Criteria\n"
    return sql


class ParseTest(unittest.TestCase):
    def test_plain_descendant_and_excluded_concepts_are_told_apart(self):
        sql = codeset_sql(3, include=[11], descendants=[12], exclude=[21], exclude_descendants=[22])
        codeset = lo.parse_codesets(sql)[3]
        self.assertEqual(codeset.include, [11])
        self.assertEqual(codeset.include_descendants, [12])
        self.assertEqual(codeset.exclude, [21])
        self.assertEqual(codeset.exclude_descendants, [22])

    def test_a_codeset_without_exclusions(self):
        codeset = lo.parse_codesets(codeset_sql(0, descendants=[262, 9201]))[0]
        self.assertEqual(codeset.include_descendants, [262, 9201])
        self.assertEqual((codeset.exclude, codeset.exclude_descendants), ([], []))

    def test_several_codesets_in_one_file(self):
        sql = codeset_sql(2, descendants=[262]) + codeset_sql(3, descendants=[4], exclude_descendants=[5])
        self.assertEqual(sorted(lo.parse_codesets(sql)), [2, 3])

    def test_primary_and_correlated_criteria(self):
        sql = outcome_sql(
            [codeset_sql(2, descendants=[262]), codeset_sql(3, descendants=[4])],
            primary=[("Condition Occurrence", 3)],
            correlated=[("Visit Occurrence", 2)],
        )
        outcome = lo.parse_outcome("x", sql)
        self.assertEqual([(c.domain, c.codeset_ids) for c in outcome.primary], [("Condition Occurrence", [3])])
        self.assertEqual([(c.domain, c.codeset_ids) for c in outcome.correlated], [("Visit Occurrence", [2])])
        self.assertEqual(outcome.visit_codeset_ids, [2])


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

    def resolve(self, sql, name="Outcome"):
        outcome = lo.parse_outcome(name, sql)
        ids = lo.codeset_concept_ids(list(outcome.codesets.values()))
        return lo.resolve_outcome(outcome, lo.descendants_of(self.vocab, ids))

    def test_descendants_are_expanded_and_exclusions_removed(self):
        sql = outcome_sql(
            [codeset_sql(1, include=[9], descendants=[4], exclude=[42])],
            primary=[("Condition Occurrence", 1)],
        )
        resolved = self.resolve(sql)
        self.assertTrue(resolved.supported)
        self.assertEqual(resolved.concepts, ["4", "41", "9"])
        self.assertFalse(resolved.inpatient_only)
        self.assertEqual(resolved.label, "outcome")

    def test_an_excluded_ancestor_removes_its_descendants_too(self):
        sql = outcome_sql(
            [codeset_sql(1, descendants=[4, 5], exclude_descendants=[5])],
            primary=[("Condition Occurrence", 1)],
        )
        self.assertEqual(self.resolve(sql).concepts, ["4", "41", "42"])

    def test_a_visit_criterion_makes_the_outcome_inpatient_only(self):
        sql = outcome_sql(
            [codeset_sql(2, descendants=[262]), codeset_sql(3, descendants=[4])],
            primary=[("Condition Occurrence", 3)],
            correlated=[("Visit Occurrence", 2)],
        )
        resolved = self.resolve(sql)
        self.assertTrue(resolved.inpatient_only)
        self.assertEqual(resolved.concepts, ["4", "41", "42"])

    def test_a_hospitalization_is_the_condition_during_an_inpatient_visit(self):
        sql = outcome_sql(
            [codeset_sql(0, descendants=[262]), codeset_sql(1, descendants=[5])],
            primary=[("Visit Occurrence", 0)],
            correlated=[("Visit Occurrence", 0), ("Condition Occurrence", 1)],
        )
        resolved = self.resolve(sql, "Hospitalization")
        self.assertTrue(resolved.supported and resolved.inpatient_only)
        self.assertEqual(resolved.concepts, ["5", "51"])

    def test_death_is_the_death_token(self):
        sql = outcome_sql([], primary=[("Death", 0)]).replace("codesets.codeset_id = 0", "")
        resolved = self.resolve(sql, "All-cause mortality")
        self.assertTrue(resolved.supported)
        self.assertEqual(resolved.concepts, [lo.DEATH_TOKEN])

    def test_outcomes_that_cannot_be_matched_on_tokens_say_why(self):
        confirmed = outcome_sql(
            [codeset_sql(1, descendants=[4])],
            primary=[("Condition Occurrence", 1)],
            correlated=[("Condition Occurrence", 1)],
        )
        measured = outcome_sql([codeset_sql(1, descendants=[4])], primary=[("Measurement", 1)])
        self.assertIn("confirming record", self.resolve(confirmed).unsupported_reason)
        self.assertIn("Measurement", self.resolve(measured).unsupported_reason)
        self.assertFalse(self.resolve(measured).supported)

    def test_a_negative_control_is_the_concept_and_its_descendants(self):
        resolved = lo.resolve_negative_control(7, 1001, "Rash", lo.descendants_of(self.vocab, [7]))
        self.assertEqual((resolved.label, resolved.role), ("7", lo.NEGATIVE_CONTROL))
        self.assertEqual(resolved.concepts, ["7", "71"])

    def test_the_definitions_round_trip_and_unsupported_outcomes_are_left_out(self):
        supported = outcome_sql([codeset_sql(1, descendants=[4])], primary=[("Condition Occurrence", 1)])
        measured = outcome_sql([codeset_sql(1, descendants=[4])], primary=[("Measurement", 1)])
        resolved = [self.resolve(supported, "A b"), self.resolve(measured, "C")]
        path = self.root / "definitions.json"
        lo.write_resolved(resolved, path)
        loaded = lo.load_outcome_definitions(str(path))
        self.assertEqual([e["label"] for e in loaded], ["a_b"])
        self.assertEqual(lo.definitions_fingerprint(str(path)), lo.definitions_fingerprint(str(path)))


@unittest.skipUnless(LEGEND.is_dir(), "the LEGEND checkout is not available")
class LegendFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outcomes = {o.name: o for o in lo.read_outcomes(LEGEND)}

    def test_the_hypertension_outcomes_and_controls_are_read(self):
        self.assertEqual(len(self.outcomes), 58)
        self.assertEqual(len(lo.negative_controls(LEGEND)), 76)

    def test_acute_myocardial_infarction_keeps_its_exclusion_and_visit_restriction(self):
        outcome = self.outcomes["Acute myocardial infarction"]
        codeset = outcome.codesets[3]
        self.assertEqual(codeset.include_descendants, [4329847])
        self.assertEqual(codeset.exclude_descendants, [314666])
        self.assertEqual(outcome.codesets[outcome.visit_codeset_ids[0]].include_descendants, [262, 9201, 9203])

    def test_the_outcomes_of_the_paper_that_need_an_inpatient_or_er_visit_have_the_constraint(self):
        needs_visit = [
            "Acute myocardial infarction", "Acute pancreatitis", "Acute renal failure",
            "Anaphylactoid reaction", "Angioedema", "Cardiovascular disease",
            "Gastrointestinal bleeding", "Hemorrhagic stroke", "Hospitalization with heart failure",
            "Ischemic stroke", "Rhabdomyolysis", "Stroke", "Sudden cardiac death",
            "Transient ischemic attack", "Hospitalization with preinfarction syndrome",
        ]  # fmt: skip
        with tempfile.TemporaryDirectory() as folder:
            ids = lo.codeset_concept_ids([cs for o in self.outcomes.values() for cs in o.codesets.values()])
            (Path(folder) / "concept_ancestor").mkdir()
            pl.DataFrame({"ancestor_concept_id": ids, "descendant_concept_id": ids}).write_parquet(
                Path(folder) / "concept_ancestor" / "part.parquet"
            )
            descendants = lo.descendants_of(folder, ids)
            resolved = {n: lo.resolve_outcome(o, descendants) for n, o in self.outcomes.items()}
        for name in needs_visit:
            self.assertTrue(resolved[name].supported, name)
            self.assertTrue(resolved[name].inpatient_only, name)
        self.assertEqual(len(needs_visit), 15)
        self.assertIn("creatine kinase", resolved["Rhabdomyolysis"].approximation)
        self.assertFalse(resolved["Hyperkalemia"].inpatient_only)

    def test_hemorrhagic_stroke_members_do_not_include_descendants(self):
        codeset = self.outcomes["Hemorrhagic stroke"].codesets[4]
        self.assertEqual(codeset.include, [376713, 432923, 439847])
        self.assertEqual(codeset.include_descendants, [])


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
