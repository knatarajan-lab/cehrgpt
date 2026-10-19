import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock

import meds_reader
import pandas as pd
import polars as pl
import yaml
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import (
    MedsToBertMimic4,
    MedsToCehrbertOMOP,
)
from cehrbert.data_generators.hf_data_generator.meds_utils import convert_one_patient as legacy_convert_one_patient
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import config as datasets_config
from transformers import TrainingArguments

from cehrgpt.data.hf_cehrgpt_dataset_mapping import MedToCehrGPTDatasetMapping
from cehrgpt.data.meds_dataset import (
    convert_one_patient,
    create_meds_dataset,
    generate_demographics_and_patient_blocks,
)
from cehrgpt.data.meds_visit_rules import (
    DEFAULT_RULES_PATH,
    VisitRules,
    VisitRulesError,
    YamlDrivenConversion,
    build_conversion,
    resolve_visit_rules,
)
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

OMOP = "MedsToCehrbertOMOP"
MIMIC = "MedsToBertMimic4"
DEFAULT_RULES = DEFAULT_RULES_PATH


def converter():
    """The meds_reader_convert script of the environment, None if it is not installed."""
    for candidate in (
        os.path.join(os.path.dirname(sys.executable), "meds_reader_convert"),
        shutil.which("meds_reader_convert"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def event(code, day, table=None, visit_id=None, end_day=None, text_value=None):
    return dict(code=code, day=day, table=table, visit_id=visit_id, end_day=end_day, text_value=text_value)


def build_database(root, name, events):
    """
    Three subjects (train, tuning, held_out) with the same events, as a meds_reader database. Its metadata is the
    standard one of MEDS: there is no file of visit rules in it.
    """
    source = os.path.join(root, name)
    os.makedirs(os.path.join(source, "data"))
    os.makedirs(os.path.join(source, "metadata"))
    start = dt.datetime(2020, 1, 1)
    subjects = (1, 2, 3)
    rows = [
        dict(
            subject_id=subject,
            time=start + dt.timedelta(days=e["day"]),
            code=e["code"],
            numeric_value=None,
            end=None if e["end_day"] is None else start + dt.timedelta(days=e["end_day"]),
            table=e["table"],
            visit_id=e["visit_id"],
            text_value=e["text_value"],
        )
        for subject in subjects
        # MEDS is sorted by time within a subject
        for e in sorted(events, key=lambda e: e["day"])
    ]
    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.String,
        "numeric_value": pl.Float32,
        "end": pl.Datetime("us"),
        "table": pl.String,
        "visit_id": pl.Int32,
        "text_value": pl.String,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(os.path.join(source, "data", "0.parquet"))
    pl.DataFrame({"subject_id": list(subjects), "split": ["train", "tuning", "held_out"]}).write_parquet(
        os.path.join(source, "metadata", "subject_splits.parquet")
    )
    with open(os.path.join(source, "metadata", "dataset.json"), "w") as f:
        json.dump(
            {"dataset_name": name, "dataset_version": "1", "etl_name": "t", "etl_version": "1", "meds_version": "0.3.3"}, f
        )
    pl.DataFrame(
        {"code": sorted({e["code"] for e in events}), "description": None, "parent_codes": None},
        schema={"code": pl.String, "description": pl.String, "parent_codes": pl.List(pl.String)},
    ).write_parquet(os.path.join(source, "metadata", "codes.parquet"))
    database = os.path.join(root, name + "_db")
    subprocess.run([converter(), source, database, "--num_threads", "1"], check=True, capture_output=True)
    return database


# An OMOP style dataset: the events carry the id of their visit. The discharge is its own event, at the end of the visit
OMOP_EVENTS = [
    event("MEDS_BIRTH", 0, "person"),
    event("Gender/F", 0, "person"),
    event("Race/White", 0, "person"),
    event("Visit/IP", 10, "visit", 1, end_day=13),
    event("ICD9CM/296.90", 10, "condition", 1),
    event("CMS Place of Service/12", 13, "discharge", 1),
    event("Visit/OP", 20, "visit", 2, end_day=20),
    event("LOINC/1", 20, "measurement", 2),
]
# The same dataset with the codes of another ETL: nothing in cehrbert knows them
CUSTOM_EVENTS = [
    event("MEDS_BIRTH", 0, "person"),
    event("Gender/F", 0, "person"),
    event("Race/White", 0, "person"),
    event("ENC//INPATIENT", 10, "visit", 1, end_day=13),
    event("ICD9CM/296.90", 10, "condition", 1),
    event("DISCH//HOME", 13, "discharge", 1),
    event("ENC//OUTPATIENT", 20, "visit", 2, end_day=20),
    event("LOINC/1", 20, "measurement", 2),
]
CUSTOM_RULES = dict(
    visit_events=["ENC//"],
    admission_events=["ENC//INPATIENT"],
    discharge_events=["DISCH//"],
)
# Some events carry the id of their visit and a stay has none: the stay is found by the admission and discharge rules
MIXED_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("Visit/OP", 10, "visit", 1, end_day=10),
    event("LOINC/1", 10, "measurement", 1),
    event("HOSPITAL_ADMISSION//EW EMER.", 20),
    event("LAB//1", 21),
    event("HOSPITAL_DISCHARGE//HOME", 23),
]
# A MIMIC-IV style dataset: no visit ids, the stay is the time between the admission and the discharge
MIMIC_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("HOSPITAL_ADMISSION//EW EMER.", 10),
    event("LAB//1", 11),
    event("HOSPITAL_DISCHARGE//HOME", 13),
]
# The same with the events of an ED stay that becomes an admission, numbers kept as text, an outpatient measurement
# and a second stay
MIMIC_ED_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("Blood Pressure", 5, text_value="134/82"),
    event("Weight (Lbs)", 5, text_value="127.5"),
    event("TRANSFER_TO//ED//Emergency Department", 10),
    event("ED_REGISTRATION//", 10),
    event("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", 10),
    event("LAB//1", 11),
    event("Blood Pressure Sitting", 11, text_value="127/51"),
    event("HOSPITAL_DISCHARGE//HOME", 13),
    event("BMI (kg/m2)", 30, text_value="19.5"),
    event("HOSPITAL_ADMISSION//ELECTIVE//PHYSICIAN REFERRAL", 50),
    event("HOSPITAL_DISCHARGE//SKILLED NURSING FACILITY", 55),
]
def data_args(folder, **extra):
    arguments = dict(
        data_folder=folder,
        dataset_prepared_path=os.path.join(folder, "prepared"),
        is_data_in_meds=True,
        preprocessing_num_workers=1,
    )
    arguments.update(extra)
    return DataTrainingArguments(**arguments)


def visits_of(database, conversion, subject=1):
    with meds_reader.SubjectDatabase(database) as db:
        convert = convert_one_patient if isinstance(conversion, YamlDrivenConversion) else legacy_convert_one_patient
        return convert(db[subject], conversion)["visits"]


def describe(visits):
    return [
        (
            v["visit_type"],
            v["visit_start_datetime"],
            v.get("visit_end_datetime"),
            v.get("discharge_facility"),
            [e["code"] for e in v["events"]],
        )
        for v in visits
    ]


def without_visit_classification(visits):
    """Compare with legacy conversion classes, which do not emit the generic builder's classification field."""
    return [{key: value for key, value in visit.items() if key != "is_inpatient_or_ed"} for visit in visits]


class VisitRulesTest(unittest.TestCase):
    def valid(self, **changes):
        values = dict(visit_events=["Visit/"], discharge_events=["X/1"])
        values.update(changes)
        return values

    def test_valid_rules(self):
        rules = VisitRules.from_dict(self.valid(text_events=[dict(code="BP", pattern=r"(\d+)/(\d+)", labels=["S", "D"])]))
        self.assertEqual(rules.visit_events, ["Visit/"])
        self.assertEqual(rules.text_events[0].labels, ["S", "D"])
        self.assertEqual(VisitRules.from_dict(rules.to_dict()).to_dict(), rules.to_dict())

    def test_disconnecting_events_is_not_configurable(self):
        with self.assertRaisesRegex(VisitRulesError, "Unknown keys"):
            VisitRules.from_dict(self.valid(disconnect_problem_list_events=False))
        self.assertNotIn("disconnect_problem_list_events", VisitRules.from_dict(self.valid()).to_dict())
        conversion = build_conversion(
            VisitRules.from_dict(self.valid()),
            disconnect_problem_list_events=False,
        )
        self.assertTrue(conversion.disconnect_problem_list_events)

    def test_visit_assignment_is_not_a_yaml_switch(self):
        rules = VisitRules.from_dict(dict(visit_events=["Visit/"]))
        self.assertNotIn("strategy", rules.to_dict())
        with self.assertRaisesRegex(VisitRulesError, "Unknown keys"):
            VisitRules.from_dict(self.valid(strategy="visit_id"))

    def test_a_typo_in_a_key_is_an_error_and_not_a_missing_rule(self):
        with self.assertRaisesRegex(VisitRulesError, "discharge_event"):
            VisitRules.from_dict(self.valid(discharge_event=["X/1"]))

    def test_the_patterns_must_be_regular_expressions_in_lists(self):
        with self.assertRaisesRegex(VisitRulesError, "regular expression"):
            VisitRules.from_dict(self.valid(visit_events=["Visit/(" ]))
        with self.assertRaisesRegex(VisitRulesError, "list of codes"):
            VisitRules.from_dict(self.valid(visit_events="Visit/"))

    def test_a_text_event_needs_a_label_for_every_group(self):
        with self.assertRaisesRegex(VisitRulesError, "2 groups but there are 1 labels"):
            VisitRules.from_dict(self.valid(text_events=[dict(code="BP", pattern=r"(\d+)/(\d+)", labels=["S"])]))
        with self.assertRaisesRegex(VisitRulesError, "code, pattern and labels"):
            VisitRules.from_dict(self.valid(text_events=[dict(code="BP")]))

    def test_a_stay_without_a_discharge_ends_after_a_week_by_default_and_the_limit_is_a_positive_number(self):
        self.assertEqual(VisitRules.from_dict({}).max_stay_gap_days, 7.0)
        self.assertEqual(VisitRules.from_dict(dict(max_stay_gap_days=2)).max_stay_gap_days, 2.0)
        for bad in (0, -1, "week", True, None):
            with self.assertRaisesRegex(VisitRulesError, "max_stay_gap_days"):
                VisitRules.from_dict(dict(max_stay_gap_days=bad))

    def test_the_default_rules_only_declare_event_matching_rules(self):
        with open(DEFAULT_RULES) as f:
            rules = VisitRules.from_dict(yaml.safe_load(f))
        self.assertTrue(rules.visit_events)
        self.assertNotIn("strategy", rules.to_dict())

    def test_the_default_rules_match_both_supported_data_formats(self):
        def matches(patterns, code):
            return any(re.match(pattern, code) for pattern in patterns)

        with open(DEFAULT_RULES) as f:
            rules = VisitRules.from_dict(yaml.safe_load(f))
        self.assertTrue(matches(rules.admission_events, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"))
        self.assertTrue(matches(rules.discharge_events, "HOSPITAL_DISCHARGE//HOME"))
        self.assertTrue(matches(rules.ed_admission_events, "TRANSFER_TO//ED//Emergency Department"))
        # the ICU stays are inside a hospital stay
        self.assertFalse(matches(rules.admission_events, "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"))
        for code in ("Visit/OP", "Visit/IP", "Visit/ER", "Visit/ERIP"):
            self.assertTrue(matches(rules.visit_events, code), code)
        # the discharge facilities of the OMOP sample of 713 persons
        for code in ("PCORNet/Generic-NI", "CMS Place of Service/12", "SNOMED/371827001", "NUCC/261Q00000X", "Medicare Specialty/A4"):
            self.assertTrue(matches(rules.discharge_events, code), code)
        self.assertTrue(rules.text_events)

    def test_yaml_builds_one_generic_conversion(self):
        conversion = build_conversion(VisitRules.from_dict(self.valid()))
        self.assertIs(type(conversion), YamlDrivenConversion)

    def test_visit_objects_become_visit_intervisit_and_inpatient_time_tokens(self):
        args = data_args("unused", att_function_type="day", inpatient_att_function_type="day")
        mapping = MedToCehrGPTDatasetMapping(args, include_inpatient_hour_token=True)
        record = {
            "patient_id": 1,
            "birth_datetime": dt.datetime(1990, 1, 1),
            "gender": "Gender/F",
            "race": "Race/White",
            "visits": [
                {
                    "visit_type": "Visit/IP",
                    "visit_start_datetime": dt.datetime(2020, 1, 10, 2),
                    "visit_end_datetime": dt.datetime(2020, 1, 13),
                    "discharge_facility": "HOME",
                    "events": [
                        {"code": "LAB//SAME_DAY", "time": dt.datetime(2020, 1, 10, 6)},
                        {"code": "LAB//NEXT_DAY", "time": dt.datetime(2020, 1, 11, 6)},
                    ],
                },
                {
                    "visit_type": "Visit/IP",
                    "visit_start_datetime": dt.datetime(2020, 1, 20),
                    "visit_end_datetime": dt.datetime(2020, 1, 21),
                    "discharge_facility": "HOME",
                    "events": [{"code": "LAB//SECOND_STAY", "time": dt.datetime(2020, 1, 20, 3)}],
                },
            ],
        }

        concepts = mapping.transform(record)["concept_ids"]
        self.assertEqual(concepts.count("[VS]"), 2)
        self.assertEqual(concepts.count("[VE]"), 2)
        self.assertIn("D7", concepts)
        self.assertIn("i-D1", concepts)
        for hour_token in ("i-H2", "i-H4", "i-H6", "i-H3"):
            self.assertIn(hour_token, concepts)

    def test_an_active_visit_at_prediction_time_has_no_visit_end_token(self):
        conversion = build_conversion(VisitRules.from_dict(dict(admission_events=["ADMIT//"])))

        def raw_event(code, time):
            return types.SimpleNamespace(
                code=code,
                time=time,
                end=None,
                table=None,
                visit_id=None,
                text_value=None,
                numeric_value=None,
                unit=None,
            )

        prediction_time = dt.datetime(2020, 1, 2, 12)
        patient = types.SimpleNamespace(
            subject_id=1,
            events=[
                raw_event("MEDS_BIRTH", dt.datetime(1990, 1, 1)),
                raw_event("Gender/F", dt.datetime(1990, 1, 1)),
                raw_event("Race/White", dt.datetime(1990, 1, 1)),
                raw_event("ADMIT//URGENT", dt.datetime(2020, 1, 1, 2)),
                raw_event("LAB//OBSERVED", dt.datetime(2020, 1, 2, 2)),
            ],
        )
        converted = convert_one_patient(patient, conversion, prediction_time=prediction_time)
        self.assertIsNone(converted["visits"][0]["visit_end_datetime"])
        self.assertTrue(converted["visits"][0]["is_inpatient_or_ed"])

        args = data_args("unused", att_function_type="day", inpatient_att_function_type="day")
        concepts = MedToCehrGPTDatasetMapping(args, include_inpatient_hour_token=True).transform(converted)[
            "concept_ids"
        ]
        self.assertEqual(concepts.count("[VS]"), 1)
        self.assertNotIn("[VE]", concepts)
        self.assertIn("i-D1", concepts)

    def test_yaml_defined_ed_visits_use_inpatient_days_and_optional_hours(self):
        args = data_args("unused", att_function_type="day", inpatient_att_function_type="day")
        record = {
            "patient_id": 1,
            "birth_datetime": dt.datetime(1990, 1, 1),
            "gender": "Gender/F",
            "race": "Race/White",
            "visits": [
                {
                    "visit_type": "CUSTOM_ED//ARRIVAL",
                    "visit_start_datetime": dt.datetime(2020, 1, 1, 2),
                    "visit_end_datetime": dt.datetime(2020, 1, 2, 8),
                    "discharge_facility": "HOME",
                    "is_inpatient_or_ed": True,
                    "events": [{"code": "LAB//ED", "time": dt.datetime(2020, 1, 2, 6)}],
                }
            ],
        }

        with_hours = MedToCehrGPTDatasetMapping(args, include_inpatient_hour_token=True).transform(record)["concept_ids"]
        without_hours = MedToCehrGPTDatasetMapping(args, include_inpatient_hour_token=False).transform(record)[
            "concept_ids"
        ]
        self.assertIn("i-D1", with_hours)
        self.assertIn("i-D1", without_hours)
        self.assertIn("i-H2", with_hours)
        self.assertIn("i-H6", with_hours)
        self.assertFalse(any(token.startswith("i-H") for token in without_hours))

    def test_columnar_visits_keep_the_yaml_derived_inpatient_or_ed_flag(self):
        converted = MedToCehrGPTDatasetMapping.convert_visit_columnar_to_python(
            {
                "visit_start_datetime": [dt.datetime(2020, 1, 1)],
                "visit_end_datetime": [dt.datetime(2020, 1, 2)],
                "visit_type": ["CUSTOM_ED//ARRIVAL"],
                "discharge_facility": ["HOME"],
                "is_inpatient_or_ed": [True],
                "events": [
                    {
                        "time": [],
                        "code": [],
                        "text_value": [],
                        "numeric_value": [],
                        "unit": [],
                        "properties": [],
                    }
                ],
            }
        )
        self.assertTrue(converted[0].is_inpatient_or_ed)


class ResolveVisitRulesTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def rules_file(self, values, name="rules.yaml"):
        path = os.path.join(self.root, name)
        with open(path, "w") as f:
            yaml.safe_dump(values, f)
        return path

    def test_without_a_file_the_defaults_are_used_as_they_are(self):
        rules = resolve_visit_rules()
        self.assertEqual(rules.source, DEFAULT_RULES)
        self.assertIn("Visit/", rules.visit_events)
        self.assertIn("HOSPITAL_ADMISSION//", rules.admission_events)
        self.assertNotIn("ENC//INPATIENT", rules.admission_events)

    def test_a_file_of_the_user_is_applied_on_top_of_the_defaults(self):
        rules = resolve_visit_rules(self.rules_file(CUSTOM_RULES))
        self.assertIn("HOSPITAL_ADMISSION//", rules.admission_events)
        self.assertIn("ENC//INPATIENT", rules.admission_events)
        self.assertIn("Visit/", rules.visit_events)
        self.assertIn("ENC//", rules.visit_events)
        self.assertEqual(rules.source, f"{DEFAULT_RULES} + {self.rules_file(CUSTOM_RULES)}")

    def test_rule_files_add_patterns_with_stable_deduplication(self):
        rules = resolve_visit_rules(self.rules_file(dict(discharge_events=["OTHER//", "OTHER//", "HOSPITAL_DISCHARGE//"])))
        self.assertIn("OTHER//", rules.discharge_events)
        self.assertEqual(rules.discharge_events.count("OTHER//"), 1)
        self.assertEqual(rules.discharge_events.count("HOSPITAL_DISCHARGE//"), 1)

    def test_a_rules_file_does_not_have_to_repeat_the_defaults(self):
        rules = resolve_visit_rules(self.rules_file(dict(discharge_events=["OTHER//"])))
        self.assertIn("HOSPITAL_DISCHARGE//", rules.discharge_events)
        self.assertIn("Visit/", rules.visit_events)

    def test_json_is_accepted_too(self):
        path = os.path.join(self.root, "rules.json")
        with open(path, "w") as f:
            json.dump(CUSTOM_RULES, f)
        self.assertIn("ENC//INPATIENT", resolve_visit_rules(path).admission_events)

    def test_a_file_that_is_not_valid_rules_is_an_error(self):
        with self.assertRaisesRegex(VisitRulesError, "regular expression"):
            resolve_visit_rules(self.rules_file(dict(visit_events=["Visit/("])))
        with self.assertRaisesRegex(VisitRulesError, "discharge_event"):
            resolve_visit_rules(self.rules_file(dict(discharge_event=["X//"])))


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class VisitsFromRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.omop = build_database(cls.root, "omop", OMOP_EVENTS)
        cls.problem_list = build_database(cls.root, "problem_list", OMOP_EVENTS + [event("ICD9CM/250.00", 2, "condition", 1)])
        cls.mimic = build_database(cls.root, "mimic", MIMIC_EVENTS)
        cls.mimic_ed = build_database(cls.root, "mimic_ed", MIMIC_ED_EVENTS)
        cls.custom = build_database(cls.root, "custom", CUSTOM_EVENTS)
        cls.mixed = build_database(cls.root, "mixed", MIXED_EVENTS)
        # the rules of the user for the codes of that dataset
        cls.custom_rules = os.path.join(cls.root, "custom_rules.yaml")
        with open(cls.custom_rules, "w") as f:
            yaml.safe_dump(CUSTOM_RULES, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def rules(self, path):
        with open(path) as f:
            return VisitRules.from_dict(yaml.safe_load(f), source=path)

    def test_the_omop_rules_make_the_visits_that_the_omop_class_of_cehrbert_makes(self):
        by_rules = visits_of(self.omop, build_conversion(self.rules(DEFAULT_RULES)))
        by_class = visits_of(self.omop, MedsToCehrbertOMOP(disconnect_problem_list_events=True))
        self.assertEqual(describe(by_rules), describe(by_class))
        self.assertEqual([v[0] for v in describe(by_rules)], ["Visit/IP", "Visit/OP"])
        self.assertEqual(describe(by_rules)[0][3], "CMS Place of Service/12")

    def test_the_mimic_rules_make_the_visits_that_the_mimic_class_of_cehrbert_makes(self):
        by_rules = visits_of(self.mimic, build_conversion(self.rules(DEFAULT_RULES), default_visit_id=1))
        by_class = visits_of(self.mimic, MedsToBertMimic4(default_visit_id=1))
        self.assertEqual(describe(by_rules), describe(by_class))
        # the admission and the discharge are one inpatient visit
        self.assertEqual([v[0] for v in describe(by_rules)], ["Visit/IP"])
        self.assertEqual(describe(by_rules)[0][3], "HOSPITAL_DISCHARGE//HOME")

    def test_the_mimic_rules_replicate_the_mimic_class_on_ed_stays_and_numbers_kept_as_text(self):
        by_rules = visits_of(self.mimic_ed, build_conversion(self.rules(DEFAULT_RULES), default_visit_id=1))
        by_class = visits_of(self.mimic_ed, MedsToBertMimic4(default_visit_id=1))
        self.assertEqual(without_visit_classification(by_rules), by_class)
        codes = {e["code"] for v in by_rules for e in v["events"]}
        # the text events are made numeric by the rules: one event per group of the pattern
        self.assertIn("Systolic Blood Pressure", codes)
        self.assertIn("Diastolic Blood Pressure Sitting", codes)
        self.assertNotIn("Blood Pressure", codes)

    def test_the_omop_rules_replicate_the_omop_class_with_its_options(self):
        for rules_change, options in (
            ({}, {}),
            ({}, {"meds_exclude_tables": ["measurement"]}),
        ):
            rules = self.rules(DEFAULT_RULES)
            rules = VisitRules.from_dict({**rules.to_dict(), **rules_change})
            by_rules = visits_of(self.omop, build_conversion(rules, **options))
            class_options = {"disconnect_problem_list_events": True, **options}
            by_class = visits_of(self.omop, MedsToCehrbertOMOP(**class_options))
            self.assertEqual(without_visit_classification(by_rules), by_class, (rules_change, options))
        excluded = build_conversion(self.rules(DEFAULT_RULES), meds_exclude_tables=["measurement"])
        self.assertNotIn("LOINC/1", [e["code"] for v in visits_of(self.omop, excluded) for e in v["events"]])

    def test_an_event_far_from_its_visit_is_always_put_where_its_time_is(self):
        def codes_of_the_stay(rules):
            visits = visits_of(self.problem_list, build_conversion(rules))
            stay = [v for v in visits if v["visit_type"] == "Visit/IP"][0]
            return {e["code"] for e in stay["events"]}

        default = self.rules(DEFAULT_RULES)
        # a problem list entry from 8 days before the stay is not part of the stay
        self.assertNotIn("ICD9CM/250.00", codes_of_the_stay(default))
        self.assertIn("ICD9CM/296.90", codes_of_the_stay(default))
        # and it is not lost: it is in a visit of its own, at its own time
        everything = visits_of(self.problem_list, build_conversion(default))
        entry = [v for v in everything if "ICD9CM/250.00" in {e["code"] for e in v["events"]}][0]
        self.assertEqual(entry["visit_start_datetime"].date(), dt.date(2020, 1, 3))

    def test_visit_ids_and_rules_build_the_visits_of_one_patient_together(self):
        conversion = YamlDrivenConversion(resolve_visit_rules(), default_visit_id=1)
        with meds_reader.SubjectDatabase(self.mixed) as database:
            visits = convert_one_patient(database[1], conversion)["visits"]
            _, blocks = generate_demographics_and_patient_blocks(database[1], conversion)
        outpatient, stay = visits
        # the visit with an id is built from its id
        self.assertEqual(outpatient["visit_type"], "Visit/OP")
        self.assertEqual([e["code"] for e in outpatient["events"]], ["LOINC/1"])
        # the events without one are joined into a stay by the admission and discharge rules
        self.assertEqual(stay["visit_type"], "Visit/IP")
        self.assertTrue(stay["is_inpatient_or_ed"])
        self.assertEqual((stay["visit_start_datetime"].day, stay["visit_end_datetime"].day), (21, 24))
        self.assertEqual(stay["discharge_facility"], "HOSPITAL_DISCHARGE//HOME")
        self.assertIn("LAB//1", [e["code"] for e in stay["events"]])
        # a block that the rules make never takes the id of a visit
        from_the_id = [block for block in blocks if "LOINC/1" in {e.code for e in block.events}]
        self.assertEqual([block.visit_id for block in from_the_id], [1])
        self.assertTrue(all(block.visit_id > 1 for block in blocks if block not in from_the_id))

    def test_the_codes_of_a_dataset_that_cehrbert_does_not_know_are_read_from_its_rules(self):
        rules = VisitRules.from_dict(CUSTOM_RULES)
        by_rules = describe(visits_of(self.custom, build_conversion(rules)))
        self.assertEqual([v[0] for v in by_rules], ["ENC//INPATIENT", "ENC//OUTPATIENT"])
        inpatient = by_rules[0]
        self.assertEqual(inpatient[1], dt.datetime(2020, 1, 11))
        self.assertEqual(inpatient[2], dt.datetime(2020, 1, 14))
        self.assertEqual(inpatient[3], "DISCH//HOME")
        # the code of cehrbert does not see the discharge of the same data
        by_class = describe(visits_of(self.custom, MedsToCehrbertOMOP()))
        self.assertNotEqual(by_class[0][3], "DISCH//HOME")

    def _discharge_facility(self, dataset):
        visits = dataset["train"][0]["visits"]
        return dict(zip(visits["visit_type"], visits["discharge_facility"]))["ENC//INPATIENT"]

    def test_a_dataset_with_other_codes_is_read_with_the_rules_file_of_the_user(self):
        args = data_args(self.custom, att_function_type="day", inpatient_att_function_type="day")
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache_with")):
            with_rules = create_meds_dataset(args, rules_file=self.custom_rules)
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache_without")):
            without_rules = create_meds_dataset(args)
        # the visits of a patient are stored column by column
        self.assertEqual(set(with_rules["train"][0]["visits"]["visit_type"]), {"ENC//INPATIENT", "ENC//OUTPATIENT"})
        self.assertEqual(self._discharge_facility(with_rules), "DISCH//HOME")
        # the defaults do not know the discharge codes of this dataset
        self.assertNotEqual(self._discharge_facility(without_rules), "DISCH//HOME")

    def test_a_rules_file_in_the_metadata_of_a_dataset_is_not_read(self):
        # not a standard file of MEDS: whatever is left there by hand or by another tool does nothing
        metadata = os.path.join(self.custom, "metadata")
        with open(os.path.join(metadata, "visit_rules.yaml"), "w") as f:
            yaml.safe_dump(CUSTOM_RULES, f)
        pl.DataFrame({"discharge_events": [["DISCH//"]]}).write_parquet(os.path.join(metadata, "visit_rules.parquet"))
        self.addCleanup(lambda: [os.remove(os.path.join(metadata, n)) for n in ("visit_rules.yaml", "visit_rules.parquet")])
        args = data_args(self.custom)
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache_stray")):
            dataset = create_meds_dataset(args)
        self.assertNotEqual(self._discharge_facility(dataset), "DISCH//HOME")

    def test_a_dataset_without_rules_uses_the_same_defaults_for_both_data_formats(self):
        for database, conversion_type, name, visit_types in (
            (self.omop, OMOP, "omop", {"Visit/IP", "Visit/OP"}),
            (self.mimic, MIMIC, "mimic4", {"Visit/IP"}),
        ):
            args = data_args(database, meds_to_cehrbert_conversion_type=conversion_type)
            with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache_" + name)):
                dataset = create_meds_dataset(args)
            self.assertGreater(dataset["train"].num_rows, 0)
            self.assertEqual(set(dataset["train"][0]["visits"]["visit_type"]), visit_types)

# MIMIC-IV style stays whose discharge is missing from the record
NO_DISCHARGE_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("HOSPITAL_ADMISSION//EW EMER.", 10),
    event("LAB//1", 11),
    event("LAB//2", 12),
    event("LAB//3", 100),
    event("LAB//4", 200),
    event("LAB//5", 300),
]
DISCHARGE_AFTER_A_LONG_GAP_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("HOSPITAL_ADMISSION//EW EMER.", 10),
    event("LAB//1", 11),
    event("HOSPITAL_DISCHARGE//HOME", 25),
]
TWO_ADMISSIONS_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("HOSPITAL_ADMISSION//EW EMER.", 10),
    event("LAB//1", 11),
    event("HOSPITAL_ADMISSION//ELECTIVE//PHYSICIAN REFERRAL", 40),
    event("LAB//2", 41),
    event("HOSPITAL_DISCHARGE//HOME", 43),
]
LONE_ADMISSION_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("HOSPITAL_ADMISSION//EW EMER.", 10),
    event("LAB//1", 200),
]
ED_THEN_A_LATER_ADMISSION_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("TRANSFER_TO//ED//Emergency Department", 10),
    event("LAB//ED", 10),
    event("HOSPITAL_ADMISSION//ELECTIVE//PHYSICIAN REFERRAL", 14),
    event("LAB//1", 15),
    event("HOSPITAL_DISCHARGE//HOME", 17),
]
BACK_TO_BACK_STAYS_EVENTS = [
    event("MEDS_BIRTH", 0),
    event("Gender/F", 0),
    event("Race/White", 0),
    event("HOSPITAL_ADMISSION//EU OBSERVATION//EMERGENCY ROOM", 10),
    event("HOSPITAL_DISCHARGE//UNK", 11),
    event("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", 11),
    event("LAB//1", 13),
    event("HOSPITAL_DISCHARGE//HOME", 15),
]


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class StaysWithoutADischargeTest(unittest.TestCase):
    """A hospitalization whose discharge is not in the record must end, and not last until the cutoff."""

    START = dt.datetime(2020, 1, 1)

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.no_discharge = build_database(cls.root, "no_discharge", NO_DISCHARGE_EVENTS)
        cls.late_discharge = build_database(cls.root, "late_discharge", DISCHARGE_AFTER_A_LONG_GAP_EVENTS)
        cls.two_admissions = build_database(cls.root, "two_admissions", TWO_ADMISSIONS_EVENTS)
        cls.lone_admission = build_database(cls.root, "lone_admission", LONE_ADMISSION_EVENTS)
        cls.back_to_back = build_database(cls.root, "back_to_back", BACK_TO_BACK_STAYS_EVENTS)
        cls.ed_then_admission = build_database(cls.root, "ed_then_admission", ED_THEN_A_LATER_ADMISSION_EVENTS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def visits(self, database, cutoff_day=None, **rules):
        path = None
        if rules:
            path = os.path.join(self.root, "rules_%s.yaml" % "_".join(f"{k}{v}" for k, v in rules.items()))
            with open(path, "w") as f:
                yaml.safe_dump(rules, f)
        conversion = YamlDrivenConversion(resolve_visit_rules(path), default_visit_id=1)
        cutoff = None if cutoff_day is None else self.START + dt.timedelta(days=cutoff_day)
        with meds_reader.SubjectDatabase(database) as db:
            visits = convert_one_patient(db[1], conversion, prediction_time=cutoff)["visits"]

        def day(value):
            return None if value is None else (value - self.START).days

        return [(v["visit_type"], day(v["visit_start_datetime"]), day(v["visit_end_datetime"])) for v in visits]

    def test_a_discharge_and_the_next_admission_on_the_same_day_are_two_stays(self):
        self.assertEqual(self.visits(self.back_to_back), [("Visit/IP", 10, 11), ("Visit/IP", 11, 15)])

    def test_an_ed_visit_days_before_an_admission_does_not_swallow_the_stay(self):
        # the ED is more than 24 hours before the admission, so it is a visit of its own and the stay still happens
        self.assertEqual(
            self.visits(self.ed_then_admission),
            [("TRANSFER_TO//ED//Emergency Department", 10, 10), ("Visit/IP", 14, 17)],
        )

    def test_the_stay_ends_where_its_events_stop_and_the_later_events_are_not_part_of_it(self):
        self.assertEqual(
            self.visits(self.no_discharge),
            [("Visit/IP", 10, 12), ("Visit/OP", 100, 100), ("Visit/OP", 200, 200), ("Visit/OP", 300, 300)],
        )

    def test_a_cutoff_months_after_the_last_event_of_the_stay_does_not_leave_it_open(self):
        self.assertEqual(
            self.visits(self.no_discharge, cutoff_day=250),
            [("Visit/IP", 10, 12), ("Visit/OP", 100, 100), ("Visit/OP", 200, 200)],
        )

    def test_a_cutoff_right_after_the_last_event_of_the_stay_leaves_it_open(self):
        self.assertEqual(self.visits(self.no_discharge, cutoff_day=13), [("Visit/IP", 10, None)])
        # a week later is still within the limit, a day more is not
        self.assertEqual(self.visits(self.no_discharge, cutoff_day=19), [("Visit/IP", 10, None)])
        self.assertEqual(self.visits(self.no_discharge, cutoff_day=20), [("Visit/IP", 10, 12)])

    def test_a_discharge_ends_a_stay_however_long_it_took(self):
        # 14 days without an event inside the stay: the discharge still closes it
        self.assertEqual(
            self.visits(self.late_discharge),
            [("Visit/IP", 10, 25)],
        )

    def test_a_stay_without_a_discharge_ends_before_the_next_admission(self):
        self.assertEqual(
            self.visits(self.two_admissions),
            [("Visit/IP", 10, 11), ("Visit/IP", 40, 43)],
        )

    def test_an_admission_with_nothing_after_it_is_an_inpatient_visit_of_its_own(self):
        self.assertEqual(self.visits(self.lone_admission), [("Visit/IP", 10, 10), ("Visit/OP", 200, 200)])

    def test_the_limit_is_a_rule_of_the_user(self):
        # longer: the stay reaches the events of day 100 (88 days later) only with a limit of that size
        self.assertEqual(self.visits(self.no_discharge, max_stay_gap_days=100)[0], ("Visit/IP", 10, 300))
        # shorter: even the next day is too far
        self.assertEqual(self.visits(self.no_discharge, max_stay_gap_days=0.5)[:2], [("Visit/IP", 10, 10), ("Visit/OP", 11, 11)])


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class PrepareFinetuneDatasetTest(unittest.TestCase):
    """The runners read MEDS through prepare_finetune_dataset: nothing is passed about the shape of the data."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.old_cache = datasets_config.HF_DATASETS_CACHE
        datasets_config.HF_DATASETS_CACHE = os.path.join(cls.root, "hf_cache")
        events = MIMIC_EVENTS + [event("HOSPITAL_DISCHARGE//HOME", 14)]
        cls.database = build_database(cls.root, "mimic", events)
        cls.cohort = os.path.join(cls.root, "cohort")
        os.makedirs(cls.cohort)
        pd.DataFrame(
            {"subject_id": [1, 2, 3], "prediction_time": [dt.datetime(2020, 1, 30)] * 3, "boolean_value": [1, 0, 1]}
        ).to_parquet(os.path.join(cls.cohort, "cohort.parquet"))

    @classmethod
    def tearDownClass(cls):
        datasets_config.HF_DATASETS_CACHE = cls.old_cache
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_defaults_make_a_stay_one_inpatient_visit_without_any_flag(self):
        args = data_args(
            self.database,
            cohort_folder=self.cohort,
            att_function_type="day",
            inpatient_att_function_type="day",
            include_auxiliary_token=True,
        )
        # the default of the arguments is the OMOP conversion: the defaults decide, the argument does not
        self.assertEqual(args.meds_to_cehrbert_conversion_type, "MedsToCehrbertOMOP")
        training_args = TrainingArguments(output_dir=os.path.join(self.root, "out"), report_to="none", use_cpu=True)
        dataset = prepare_finetune_dataset(args, training_args, CehrGPTArguments(), None)
        concept_ids = list(dataset["train"][0]["concept_ids"])
        self.assertEqual(concept_ids.count("[VS]"), 1)
        self.assertEqual(concept_ids.count("[VE]"), 1)
        self.assertIn("i-D1", concept_ids)
        self.assertIn("Visit/IP", concept_ids)
        self.assertIn("HOSPITAL_DISCHARGE//HOME", concept_ids)
