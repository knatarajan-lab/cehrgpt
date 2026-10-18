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
from cehrbert.data_generators.hf_data_generator.meds_utils import convert_one_patient
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import config as datasets_config
from transformers import TrainingArguments

from cehrgpt.data.hf_cehrgpt_dataset_mapping import MedToCehrGPTDatasetMapping
from cehrgpt.data.meds_dataset import (
    convert_one_patient_with_rules,
    create_meds_dataset,
    determine_visit_boundary_method,
)
from cehrgpt.data.meds_visit_rules import (
    VisitRules,
    VisitRulesError,
    YamlDrivenConversion,
    build_conversion,
    packaged_rules_for_conversion,
    packaged_rules_path,
    resolve_visit_rules,
)
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

OMOP = "MedsToCehrbertOMOP"
MIMIC = "MedsToBertMimic4"
OMOP_RULES = packaged_rules_path("omop")
MIMIC_RULES = packaged_rules_path("mimic4")


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


def build_database(root, name, events, rules=None):
    """
    Three subjects (train, tuning, held_out) with the same events, as a meds_reader database. The visit rules,
    a dict, are written to the metadata of the MEDS dataset.
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
    if rules is not None:
        with open(os.path.join(source, "metadata", "visit_rules.yaml"), "w") as f:
            yaml.safe_dump(rules, f)
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
        convert = convert_one_patient_with_rules if isinstance(conversion, YamlDrivenConversion) else convert_one_patient
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

    def test_the_packaged_rules_only_declare_event_matching_rules(self):
        with open(OMOP_RULES) as f:
            omop = VisitRules.from_dict(yaml.safe_load(f))
        with open(MIMIC_RULES) as f:
            mimic = VisitRules.from_dict(yaml.safe_load(f))
        self.assertTrue(omop.visit_events)
        self.assertFalse(mimic.visit_events)
        self.assertNotIn("strategy", omop.to_dict())
        self.assertNotIn("strategy", mimic.to_dict())

    def test_the_packaged_rules_match_the_codes_of_the_datasets_they_are_for(self):
        def matches(patterns, code):
            return any(re.match(pattern, code) for pattern in patterns)

        with open(MIMIC_RULES) as f:
            mimic = VisitRules.from_dict(yaml.safe_load(f))
        self.assertTrue(matches(mimic.admission_events, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"))
        self.assertTrue(matches(mimic.discharge_events, "HOSPITAL_DISCHARGE//HOME"))
        self.assertTrue(matches(mimic.ed_admission_events, "TRANSFER_TO//ED//Emergency Department"))
        # the ICU stays are inside a hospital stay
        self.assertFalse(matches(mimic.admission_events, "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"))
        with open(OMOP_RULES) as f:
            omop = VisitRules.from_dict(yaml.safe_load(f))
        for code in ("Visit/OP", "Visit/IP", "Visit/ER", "Visit/ERIP"):
            self.assertTrue(matches(omop.visit_events, code), code)
        # the discharge facilities of the OMOP sample of 713 persons
        for code in ("PCORNet/Generic-NI", "CMS Place of Service/12", "SNOMED/371827001", "NUCC/261Q00000X", "Medicare Specialty/A4"):
            self.assertTrue(matches(omop.discharge_events, code), code)

    def test_a_packaged_name_stands_for_its_file(self):
        mimic = packaged_rules_for_conversion(MIMIC)
        omop = packaged_rules_for_conversion(OMOP)
        for rules in (mimic, omop):
            self.assertIn("Visit/", rules.visit_events)
            self.assertIn("HOSPITAL_ADMISSION//", rules.admission_events)
            self.assertIn("HOSPITAL_DISCHARGE//", rules.discharge_events)
            self.assertIn("CMS Place of Service/12", rules.discharge_events)
        self.assertTrue(mimic.text_events)
        with self.assertRaisesRegex(VisitRulesError, "no packaged visit rules"):
            packaged_rules_path("eicu")
        with self.assertRaisesRegex(VisitRulesError, "packaged rules"):
            packaged_rules_for_conversion("SomeConversionOfMine")

    def test_yaml_builds_one_generic_conversion(self):
        conversion = build_conversion(VisitRules.from_dict(self.valid()))
        self.assertIs(type(conversion), YamlDrivenConversion)

    def test_visit_boundaries_are_detected_from_the_events(self):
        visit_rules = build_conversion(VisitRules.from_dict(self.valid()))
        admission_rules = build_conversion(
            VisitRules.from_dict(dict(admission_events=["ADMIT//"], discharge_events=["DISCHARGE//"]))
        )

        def patient(*events):
            return types.SimpleNamespace(events=list(events))

        def event(code, visit_id=None):
            return types.SimpleNamespace(code=code, visit_id=visit_id)

        self.assertEqual(determine_visit_boundary_method(patient(event("LAB", 12)), visit_rules), "visit_id")
        self.assertEqual(determine_visit_boundary_method(patient(event("Visit/IP")), visit_rules), "visit_id")
        self.assertEqual(
            determine_visit_boundary_method(patient(event("ADMIT//IP"), event("DISCHARGE//HOME")), admission_rules),
            "admission_window",
        )

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
        converted = convert_one_patient_with_rules(patient, conversion, prediction_time=prediction_time)
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

    def folder(self, declared=None):
        folder = os.path.join(self.root, "database")
        os.makedirs(os.path.join(folder, "metadata"), exist_ok=True)
        if declared is not None:
            with open(os.path.join(folder, "metadata", "visit_rules.yaml"), "w") as f:
                yaml.safe_dump(declared, f)
        return folder

    def override(self, values, name="override.yaml"):
        path = os.path.join(self.root, name)
        with open(path, "w") as f:
            yaml.safe_dump(values, f)
        return path

    def test_no_rules_at_all(self):
        self.assertIsNone(resolve_visit_rules(self.folder()))

    def test_the_rules_of_the_dataset(self):
        rules = resolve_visit_rules(self.folder(CUSTOM_RULES))
        self.assertIn("HOSPITAL_ADMISSION//", rules.admission_events)
        self.assertIn("ENC//INPATIENT", rules.admission_events)

    def test_rule_files_add_patterns_with_stable_deduplication(self):
        path = self.override(dict(discharge_events=["OTHER//", "OTHER//"]))
        rules = resolve_visit_rules(self.folder(CUSTOM_RULES), path)
        self.assertIn("HOSPITAL_DISCHARGE//", rules.discharge_events)
        self.assertIn("DISCH//", rules.discharge_events)
        self.assertIn("OTHER//", rules.discharge_events)
        self.assertEqual(rules.discharge_events.count("OTHER//"), 1)
        self.assertIn("Visit/", rules.visit_events)
        self.assertIn("ENC//", rules.visit_events)

    def test_any_valid_rules_file_can_stand_alone(self):
        rules = resolve_visit_rules(self.folder(), self.override(CUSTOM_RULES))
        self.assertIn("Visit/", rules.visit_events)
        self.assertIn("ENC//", rules.visit_events)
        rules = resolve_visit_rules(self.folder(), self.override(dict(discharge_events=["OTHER//"])))
        self.assertIn("HOSPITAL_DISCHARGE//", rules.discharge_events)
        self.assertIn("OTHER//", rules.discharge_events)

    def test_json_is_accepted_too(self):
        path = os.path.join(self.root, "rules.json")
        with open(path, "w") as f:
            json.dump(CUSTOM_RULES, f)
        rules = resolve_visit_rules(self.folder(), path)
        self.assertIn("HOSPITAL_ADMISSION//", rules.admission_events)
        self.assertIn("ENC//INPATIENT", rules.admission_events)


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class VisitsFromRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.omop = build_database(cls.root, "omop", OMOP_EVENTS)
        cls.problem_list = build_database(cls.root, "problem_list", OMOP_EVENTS + [event("ICD9CM/250.00", 2, "condition", 1)])
        cls.mimic = build_database(cls.root, "mimic", MIMIC_EVENTS)
        cls.mimic_ed = build_database(cls.root, "mimic_ed", MIMIC_ED_EVENTS)
        cls.custom = build_database(cls.root, "custom", CUSTOM_EVENTS, rules=CUSTOM_RULES)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def rules(self, path):
        with open(path) as f:
            return VisitRules.from_dict(yaml.safe_load(f), source=path)

    def test_the_omop_rules_make_the_visits_that_the_omop_class_of_cehrbert_makes(self):
        by_rules = visits_of(self.omop, build_conversion(self.rules(OMOP_RULES)))
        by_class = visits_of(self.omop, MedsToCehrbertOMOP(disconnect_problem_list_events=True))
        self.assertEqual(describe(by_rules), describe(by_class))
        self.assertEqual([v[0] for v in describe(by_rules)], ["Visit/IP", "Visit/OP"])
        self.assertEqual(describe(by_rules)[0][3], "CMS Place of Service/12")

    def test_the_mimic_rules_make_the_visits_that_the_mimic_class_of_cehrbert_makes(self):
        by_rules = visits_of(self.mimic, build_conversion(self.rules(MIMIC_RULES), default_visit_id=1))
        by_class = visits_of(self.mimic, MedsToBertMimic4(default_visit_id=1))
        self.assertEqual(describe(by_rules), describe(by_class))
        # the admission and the discharge are one inpatient visit
        self.assertEqual([v[0] for v in describe(by_rules)], ["Visit/IP"])
        self.assertEqual(describe(by_rules)[0][3], "HOSPITAL_DISCHARGE//HOME")

    def test_the_mimic_rules_replicate_the_mimic_class_on_ed_stays_and_numbers_kept_as_text(self):
        by_rules = visits_of(self.mimic_ed, build_conversion(self.rules(MIMIC_RULES), default_visit_id=1))
        by_class = visits_of(self.mimic_ed, MedsToBertMimic4(default_visit_id=1))
        self.assertEqual(by_rules, by_class)
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
            rules = self.rules(OMOP_RULES)
            rules = VisitRules.from_dict({**rules.to_dict(), **rules_change})
            by_rules = visits_of(self.omop, build_conversion(rules, **options))
            class_options = {"disconnect_problem_list_events": True, **options}
            by_class = visits_of(self.omop, MedsToCehrbertOMOP(**class_options))
            self.assertEqual(by_rules, by_class, (rules_change, options))
        excluded = build_conversion(self.rules(OMOP_RULES), meds_exclude_tables=["measurement"])
        self.assertNotIn("LOINC/1", [e["code"] for v in visits_of(self.omop, excluded) for e in v["events"]])

    def test_an_event_far_from_its_visit_is_always_put_where_its_time_is(self):
        def codes_of_the_stay(rules):
            visits = visits_of(self.problem_list, build_conversion(rules))
            stay = [v for v in visits if v["visit_type"] == "Visit/IP"][0]
            return {e["code"] for e in stay["events"]}

        default = self.rules(OMOP_RULES)
        # a problem list entry from 8 days before the stay is not part of the stay
        self.assertNotIn("ICD9CM/250.00", codes_of_the_stay(default))
        self.assertIn("ICD9CM/296.90", codes_of_the_stay(default))
        # and it is not lost: it is in a visit of its own, at its own time
        everything = visits_of(self.problem_list, build_conversion(default))
        entry = [v for v in everything if "ICD9CM/250.00" in {e["code"] for e in v["events"]}][0]
        self.assertEqual(entry["visit_start_datetime"].date(), dt.date(2020, 1, 3))

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

    def test_create_meds_dataset_reads_the_rules_from_the_metadata_of_the_dataset(self):
        args = data_args(self.custom, att_function_type="day", inpatient_att_function_type="day")
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache1")):
            dataset = create_meds_dataset(args)
        # the visits of a patient are stored column by column
        self.assertEqual(set(dataset["train"][0]["visits"]["visit_type"]), {"ENC//INPATIENT", "ENC//OUTPATIENT"})

    def test_a_rules_file_adds_to_the_discharge_rules_of_the_dataset(self):
        override = os.path.join(self.root, "discharge.yaml")
        with open(override, "w") as f:
            yaml.safe_dump(dict(discharge_events=["NO_SUCH//"]), f)
        args = data_args(self.custom)
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache2")):
            dataset = create_meds_dataset(args, rules_file=override)
        visits = dataset["train"][0]["visits"]
        facility = dict(zip(visits["visit_type"], visits["discharge_facility"]))["ENC//INPATIENT"]
        self.assertEqual(facility, "DISCH//HOME")

    def test_a_dataset_without_rules_is_read_with_the_packaged_rules_of_the_conversion_that_is_named(self):
        for database, conversion_type, packaged in (
            (self.omop, OMOP, "omop"),
            (self.mimic, MIMIC, "mimic4"),
        ):
            args = data_args(database, meds_to_cehrbert_conversion_type=conversion_type)
            with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache_" + packaged)):
                with self.assertLogs("transformers", level="WARNING") as logs:
                    dataset = create_meds_dataset(args)
            self.assertTrue(any("declares no visit rules" in line and packaged in line for line in logs.output))
            self.assertGreater(dataset["train"].num_rows, 0)
        # and the visits are the ones of the class of cehrbert
        self.assertEqual(set(dataset["train"][0]["visits"]["visit_type"]), {"Visit/IP"})

    def test_a_packaged_name_can_be_the_rules_file(self):
        args = data_args(self.omop)
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache_named")):
            dataset = create_meds_dataset(args, rules_file="omop")
        self.assertEqual(set(dataset["train"][0]["visits"]["visit_type"]), {"Visit/IP", "Visit/OP"})

@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class PrepareFinetuneDatasetTest(unittest.TestCase):
    """The runners read MEDS through prepare_finetune_dataset: nothing is passed about the shape of the data."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.old_cache = datasets_config.HF_DATASETS_CACHE
        datasets_config.HF_DATASETS_CACHE = os.path.join(cls.root, "hf_cache")
        with open(MIMIC_RULES) as f:
            mimic_rules = yaml.safe_load(f)
        events = MIMIC_EVENTS + [event("HOSPITAL_DISCHARGE//HOME", 14)]
        cls.database = build_database(cls.root, "mimic", events, rules=mimic_rules)
        cls.cohort = os.path.join(cls.root, "cohort")
        os.makedirs(cls.cohort)
        pd.DataFrame(
            {"subject_id": [1, 2, 3], "prediction_time": [dt.datetime(2020, 1, 30)] * 3, "boolean_value": [1, 0, 1]}
        ).to_parquet(os.path.join(cls.cohort, "cohort.parquet"))

    @classmethod
    def tearDownClass(cls):
        datasets_config.HF_DATASETS_CACHE = cls.old_cache
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_the_rules_of_the_dataset_make_a_stay_one_inpatient_visit_without_any_flag(self):
        args = data_args(
            self.database,
            cohort_folder=self.cohort,
            att_function_type="day",
            inpatient_att_function_type="day",
            include_auxiliary_token=True,
        )
        # the default of the arguments is the OMOP conversion: the rules of the dataset are what decides
        self.assertEqual(args.meds_to_cehrbert_conversion_type, "MedsToCehrbertOMOP")
        training_args = TrainingArguments(output_dir=os.path.join(self.root, "out"), report_to="none", use_cpu=True)
        dataset = prepare_finetune_dataset(args, training_args, CehrGPTArguments(), None)
        concept_ids = list(dataset["train"][0]["concept_ids"])
        self.assertEqual(concept_ids.count("[VS]"), 1)
        self.assertEqual(concept_ids.count("[VE]"), 1)
        self.assertIn("i-D1", concept_ids)
        self.assertIn("Visit/IP", concept_ids)
        self.assertIn("HOSPITAL_DISCHARGE//HOME", concept_ids)
