import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
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

from cehrgpt.data.meds_dataset import create_meds_dataset
from cehrgpt.data.meds_visit_rules import (
    VisitRules,
    VisitRulesError,
    build_conversion,
    resolve_visit_rules,
    rules_of_conversion,
)
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

SAMPLE_RULES = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "sample_configs", "meds_visit_rules"
)
OMOP_RULES = os.path.join(SAMPLE_RULES, "omop.yaml")
MIMIC_RULES = os.path.join(SAMPLE_RULES, "mimic4.yaml")


def converter():
    """The meds_reader_convert script of the environment, None if it is not installed."""
    for candidate in (
        os.path.join(os.path.dirname(sys.executable), "meds_reader_convert"),
        shutil.which("meds_reader_convert"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def event(code, day, table=None, visit_id=None, end_day=None):
    return dict(code=code, day=day, table=table, visit_id=visit_id, end_day=end_day)


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
        )
        for subject in subjects
        for e in events
    ]
    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.String,
        "numeric_value": pl.Float32,
        "end": pl.Datetime("us"),
        "table": pl.String,
        "visit_id": pl.Int32,
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
    strategy="visit_id",
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
        return convert_one_patient(db[subject], conversion)["visits"]


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
        values = dict(strategy="visit_id", visit_events=["Visit/"], discharge_events=["X/1"])
        values.update(changes)
        return values

    def test_valid_rules(self):
        rules = VisitRules.from_dict(self.valid(text_events=[dict(code="BP", pattern=r"(\d+)/(\d+)", labels=["S", "D"])]))
        self.assertEqual(rules.strategy, "visit_id")
        self.assertEqual(rules.visit_events, ["Visit/"])
        self.assertEqual(rules.text_events[0].labels, ["S", "D"])
        self.assertEqual(VisitRules.from_dict(rules.to_dict()).to_dict(), rules.to_dict())

    def test_a_strategy_is_needed(self):
        with self.assertRaisesRegex(VisitRulesError, "strategy"):
            VisitRules.from_dict(dict(visit_events=["Visit/"]))
        with self.assertRaisesRegex(VisitRulesError, "strategy"):
            VisitRules.from_dict(self.valid(strategy="by_magic"))

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

    def test_the_example_rules_are_what_cehrbert_has_in_its_code(self):
        # the example files are written from the classes: if either changes, this says so
        for path, conversion in ((OMOP_RULES, MedsToCehrbertOMOP()), (MIMIC_RULES, MedsToBertMimic4(default_visit_id=1))):
            with open(path) as f:
                written = VisitRules.from_dict(yaml.safe_load(f), source=path).to_dict()
            self.assertEqual(written, rules_of_conversion(conversion).to_dict(), path)

    def test_the_strategies_are_the_two_algorithms_of_cehrbert(self):
        self.assertIsInstance(build_conversion(VisitRules.from_dict(self.valid())), MedsToCehrbertOMOP)
        self.assertIsInstance(
            build_conversion(VisitRules.from_dict(self.valid(strategy="admission_window"))), MedsToBertMimic4
        )


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
        self.assertEqual(rules.admission_events, ["ENC//INPATIENT"])

    def test_the_keys_of_a_file_replace_the_ones_of_the_dataset_and_the_others_stay(self):
        path = self.override(dict(discharge_events=["OTHER//"]))
        rules = resolve_visit_rules(self.folder(CUSTOM_RULES), path)
        self.assertEqual(rules.discharge_events, ["OTHER//"])
        self.assertEqual(rules.visit_events, ["ENC//"])
        self.assertEqual(rules.strategy, "visit_id")

    def test_a_file_can_stand_alone_if_it_has_a_strategy(self):
        self.assertEqual(resolve_visit_rules(self.folder(), self.override(CUSTOM_RULES)).visit_events, ["ENC//"])
        with self.assertRaisesRegex(VisitRulesError, "strategy"):
            resolve_visit_rules(self.folder(), self.override(dict(discharge_events=["OTHER//"])))

    def test_json_is_accepted_too(self):
        path = os.path.join(self.root, "rules.json")
        with open(path, "w") as f:
            json.dump(CUSTOM_RULES, f)
        self.assertEqual(resolve_visit_rules(self.folder(), path).admission_events, ["ENC//INPATIENT"])


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class VisitsFromRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.omop = build_database(cls.root, "omop", OMOP_EVENTS)
        cls.mimic = build_database(cls.root, "mimic", MIMIC_EVENTS)
        cls.custom = build_database(cls.root, "custom", CUSTOM_EVENTS, rules=CUSTOM_RULES)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def rules(self, path):
        with open(path) as f:
            return VisitRules.from_dict(yaml.safe_load(f), source=path)

    def test_the_omop_rules_make_the_visits_that_the_omop_class_of_cehrbert_makes(self):
        by_rules = visits_of(self.omop, build_conversion(self.rules(OMOP_RULES)))
        by_class = visits_of(self.omop, MedsToCehrbertOMOP())
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

    def test_a_rules_file_replaces_the_discharge_of_the_dataset(self):
        override = os.path.join(self.root, "discharge.yaml")
        with open(override, "w") as f:
            yaml.safe_dump(dict(discharge_events=["NO_SUCH//"]), f)
        args = data_args(self.custom)
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache2")):
            dataset = create_meds_dataset(args, rules_file=override)
        visits = dataset["train"][0]["visits"]
        facility = dict(zip(visits["visit_type"], visits["discharge_facility"]))["ENC//INPATIENT"]
        self.assertNotEqual(facility, "DISCH//HOME")

    def test_a_dataset_without_rules_is_read_by_the_code_of_cehrbert_with_a_warning(self):
        args = data_args(self.omop)
        with unittest.mock.patch.object(datasets_config, "HF_DATASETS_CACHE", os.path.join(self.root, "cache3")):
            with self.assertLogs("transformers", level="WARNING") as logs:
                dataset = create_meds_dataset(args)
        self.assertTrue(any("No visit rules" in line for line in logs.output))
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
        self.assertIn("Visit/IP", concept_ids)
        self.assertIn("HOSPITAL_DISCHARGE//HOME", concept_ids)
