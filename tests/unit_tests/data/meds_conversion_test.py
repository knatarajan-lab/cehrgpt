import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import meds_reader
import pandas as pd
import polars as pl
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    convert_one_patient,
    get_meds_to_cehrbert_conversion_cls,
)
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import config as datasets_config
from transformers import TrainingArguments

from cehrgpt.data import meds_conversion as mc
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments


def converter():
    """The meds_reader_convert script of the environment, None if not installed."""
    candidates = (
        os.path.join(os.path.dirname(sys.executable), "meds_reader_convert"),
        shutil.which("meds_reader_convert"),
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_database(root, name, codes):
    """Three subjects (train, tuning, held_out) with the given codes, as a meds_reader database."""
    source = os.path.join(root, name)
    os.makedirs(os.path.join(source, "data"))
    os.makedirs(os.path.join(source, "metadata"))
    subjects = (1, 2, 3)
    rows = [
        (subject, dt.datetime(2020, 1, 1 + i), code, None)
        for subject in subjects
        for i, code in enumerate(codes)
    ]
    schema = {
        "subject_id": pl.Int64,
        "time": pl.Datetime("us"),
        "code": pl.String,
        "numeric_value": pl.Float32,
    }
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(
        os.path.join(source, "data", "0.parquet")
    )
    pl.DataFrame(
        {"subject_id": list(subjects), "split": ["train", "tuning", "held_out"]}
    ).write_parquet(os.path.join(source, "metadata", "subject_splits.parquet"))
    with open(os.path.join(source, "metadata", "dataset.json"), "w") as f:
        json.dump(
            {
                "dataset_name": name,
                "dataset_version": "1",
                "etl_name": "t",
                "etl_version": "1",
                "meds_version": "0.4.0",
            },
            f,
        )
    pl.DataFrame(
        {"code": sorted(set(codes)), "description": None, "parent_codes": None},
        schema={
            "code": pl.String,
            "description": pl.String,
            "parent_codes": pl.List(pl.String),
        },
    ).write_parquet(os.path.join(source, "metadata", "codes.parquet"))
    database = os.path.join(root, name + "_db")
    subprocess.run(
        [converter(), source, database, "--num_threads", "1"],
        check=True,
        capture_output=True,
    )
    return database


def data_args(folder, conversion_type=None, **kwargs):
    arguments = dict(
        data_folder=folder,
        dataset_prepared_path=os.path.join(folder, "prepared"),
        is_data_in_meds=True,
        preprocessing_num_workers=1,
    )
    if conversion_type:
        arguments["meds_to_cehrbert_conversion_type"] = conversion_type
    arguments.update(kwargs)
    return DataTrainingArguments(**arguments)


OMOP = "MedsToCehrbertOMOP"
MIMIC = "MedsToBertMimic4"
OMOP_CODES = ["MEDS_BIRTH", "Visit/IP", "Visit/ER", "SNOMED/201826", "LOINC/1"]
MIMIC_CODES = [
    "MEDS_BIRTH",
    "HOSPITAL_ADMISSION//EW EMER.",
    "HOSPITAL_DISCHARGE//HOME",
    "ED_REGISTRATION//X",
    "LAB//50827//UNK",
]
STAY_CODES = ["MEDS_BIRTH", "HOSPITAL_ADMISSION//EW EMER.", "HOSPITAL_DISCHARGE//HOME"]
LOGGER = "transformers"


class DetectConversionTypeTest(unittest.TestCase):
    def test_both_conversions_are_registered(self):
        self.assertEqual(mc.registered_conversion_types(), [MIMIC, OMOP])

    def test_the_conversion_whose_rules_match_the_codes_is_found(self):
        self.assertEqual(mc.detect_conversion_type(OMOP_CODES), OMOP)
        self.assertEqual(mc.detect_conversion_type(MIMIC_CODES), MIMIC)

    def test_the_most_matching_events_decide_in_a_mixed_dataset(self):
        codes = OMOP_CODES + ["HOSPITAL_ADMISSION//A"]
        self.assertEqual(mc.detect_conversion_type(codes), OMOP)

    def test_codes_that_no_rule_matches_give_none(self):
        self.assertIsNone(mc.detect_conversion_type(["MEDS_BIRTH", "LOINC/1"]))
        self.assertIsNone(mc.detect_conversion_type([]))

    def test_a_tie_gives_none(self):
        codes = ["Visit/IP", "HOSPITAL_ADMISSION//A"]
        self.assertIsNone(mc.detect_conversion_type(codes))

    def test_scores_count_the_matching_events(self):
        codes = ["Visit/IP", "Visit/IP", "HOSPITAL_DISCHARGE//X", "LAB/1"]
        self.assertEqual(mc.score_conversion_types(codes), {MIMIC: 1, OMOP: 2})


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class ResolveConversionTypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.omop_db = build_database(cls.root, "omop", OMOP_CODES)
        cls.mimic_db = build_database(cls.root, "mimic", MIMIC_CODES)
        cls.stay_db = build_database(cls.root, "stay", STAY_CODES)
        cls.other_db = build_database(
            cls.root, "other", ["MEDS_BIRTH", "LOINC/1", "RxNorm/2"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_codes_are_read_from_a_real_database(self):
        codes = mc.sample_event_codes(self.mimic_db)
        self.assertEqual(set(codes), set(MIMIC_CODES))

    def test_nothing_has_to_be_passed_for_a_mimic_dataset(self):
        args = data_args(self.mimic_db)
        # the default of the arguments
        self.assertEqual(args.meds_to_cehrbert_conversion_type, OMOP)
        self.assertEqual(mc.resolve_meds_conversion_type(args), MIMIC)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, MIMIC)

    def test_an_omop_dataset_keeps_the_omop_conversion(self):
        args = data_args(self.omop_db)
        self.assertEqual(mc.resolve_meds_conversion_type(args), OMOP)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, OMOP)

    def test_a_conversion_that_is_asked_for_wins_over_the_data(self):
        args = data_args(self.omop_db, MIMIC)
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.assertEqual(mc.resolve_meds_conversion_type(args), MIMIC)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, MIMIC)
        self.assertIn("looks like", logs.output[0])

    def test_data_that_no_conversion_matches_keeps_the_default_and_says_so(self):
        args = data_args(self.other_db)
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.assertEqual(mc.resolve_meds_conversion_type(args), OMOP)
        self.assertIn("No MEDS conversion matches", logs.output[0])

    def test_the_resolved_conversion_keeps_a_mimic_stay_as_one_inpatient_visit(self):
        def visits(conversion_type):
            conversion = get_meds_to_cehrbert_conversion_cls(
                conversion_type, default_visit_id=1
            )
            with meds_reader.SubjectDatabase(self.stay_db) as database:
                return convert_one_patient(database[1], conversion)["visits"]

        # the default conversion cuts the stay at its admission and its discharge
        self.assertEqual(
            [v["visit_type"] for v in visits(OMOP)], ["Visit/OP", "Visit/OP"]
        )
        resolved = mc.resolve_meds_conversion_type(data_args(self.stay_db))
        stay = visits(resolved)
        self.assertEqual([v["visit_type"] for v in stay], ["Visit/IP"])
        self.assertEqual(
            (stay[0]["visit_start_datetime"].day, stay[0]["visit_end_datetime"].day),
            (2, 3),
        )


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class PrepareFinetuneDatasetTest(unittest.TestCase):
    """The runners read MEDS through prepare_finetune_dataset: nothing is passed about the shape of the data."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        # keep the cache of the datasets library out of the home folder
        cls.old_cache = datasets_config.HF_DATASETS_CACHE
        datasets_config.HF_DATASETS_CACHE = os.path.join(cls.root, "hf_cache")
        codes = [
            "MEDS_BIRTH",
            "Gender/F",
            "Race/White",
            "HOSPITAL_ADMISSION//EW EMER.",
            "LAB//1",
            "HOSPITAL_DISCHARGE//HOME",
        ]
        cls.database = build_database(cls.root, "mimic", codes)
        cls.cohort = os.path.join(cls.root, "cohort")
        os.makedirs(cls.cohort)
        pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "prediction_time": [dt.datetime(2020, 1, 6)] * 3,
                "boolean_value": [1, 0, 1],
            }
        ).to_parquet(os.path.join(cls.cohort, "cohort.parquet"))

    @classmethod
    def tearDownClass(cls):
        datasets_config.HF_DATASETS_CACHE = cls.old_cache
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_a_mimic_dataset_keeps_a_stay_as_one_inpatient_visit(self):
        args = data_args(
            self.database,
            cohort_folder=self.cohort,
            att_function_type="day",
            inpatient_att_function_type="day",
            include_auxiliary_token=True,
        )
        self.assertEqual(args.meds_to_cehrbert_conversion_type, OMOP)
        training_args = TrainingArguments(
            output_dir=os.path.join(self.root, "out"), report_to="none", use_cpu=True
        )
        dataset = prepare_finetune_dataset(args, training_args, CehrGPTArguments(), None)
        self.assertEqual(args.meds_to_cehrbert_conversion_type, MIMIC)
        concept_ids = list(dataset["train"][0]["concept_ids"])
        self.assertEqual(concept_ids.count("[VS]"), 1)
        self.assertIn("Visit/IP", concept_ids)
