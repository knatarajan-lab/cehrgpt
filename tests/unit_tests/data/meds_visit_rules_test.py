import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import meds_reader
import polars as pl
import yaml
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from dateutil.relativedelta import relativedelta

from cehrgpt.data.hf_cehrgpt_dataset_mapping import MedToCehrGPTDatasetMapping, _age_in_years
from cehrgpt.data.meds_dataset import convert_one_patient
from cehrgpt.data.meds_visit_rules import VisitRules, YamlDrivenConversion, resolve_visit_rules


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


def build_database(root, name, events):
    """One subject with the events, as a meds_reader database."""
    source = os.path.join(root, name)
    os.makedirs(os.path.join(source, "data"))
    os.makedirs(os.path.join(source, "metadata"))
    start = dt.datetime(2020, 1, 1)
    rows = [
        dict(
            subject_id=1,
            time=start + dt.timedelta(days=e["day"]),
            code=e["code"],
            numeric_value=None,
            end=None if e["end_day"] is None else start + dt.timedelta(days=e["end_day"]),
            table=e["table"],
            visit_id=e["visit_id"],
            text_value=None,
        )
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
    pl.DataFrame({"subject_id": [1], "split": ["train"]}).write_parquet(
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


DEMOGRAPHICS = [event("MEDS_BIRTH", 0), event("Gender/F", 0), event("Race/White", 0)]
# A MIMIC-IV style stay whose discharge is missing from the record
NO_DISCHARGE_EVENTS = DEMOGRAPHICS + [
    event("HOSPITAL_ADMISSION//EW EMER.", 10),
    event("LAB//1", 11),
    event("LAB//2", 12),
    event("LAB//3", 100),
]
VISIT_WITH_AN_END_EVENTS = DEMOGRAPHICS + [
    event("Visit/IP", 10, "visit", 1, end_day=15),
    event("LAB//1", 11, "measurement", 1),
    event("LAB//2", 13, "measurement", 1),
]
ED_THEN_AN_ADMISSION_EVENTS = DEMOGRAPHICS + [
    event("TRANSFER_TO//ED//Emergency Department", 10),
    event("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", 10),
    event("LAB//1", 11),
    event("HOSPITAL_DISCHARGE//HOME", 12),
]
ED_THEN_A_LATER_ADMISSION_EVENTS = DEMOGRAPHICS + [
    event("TRANSFER_TO//ED//Emergency Department", 10),
    event("LAB//ED", 10),
    event("HOSPITAL_ADMISSION//ELECTIVE//PHYSICIAN REFERRAL", 14),
    event("LAB//1", 15),
    event("HOSPITAL_DISCHARGE//HOME", 17),
]
BACK_TO_BACK_STAYS_EVENTS = DEMOGRAPHICS + [
    event("HOSPITAL_ADMISSION//EU OBSERVATION//EMERGENCY ROOM", 10),
    event("HOSPITAL_DISCHARGE//UNK", 11),
    event("HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", 11),
    event("LAB//1", 13),
    event("HOSPITAL_DISCHARGE//HOME", 15),
]


class OpenVisitTokensTest(unittest.TestCase):
    def test_a_visit_open_at_the_prediction_time_has_no_visit_end_or_discharge_token(self):
        conversion = YamlDrivenConversion(VisitRules.from_dict(dict(admission_events=["ADMIT//"])))

        def raw_event(code, time):
            return types.SimpleNamespace(
                code=code, time=time, end=None, table=None, visit_id=None, text_value=None, numeric_value=None, unit=None
            )

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
        converted = convert_one_patient(patient, conversion, prediction_time=dt.datetime(2020, 1, 2, 12))
        self.assertIsNone(converted["visits"][0]["visit_end_datetime"])

        args = DataTrainingArguments(
            data_folder="unused",
            dataset_prepared_path="unused",
            att_function_type="day",
            inpatient_att_function_type="day",
            include_auxiliary_token=True,
        )
        concepts = MedToCehrGPTDatasetMapping(args).transform(converted)["concept_ids"]
        self.assertEqual(concepts.count("[VS]"), 1)
        self.assertIn("i-D1", concepts)
        # the stay goes on at the prediction time: it is neither ended nor discharged
        self.assertEqual(concepts[-1], "LAB//OBSERVED")


@unittest.skipIf(converter() is None, "meds_reader_convert is not installed")
class StayBoundariesTest(unittest.TestCase):
    START = dt.datetime(2020, 1, 1)

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.no_discharge = build_database(cls.root, "no_discharge", NO_DISCHARGE_EVENTS)
        cls.visit_with_an_end = build_database(cls.root, "visit_with_an_end", VISIT_WITH_AN_END_EVENTS)
        cls.ed_then_admission_within_a_day = build_database(cls.root, "ed_within_a_day", ED_THEN_AN_ADMISSION_EVENTS)
        cls.ed_then_admission = build_database(cls.root, "ed_then_admission", ED_THEN_A_LATER_ADMISSION_EVENTS)
        cls.back_to_back = build_database(cls.root, "back_to_back", BACK_TO_BACK_STAYS_EVENTS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def visits(self, database, cutoff_day=None, **rules):
        path = None
        if rules:
            path = os.path.join(self.root, "rules.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(rules, f)
        conversion = YamlDrivenConversion(resolve_visit_rules(path), default_visit_id=1)
        cutoff = None if cutoff_day is None else self.START + dt.timedelta(days=cutoff_day)
        with meds_reader.SubjectDatabase(database) as db:
            visits = convert_one_patient(db[1], conversion, prediction_time=cutoff)["visits"]

        def day(value):
            return None if value is None else (value - self.START).days

        return [(v["visit_type"], day(v["visit_start_datetime"]), day(v["visit_end_datetime"])) for v in visits]

    def test_an_ed_visit_followed_by_an_admission_is_one_emergency_room_and_inpatient_visit(self):
        self.assertEqual(self.visits(self.ed_then_admission_within_a_day), [("Visit/ERIP", 10, 12)])

    def test_an_ed_visit_days_before_an_admission_does_not_swallow_the_stay(self):
        self.assertEqual(
            self.visits(self.ed_then_admission),
            [("TRANSFER_TO//ED//Emergency Department", 10, 10), ("Visit/IP", 14, 17)],
        )

    def test_a_discharge_and_the_next_admission_on_the_same_day_are_two_stays(self):
        self.assertEqual(self.visits(self.back_to_back), [("Visit/IP", 10, 11), ("Visit/IP", 11, 15)])

    def test_nothing_after_the_prediction_time_is_in_the_visits(self):
        # the lab of day 13 and the end of day 15 are after the cutoff of day 12
        self.assertEqual(self.visits(self.visit_with_an_end, cutoff_day=12), [("Visit/IP", 10, None)])
        self.assertEqual(self.visits(self.visit_with_an_end, cutoff_day=20), [("Visit/IP", 10, 15)])

    def test_a_stay_without_a_discharge_ends_after_max_stay_gap_days_without_events(self):
        # the default gap is 3 days: the stay is open until day 15 and ends at its last event after that
        self.assertEqual(self.visits(self.no_discharge, cutoff_day=15), [("Visit/IP", 10, None)])
        self.assertEqual(self.visits(self.no_discharge, cutoff_day=16), [("Visit/IP", 10, 12)])
        # with a longer gap the stay reaches the event of day 100
        self.assertEqual(self.visits(self.no_discharge, max_stay_gap_days=100), [("Visit/IP", 10, 100)])


class AgeInYearsTest(unittest.TestCase):
    def test_it_agrees_with_relativedelta(self):
        births = [dt.datetime(2000, 2, 29, 13), dt.datetime(1980, 5, 17, 8, 30), dt.datetime(1999, 12, 31, 23, 59, 59)]
        for birth in births:
            for days in (-3000, -1, 0, 1, 364, 365, 366, 6574, 6575, 6576, 29000):
                for seconds in (-1, 0, 1, 43200):
                    later = birth + dt.timedelta(days=days, seconds=seconds)
                    self.assertEqual(_age_in_years(later, birth), relativedelta(later, birth).years, (birth, later))


if __name__ == "__main__":
    unittest.main()
