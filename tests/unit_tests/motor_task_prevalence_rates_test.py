import copy
import os
import tempfile
import unittest

import pandas as pd
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset

from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer
from cehrgpt.tokenization.tokenization_statistics import (
    extract_time_interval_in_days,
    is_att_token,
)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data")
DAY = 86400
FLOOR = 1e-6


def data_args():
    return DataTrainingArguments(
        data_folder="x",
        dataset_prepared_path="y",
        preprocessing_num_workers=1,
        preprocessing_batch_size=100,
    )


def brute_force_rate(frame, code):
    """frac_events / mean time to event of one task, counted one prediction point at a time."""
    events = censored = 0
    total_days = 0.0
    for tokens, times in zip(frame["concept_ids"], frame["epoch_times"]):
        tokens = [str(t) for t in tokens]
        for position, token in enumerate(tokens):
            if not (is_att_token(token) and extract_time_interval_in_days(token) > 0):
                continue
            later = [
                times[p] for p in range(position + 1, len(tokens)) if tokens[p] == code
            ]
            if later:
                events += 1
                total_days += (later[0] - times[position]) / DAY
            else:
                censored += 1
    return (events / (events + censored)) / (total_days / events)


class PrevalenceRatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(SAMPLE, "pretrain", "patient_sequence.parquet")
        cls.frame = pd.read_parquet(path)
        cls.dataset = Dataset.from_parquet(path)
        concept = pd.read_parquet(os.path.join(SAMPLE, "omop_vocab", "concept"))
        cls.names = {str(r.concept_id): r.concept_name for r in concept.itertuples()}
        cls.embeddings = PretrainedEmbeddings(
            os.path.join(SAMPLE, "pretrained_embeddings")
        )
        cls.tokenizer = cls.train(cls.dataset)

    @classmethod
    def train(cls, dataset, **kwargs):
        return CehrGptTokenizer.train_tokenizer(
            dataset,
            cls.names,
            data_args(),
            cls.embeddings,
            num_motor_tasks=40,
            **kwargs,
        )

    def test_every_task_keeps_its_mean_time_to_event(self):
        info = self.tokenizer._motor_task_info
        tasks = self.tokenizer._motor_time_to_event_codes
        self.assertTrue(tasks)
        for code in tasks:
            stats = info["task_tte_time_stats"][code]
            # one time to event for each prediction point of an event of the task
            self.assertEqual(stats.count, info["task_tte_stats"][code], code)
            self.assertGreater(stats.mean(), 0, code)

    def test_a_rate_is_the_fraction_of_events_over_the_mean_time_to_event_of_its_task(
        self,
    ):
        rates = self.tokenizer.get_motor_task_prevalence_rates()
        tasks = sorted(self.tokenizer._motor_time_to_event_codes)
        self.assertEqual(len(rates), len(tasks))
        checked = 0
        for code, rate in zip(tasks, rates):
            if checked == 6:
                break
            expected = brute_force_rate(self.frame, code)
            self.assertAlmostEqual(rate, max(expected, FLOOR), delta=expected * 1e-6, msg=code)
            checked += 1

    def test_the_rates_differ_between_tasks(self):
        rates = self.tokenizer.get_motor_task_prevalence_rates()
        self.assertGreater(len(set(rates)), 1)
        self.assertFalse(all(rate == FLOOR for rate in rates))

    def test_a_tokenizer_without_the_time_statistics_has_no_rates(self):
        old = copy.deepcopy(self.tokenizer)
        del old._motor_task_info["task_tte_time_stats"]
        with self.assertLogs("transformers", level="WARNING") as logs:
            self.assertIsNone(old.get_motor_task_prevalence_rates())
        self.assertIn("no mean time to event", logs.output[0])

    def test_a_tokenizer_without_tasks_has_no_rates(self):
        empty = CehrGptTokenizer.train_tokenizer(
            self.dataset, self.names, data_args(), self.embeddings
        )
        self.assertIsNone(empty.get_motor_task_prevalence_rates())

    def test_the_statistics_survive_saving(self):
        with tempfile.TemporaryDirectory() as folder:
            self.tokenizer.save_pretrained(folder)
            loaded = CehrGptTokenizer.from_pretrained(folder)
            # the MOTOR information is read from the folder when it is first needed
            rates = loaded.get_motor_task_prevalence_rates()
        self.assertEqual(rates, self.tokenizer.get_motor_task_prevalence_rates())

    def test_a_task_added_by_expansion_brings_its_statistics(self):
        base = self.train(self.dataset.select(range(50)))
        expanded = CehrGptTokenizer.expand_trained_tokenizer(
            cehrgpt_tokenizer=base,
            dataset=self.dataset.select(range(50, 100)),
            concept_name_mapping=self.names,
            data_args=data_args(),
            pretrained_concept_embedding_model=self.embeddings,
            num_motor_tasks=40,
        )
        added = set(expanded._motor_time_to_event_codes) - set(
            base._motor_time_to_event_codes
        )
        self.assertTrue(added, "expansion added no task, so the test says nothing")
        info = expanded._motor_task_info
        for code in added:
            for key in ("task_tte_stats", "task_censor_stats", "task_tte_time_stats"):
                self.assertIn(code, info[key], (key, code))
        rates = expanded.get_motor_task_prevalence_rates()
        self.assertEqual(len(rates), len(expanded._motor_time_to_event_codes))
        # the tasks of the base are not changed
        for code in base._motor_time_to_event_codes:
            self.assertEqual(
                info["task_tte_time_stats"][code],
                base._motor_task_info["task_tte_time_stats"][code],
            )


if __name__ == "__main__":
    unittest.main()
