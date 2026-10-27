import tempfile
import unittest
from pathlib import Path

import polars as pl

from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import process

DAY = 86_400
START = 1_600_000_000.0
SOURCE, COMPARATOR, EXCLUDED = 100, 200, 500


def patient(person_id, visits):
    """visits: list of (day, [clinical tokens]); the demographic tokens carry no time."""
    concept_ids = ["year:2020", "age:50", "8507", "0"]
    epoch_times = [0.0] * 4
    for day, tokens in visits:
        for token in ["[VS]", "9202", *tokens, "[VE]"]:
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


class ExtractionProcessTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "vocab" / "concept_ancestor").mkdir(parents=True)
        ids = [SOURCE, COMPARATOR, EXCLUDED]
        pl.DataFrame({"ancestor_concept_id": ids, "descendant_concept_id": ids}).write_parquet(
            self.root / "vocab" / "concept_ancestor" / "part.parquet"
        )
        self.output = self.root / "out"

    def write_patients(self, patients):
        pl.DataFrame(patients).write_parquet(self.root / "patients.parquet")

    def run_process(self, **kwargs):
        process(
            patient_sequence_path=str(self.root / "patients.parquet"),
            vocab_path=str(self.root / "vocab"),
            source_ingredient_ids=[SOURCE],
            comparator_ingredient_ids=[COMPARATOR],
            output_dir=self.output,
            min_context_length=1,
            **kwargs,
        )

    def drug_info(self):
        return pl.read_parquet(str(self.output / "drug_info" / "*.parquet")).sort("person_id")

    def test_the_washout_keeps_only_patients_with_enough_history_before_the_index_drug(self):
        self.write_patients(
            [
                patient(1, [(-400, ["x"]), (0, [str(SOURCE)])]),
                patient(2, [(-100, ["x"]), (0, [str(SOURCE)])]),
                patient(3, [(0, [str(SOURCE)])]),
            ]
        )
        self.run_process(min_prior_observation_days=365)
        self.assertEqual(self.drug_info()["person_id"].to_list(), [1])

    def test_without_a_washout_every_new_initiator_is_kept(self):
        self.write_patients(
            [patient(1, [(-400, ["x"]), (0, [str(SOURCE)])]), patient(3, [(0, [str(SOURCE)])])]
        )
        self.run_process()
        self.assertEqual(self.drug_info()["person_id"].to_list(), [1, 3])

    def test_an_exclusion_drug_in_the_index_visit_excludes_the_patient(self):
        self.write_patients(
            [
                patient(1, [(0, [str(SOURCE), str(EXCLUDED)])]),
                patient(2, [(0, [str(SOURCE)]), (10, [str(EXCLUDED)])]),
                patient(3, [(-5, [str(EXCLUDED)]), (0, [str(SOURCE)])]),
            ]
        )
        self.run_process(exclusion_concept_ids=[EXCLUDED])
        # patient 2 starts the excluded drug only after the index visit, which censors them
        self.assertEqual(self.drug_info()["person_id"].to_list(), [2])

    def test_an_exclusion_drug_after_the_index_visit_censors_follow_up(self):
        self.write_patients(
            [patient(1, [(0, [str(SOURCE)]), (10, [str(EXCLUDED)]), (20, [str(SOURCE)])])]
        )
        self.run_process(exclusion_concept_ids=[EXCLUDED])
        self.assertEqual(self.drug_info()["era_end_epoch_time"][0], START + 10 * DAY)

    def test_without_an_exclusion_list_the_drug_does_not_censor(self):
        self.write_patients(
            [patient(1, [(0, [str(SOURCE)]), (10, [str(EXCLUDED)]), (20, [str(SOURCE)])])]
        )
        self.run_process()
        self.assertIsNone(self.drug_info()["era_end_epoch_time"][0])

    def test_the_last_observation_is_written_to_drug_info(self):
        self.write_patients([patient(1, [(0, [str(SOURCE)]), (50, ["x"])])])
        self.run_process()
        self.assertEqual(self.drug_info()["last_observed_epoch_time"][0], START + 50 * DAY)

    def test_existing_outputs_with_the_same_parameters_are_reused(self):
        self.write_patients([patient(1, [(0, [str(SOURCE)])])])
        self.run_process()
        marker = self.output / "drug_info" / "marker.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(marker)
        self.run_process()
        self.assertTrue(marker.exists())

    def test_existing_outputs_made_with_other_parameters_raise(self):
        self.write_patients([patient(1, [(0, [str(SOURCE)])])])
        self.run_process()
        with self.assertRaisesRegex(RuntimeError, "min_prior_observation_days"):
            self.run_process(min_prior_observation_days=365)

    def test_existing_outputs_without_a_parameter_record_raise(self):
        self.write_patients([patient(1, [(0, [str(SOURCE)])])])
        self.run_process()
        (self.output / "extraction_params.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "parameter record"):
            self.run_process()

    def test_overwrite_removes_the_chunks_of_the_earlier_run(self):
        self.write_patients([patient(1, [(0, [str(SOURCE)])])])
        self.run_process()
        for folder in ("drug_info", "treated_context", "non_treated_context"):
            pl.DataFrame({"person_id": [999]}).write_parquet(
                self.output / folder / "shard_007_chunk_00000.parquet"
            )
        self.run_process(overwrite=True, min_prior_observation_days=0)
        self.assertEqual(self.drug_info()["person_id"].to_list(), [1])
        self.assertFalse((self.output / "drug_info" / "shard_007_chunk_00000.parquet").exists())


if __name__ == "__main__":
    unittest.main()
