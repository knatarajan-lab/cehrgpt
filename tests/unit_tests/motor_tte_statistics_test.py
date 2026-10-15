import unittest

from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from datasets import Dataset

from cehrgpt.omop.ontology import Ontology
from cehrgpt.tokenization.tokenization_statistics import (
    compute_motor_tte_statistics,
    get_motor_code_ancestors,
    map_motor_tte_statistics,
)

DAY = 86400

# Two prediction points, the D1 tokens on days 1 and 3, in a patient whose events are
#   day 0: a    day 2: b    day 5: a
TOKENS = ["[VS]", "a", "[VE]", "D1", "[VS]", "b", "[VE]", "D1", "[VS]", "a", "[VE]"]
DAYS = [0, 0, 0, 1, 2, 2, 2, 3, 5, 5, 5]
PATIENT = {"concept_ids": TOKENS, "epoch_times": [float(d * DAY) for d in DAYS]}


def batch(*patients):
    return {
        "concept_ids": [p["concept_ids"] for p in patients],
        "epoch_times": [p["epoch_times"] for p in patients],
    }


def ontology(parents):
    # A real Ontology object without the Athena download its constructor needs
    result = Ontology.__new__(Ontology)
    result.parents_map = parents
    result.all_parents_map = {}
    result.concept_domain_map = {}
    result.concept_vocabulary_map = {}
    return result


def run(codes, parents=None, patients=(PATIENT,)):
    ancestors = get_motor_code_ancestors(codes, ontology(parents) if parents else None)
    return map_motor_tte_statistics(batch(*patients), codes, ancestors)


def counts(stats, code):
    return stats["task_tte_stats"][code], stats["task_censor_stats"][code]


class AncestorsTest(unittest.TestCase):
    def test_without_an_ontology_a_code_covers_itself(self):
        self.assertEqual(get_motor_code_ancestors(["a", "b"], None), {"a": ("a",), "b": ("b",)})

    def test_with_an_ontology_the_code_and_all_ancestors(self):
        result = get_motor_code_ancestors(["a"], ontology({"a": {"b"}, "b": {"c"}}))
        self.assertEqual(result, {"a": ("a", "b", "c")})

    def test_a_code_missing_from_the_ontology_still_covers_itself(self):
        self.assertEqual(get_motor_code_ancestors(["z"], ontology({"a": {"b"}})), {"z": ("z",)})


class CountingTest(unittest.TestCase):
    def test_codes_without_an_ontology(self):
        stats = run(["a", "b"])
        # a is ahead on both prediction points; b is only ahead of the first
        self.assertEqual(counts(stats, "a"), (2, 0))
        self.assertEqual(counts(stats, "b"), (1, 1))
        self.assertAlmostEqual(stats["task_tte_time_stats"]["a"].mean(), 3.0)  # 4 and 2 days
        self.assertAlmostEqual(stats["task_tte_time_stats"]["b"].mean(), 1.0)

    def test_an_ancestor_counts_once_per_prediction_point(self):
        stats = run(["a", "b"], {"a": {"P"}, "b": {"P"}})
        # Adding the children (2 + 1 events, 0 + 1 censored) would give 3 and 1
        self.assertEqual(counts(stats, "P"), (2, 0))
        self.assertEqual(sum(counts(stats, "P")), 2)  # the number of prediction points
        # first of a or b after each point: b at 1 day, then a at 2 days
        self.assertAlmostEqual(stats["task_tte_time_stats"]["P"].mean(), 1.5)
        self.assertEqual(stats["task_tte_time_stats"]["P"].count, 2)

    def test_the_children_are_unchanged_by_the_ontology(self):
        plain = run(["a", "b"])
        with_ontology = run(["a", "b"], {"a": {"P"}, "b": {"P"}})
        for code in ("a", "b"):
            self.assertEqual(counts(plain, code), counts(with_ontology, code))

    def test_every_code_accounts_for_every_prediction_point(self):
        stats = run(["a", "b"], {"a": {"b"}, "b": {"c"}}, patients=(PATIENT, PATIENT))
        for code in ("a", "b", "c"):
            self.assertEqual(sum(counts(stats, code)), 4)

    def test_a_chain_does_not_count_a_descendant_twice(self):
        # b is itself observed and is the parent of a, and c is the parent of b
        stats = run(["a", "b"], {"a": {"b"}, "b": {"c"}})
        self.assertEqual(counts(stats, "c"), (2, 0))
        self.assertEqual(counts(stats, "b"), (2, 0))  # b now covers a
        self.assertEqual(counts(stats, "a"), (2, 0))

    def test_the_result_does_not_depend_on_the_order(self):
        parents = {"a": {"b"}, "b": {"c"}}
        first = run(["a", "b"], parents)
        second = run(["b", "a"], dict(reversed(list(parents.items()))))
        for code in ("a", "b", "c"):
            self.assertEqual(counts(first, code), counts(second, code))

    def test_codes_that_never_occur_are_censored_with_their_ancestors(self):
        stats = run(["a", "z"], {"z": {"Q"}})
        self.assertEqual(counts(stats, "z"), (0, 2))
        self.assertEqual(counts(stats, "Q"), (0, 2))
        self.assertEqual(counts(stats, "a"), (2, 0))

    def test_a_patient_without_prediction_points_adds_nothing(self):
        single = {"concept_ids": ["[VS]", "a", "[VE]"], "epoch_times": [0.0, 0.0, 0.0]}
        stats = run(["a"], patients=(single,))
        self.assertEqual(counts(stats, "a"), (0, 0))

    def test_zero_day_time_tokens_are_not_prediction_points(self):
        tokens = ["[VS]", "a", "[VE]", "D0", "[VS]", "a", "[VE]"]
        patient = {"concept_ids": tokens, "epoch_times": [0.0] * 3 + [DAY] + [2.0 * DAY] * 3}
        self.assertEqual(sum(counts(run(["a"], patients=(patient,)), "a")), 0)

    def test_the_global_event_times_stay_at_the_level_of_the_codes_that_occur(self):
        # 2 events of a and 1 of b: the same three times with or without ancestors
        for parents in (None, {"a": {"P"}, "b": {"P"}}, {"a": {"b"}, "b": {"c"}}):
            stats = run(["a", "b"], parents)
            self.assertEqual(sorted(stats["motor_event_times"].samples), [1.0, 2.0, 4.0])

    def test_patients_add_up(self):
        parents = {"a": {"P"}, "b": {"P"}}
        one = run(["a", "b"], parents)
        two = run(["a", "b"], parents, patients=(PATIENT, PATIENT))
        for code in ("a", "b", "P"):
            self.assertEqual(
                counts(two, code), tuple(2 * v for v in counts(one, code))
            )


class ComputeStatisticsTest(unittest.TestCase):
    def test_the_dataset_statistics_use_the_same_counting(self):
        data = Dataset.from_list([PATIENT, PATIENT])
        args = DataTrainingArguments(
            data_folder="x", dataset_prepared_path="y",
            preprocessing_num_workers=1, preprocessing_batch_size=1,
        )
        stats = compute_motor_tte_statistics(
            data, args, ["a", "b"], ontology({"a": {"P"}, "b": {"P"}})
        )
        self.assertEqual(counts(stats, "P"), (4, 0))
        self.assertEqual(counts(stats, "a"), (4, 0))
        self.assertEqual(counts(stats, "b"), (2, 2))
        self.assertAlmostEqual(stats["task_tte_time_stats"]["P"].mean(), 1.5)


if __name__ == "__main__":
    unittest.main()
