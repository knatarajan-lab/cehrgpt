import unittest

import numpy as np
import polars as pl
import torch

from cehrgpt.open_vocab import question_universe as qu
from cehrgpt.open_vocab.head import open_vocab_nll_loss
from cehrgpt.open_vocab.labels import (
    QuestionHierarchy,
    QuestionLabels,
    build_batch_question_labels,
    build_question_labels,
    sample_questions,
    to_concept_id,
)

from .question_universe_test import CATEGORIES, VISIT_TEST_CATEGORIES, tables

# the fixture hierarchy: Visit > Inpatient Hospital (500 > 510) and Outpatient Visit > Clinic/Center >
# Radiology Clinic/Center (520 > 530 > 540), Home (550) below no type; the conditions 100 > ... > 140
INPATIENT, HOSPITAL, OUTPATIENT, CLINIC, RADIOLOGY, HOME = 500, 510, 520, 530, 540, 550
ANGINA, CORONARY, HEART = 140, 130, 120


def hierarchy():
    concept, ancestor = tables()
    universes = {
        "Visit": qu.build_universe(concept, ancestor, [HOSPITAL, RADIOLOGY, HOME], "Visit", VISIT_TEST_CATEGORIES),
        "Condition": qu.build_universe(concept, ancestor, [ANGINA], "Condition", CATEGORIES),
    }
    return QuestionHierarchy.from_universes(universes, ancestor)


def naive(h, concepts, days, points, end, questions):
    """Per point and question, the first event strictly after the point, the slow way."""
    rows, slots, times = [], [], []
    for r, t in enumerate(points):
        for s, q in enumerate(questions):
            later = [d for c, d in zip(concepts, days) if q in h.questions_of(c) and d > t]
            if later:
                rows.append(r), slots.append(s), times.append(min(later) - t)
    return rows, slots, times, [max(end - t, 0) for t in points]


class QuestionHierarchyTest(unittest.TestCase):
    def test_a_visit_record_counts_for_its_visit_type(self):
        h = hierarchy()
        self.assertEqual(h.questions_of(HOSPITAL), (INPATIENT, HOSPITAL))
        self.assertEqual(h.questions_of(RADIOLOGY), (OUTPATIENT, CLINIC, RADIOLOGY))

    def test_a_visit_below_no_type_counts_for_itself_only(self):
        self.assertEqual(hierarchy().questions_of(HOME), (HOME,))

    def test_conditions_roll_up_their_chain(self):
        self.assertEqual(hierarchy().questions_of(ANGINA), (HEART, CORONARY, ANGINA))

    def test_concept_ids_can_be_ints_or_digit_strings(self):
        h = hierarchy()
        self.assertEqual(h.questions_of("510"), h.questions_of(510))
        self.assertEqual(h.questions_of(np.int64(510)), h.questions_of(510))
        self.assertIn("510", h)

    def test_tokens_that_are_not_concept_ids_count_for_nothing(self):
        h = hierarchy()
        for token in ("[VS]", "[VE]", "D3", "year:2020", "age:40", "Visit/IP", "CMS_Place_of_Service/51", "", "5.1", "-5", None, 5.5, True):
            self.assertEqual(h.questions_of(token), (), repr(token))
            self.assertNotIn(token, h)

    def test_to_concept_id(self):
        self.assertEqual([to_concept_id(x) for x in (9202, "9202", np.int32(7), "x", "9202a", None, 1.5, False)],
                         [9202, 9202, 7, None, None, None, None, None])

    def test_unknown_concepts_count_for_nothing(self):
        h = hierarchy()
        self.assertEqual(h.questions_of(999), ())
        self.assertNotIn(999, h)

    def test_a_concept_only_counts_for_the_questions_of_its_own_domain(self):
        # 510 is a visit; it must not pick up condition questions through the shared ancestor table
        self.assertTrue(set(hierarchy().questions_of(HOSPITAL)) <= {INPATIENT, HOSPITAL})


class BuildQuestionLabelsTest(unittest.TestCase):
    def setUp(self):
        self.h = hierarchy()
        # the record: an outpatient visit at day 0, an inpatient hospital stay at day 10, a radiology visit at 40
        self.concepts = [RADIOLOGY, HOSPITAL, RADIOLOGY, 12345, ANGINA]
        self.days = [0.0, 10.0, 40.0, 41.0, 25.0]
        self.questions = [INPATIENT, HOSPITAL, OUTPATIENT, CLINIC, RADIOLOGY, HOME, HEART]

    def labels(self, points, end=60.0, questions=None):
        return build_question_labels(self.h, self.concepts, self.days, points, end, questions or self.questions)

    def table(self, labels):
        return {(int(r), self.questions[int(q)]): float(t) for r, q, t in zip(labels.event_rows, labels.event_questions, labels.event_times)}

    def test_a_visit_record_is_also_every_visit_question_above_it(self):
        got = self.table(self.labels([5.0]))
        # the stay at day 10: Inpatient Hospital and Inpatient Visit
        self.assertEqual(got[(0, HOSPITAL)], 5.0)
        self.assertEqual(got[(0, INPATIENT)], 5.0)
        # the radiology visit at day 40 is a Radiology Clinic, a Clinic/Center and an Outpatient Visit
        self.assertEqual(got[(0, RADIOLOGY)], 35.0)
        self.assertEqual(got[(0, CLINIC)], 35.0)
        self.assertEqual(got[(0, OUTPATIENT)], 35.0)

    def test_only_the_first_event_after_the_point_counts(self):
        got = self.table(self.labels([-5.0]))
        self.assertEqual(got[(0, OUTPATIENT)], 5.0)  # day 0, not 40
        self.assertEqual(got[(0, RADIOLOGY)], 5.0)

    def test_events_on_or_before_the_point_do_not_count(self):
        got = self.table(self.labels([10.0, 40.0]))
        self.assertNotIn((0, HOSPITAL), got)  # the stay is at day 10 = the point
        self.assertNotIn((0, INPATIENT), got)
        self.assertEqual(got[(0, RADIOLOGY)], 30.0)
        self.assertNotIn((1, RADIOLOGY), got)  # the last radiology visit is at the point

    def test_questions_that_never_happen_are_censored_not_listed(self):
        labels = self.labels([5.0, 50.0], end=60.0)
        got = self.table(labels)
        self.assertNotIn((0, HOME), got)
        self.assertEqual(labels.censor_times.tolist(), [55.0, 10.0])
        self.assertFalse(any(r == 1 for r, _ in got))  # nothing but the end of the record after day 41

    def test_other_domains_roll_up_in_the_same_call(self):
        got = self.table(self.labels([0.0]))
        self.assertEqual(got[(0, HEART)], 25.0)

    def test_a_sequence_of_string_tokens_gives_the_same_labels_as_concept_ids(self):
        tokens = ["year:2020", "age:40", "Gender/F", "Race/5", "[VS]", str(RADIOLOGY), "D5", str(HOSPITAL),
                  "Visit/IP", str(RADIOLOGY), "[VE]", "12345", str(ANGINA)]
        days = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 40.0, 40.0, 41.0, 25.0]
        mixed = build_question_labels(self.h, tokens, days, [5.0, 20.0], 60.0, self.questions)
        clean = build_question_labels(self.h, self.concepts, self.days, [5.0, 20.0], 60.0, self.questions)
        for name in ("event_rows", "event_questions", "event_times", "censor_times"):
            np.testing.assert_array_equal(getattr(mixed, name), getattr(clean, name), name)

    def test_ignored_events_and_unasked_questions(self):
        # only the asked question is listed, in slot 0, and 12345 (not in the hierarchy) adds nothing
        labels = self.labels([5.0], questions=[HOSPITAL])
        self.assertEqual(labels.event_rows.tolist(), [0])
        self.assertEqual(labels.event_questions.tolist(), [0])
        self.assertEqual(labels.event_times.tolist(), [5.0])

    def test_matches_the_naive_computation(self):
        rng = np.random.default_rng(0)
        pool = [RADIOLOGY, HOSPITAL, HOME, ANGINA, 12345, CLINIC]
        for _ in range(20):
            concepts = rng.choice(pool, 12).tolist()
            days = np.round(rng.uniform(0, 100, 12), 1).tolist()
            points = np.round(rng.uniform(-5, 110, 6), 1).tolist()
            got = build_question_labels(self.h, concepts, days, points, 120.0, self.questions)
            rows, slots, times, censor = naive(self.h, concepts, days, points, 120.0, self.questions)
            self.assertEqual(sorted(zip(got.event_rows, got.event_questions, np.round(got.event_times, 6))),
                             sorted(zip(rows, slots, np.round(times, 6))))
            np.testing.assert_allclose(got.censor_times, censor)

    def test_matches_the_naive_computation_with_repeated_days_and_unsorted_points(self):
        rng = np.random.default_rng(1)
        pool = [RADIOLOGY, HOSPITAL, HOME, ANGINA, 12345, CLINIC, CORONARY, "[VS]", "D5", "Visit/IP", str(ANGINA)]
        compared = 0
        for _ in range(300):
            n = int(rng.integers(0, 25))
            concepts = [pool[i] for i in rng.integers(0, len(pool), n)]
            days = rng.integers(0, 12, n).astype(float).tolist()  # many events share a day
            points = rng.integers(-2, 14, int(rng.integers(0, 8))).astype(float).tolist()  # unsorted, repeated
            questions = [self.questions[i] for i in rng.permutation(len(self.questions))[: int(rng.integers(1, 8))]]
            got = build_question_labels(self.h, concepts, days, points, 15.0, questions)
            rows, slots, times, censor = naive(self.h, concepts, days, points, 15.0, questions)
            self.assertEqual(sorted(zip(got.event_rows.tolist(), got.event_questions.tolist(), got.event_times.tolist())),
                             sorted(zip(rows, slots, times)))
            np.testing.assert_allclose(got.censor_times, censor)
            compared += len(rows)
        self.assertGreater(compared, 500)

    def test_the_labels_are_ordered_by_row_then_slot(self):
        labels = self.labels([40.0, 5.0, 20.0])
        keys = list(zip(labels.event_rows.tolist(), labels.event_questions.tolist()))
        self.assertEqual(keys, sorted(keys))
        # the rows are the positions of the prediction points as they were given, not sorted
        self.assertEqual(self.table(labels)[(1, HOSPITAL)], 5.0)  # the point at day 5 is row 1

    def test_empty_inputs(self):
        for kwargs in (
            dict(concepts=[], days=[], points=[5.0]),
            dict(concepts=[HOSPITAL], days=[10.0], points=[]),
            dict(concepts=["[VS]", "D3"], days=[1.0, 2.0], points=[0.0]),
            dict(concepts=[HOSPITAL], days=[10.0], points=[11.0]),  # nothing after the point
        ):
            labels = build_question_labels(self.h, kwargs["concepts"], kwargs["days"], kwargs["points"], 20.0, self.questions)
            self.assertEqual(len(labels.event_rows), 0, kwargs)
            self.assertEqual(labels.num_rows, len(kwargs["points"]))
        asked_nothing = build_question_labels(self.h, self.concepts, self.days, [5.0], 60.0, [])
        self.assertEqual(len(asked_nothing.event_rows), 0)
        not_in_hierarchy = build_question_labels(self.h, self.concepts, self.days, [5.0], 60.0, [999999])
        self.assertEqual(len(not_in_hierarchy.event_rows), 0)

    def test_a_question_asked_twice_is_an_error(self):
        with self.assertRaises(ValueError):
            build_question_labels(self.h, self.concepts, self.days, [5.0], 60.0, [HOSPITAL, HOSPITAL])

    def test_bools_and_floats_are_not_concept_ids(self):
        h = QuestionHierarchy({1: [1], 5: [5]})
        labels = build_question_labels(h, [True, 5.0, 1, "5"], [10.0, 11.0, 12.0, 13.0], [0.0], 20.0, [1, 5])
        # only the int 1 (day 12) and the string "5" (day 13) count
        self.assertEqual(sorted(zip(labels.event_questions.tolist(), labels.event_times.tolist())), [(0, 12.0), (1, 13.0)])

    def test_concatenate_shifts_the_rows(self):
        a, b = self.labels([5.0, 20.0]), self.labels([30.0])
        both = QuestionLabels.concatenate([a, b])
        self.assertEqual(both.num_rows, 3)
        self.assertEqual(both.event_rows.max(), 2)
        self.assertEqual(len(both.event_rows), len(a.event_rows) + len(b.event_rows))

    def test_the_labels_feed_the_loss(self):
        labels = QuestionLabels.concatenate([self.labels([5.0, 20.0]), self.labels([30.0])])
        bins = [0.0, 30.0, 365.0]
        hazards = torch.full((labels.num_rows, 2, len(self.questions)), -6.0)
        loss = open_vocab_nll_loss(
            hazards, torch.as_tensor(labels.event_rows), torch.as_tensor(labels.event_questions),
            torch.as_tensor(labels.event_times), torch.as_tensor(labels.censor_times), bins,
        )
        self.assertTrue(torch.isfinite(loss))


class HierarchyArraysTest(unittest.TestCase):
    def test_vocab_numbers_map_ids_of_the_vocabulary(self):
        h = hierarchy()
        vocab = {"[PAD]": 0, "[VS]": 1, str(HOSPITAL): 2, "Visit/IP": 3, str(ANGINA): 4, "99999": 5}
        numbers = h.vocab_numbers(vocab)
        self.assertEqual(len(numbers), 6)
        self.assertEqual((numbers >= 0).tolist(), [False, False, True, False, True, False])
        owner, questions = h.expand(numbers[[2, 4]])
        self.assertEqual(sorted(questions[owner == 0].tolist()), [INPATIENT, HOSPITAL])
        self.assertEqual(sorted(questions[owner == 1].tolist()), sorted([HEART, CORONARY, ANGINA]))

    def test_save_and_load_round_trip(self):
        import os
        import tempfile

        h = hierarchy()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hierarchy.parquet")
            h.save(path)
            loaded = QuestionHierarchy.load(path)
        self.assertEqual(len(loaded), len(h))
        for concept in (HOSPITAL, RADIOLOGY, HOME, ANGINA):
            self.assertEqual(loaded.questions_of(concept), h.questions_of(concept))


class BatchLabelsTest(unittest.TestCase):
    def setUp(self):
        self.h = hierarchy()
        # the embedding rows, in some order: row i is the question concept id QUESTIONS[i]
        self.questions = np.array([HOME, INPATIENT, ANGINA, OUTPATIENT, HOSPITAL, HEART, CLINIC, RADIOLOGY, CORONARY, 77777])
        self.numbers = lambda concepts: self.h.concept_numbers(concepts)

    def segment(self, concepts, days, positions):
        return self.numbers(concepts), np.asarray(days, dtype=float), np.asarray(positions)

    def test_sample_questions_mixes_positives_and_the_rest(self):
        rng = np.random.default_rng(0)
        rows = sample_questions(np.array([3, 4, 5]), 100, 10, 0.5, rng)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(set(rows.tolist())), 10)
        self.assertEqual(len(set(rows.tolist()) & {3, 4, 5}), 3 if set([3, 4, 5]) <= set(rows.tolist()) else len(set(rows.tolist()) & {3, 4, 5}))
        # with more positives than the share, only the share is positive-drawn
        rows = sample_questions(np.arange(50), 100, 10, 0.4, rng)
        self.assertEqual(len(rows), 10)

    def test_sample_questions_cannot_ask_more_than_there_are(self):
        rows = sample_questions(np.array([0, 1]), 4, 10, 0.5, np.random.default_rng(1))
        self.assertEqual(sorted(rows.tolist()), [0, 1, 2, 3])
        self.assertEqual(len(sample_questions(np.zeros(0, dtype=np.int64), 5, 3, 0.5, np.random.default_rng(1))), 3)

    def test_one_question_set_for_the_batch_and_rows_counted_over_it(self):
        # patient A: a stay at day 10, prediction points at positions 0 and 1 (days 0 and 5)
        # patient B: a radiology visit at day 30, one prediction point at position 0 (day 20)
        a = self.segment([ANGINA, HOSPITAL, 12345], [0.0, 10.0, 11.0], [0, 1])
        a = (a[0], np.array([0.0, 5.0, 10.0]) if False else a[1], a[2])
        a = self.segment([ANGINA, ANGINA, HOSPITAL], [0.0, 5.0, 10.0], [0, 1])
        b = self.segment([ANGINA, RADIOLOGY], [20.0, 30.0], [0])
        labels = build_batch_question_labels(self.h, self.questions, [a, b], num_questions=10, positive_fraction=0.5, rng=np.random.default_rng(0))
        asked = self.questions[labels.question_rows]
        table = {(int(r), int(asked[q])): float(t) for r, q, t in zip(labels.event_rows, labels.event_questions, labels.event_times)}
        # patient A rows 0 and 1: the stay at day 10 is 10 and 5 days away
        self.assertEqual(table[(0, HOSPITAL)], 10.0)
        self.assertEqual(table[(1, HOSPITAL)], 5.0)
        self.assertEqual(table[(0, INPATIENT)], 10.0)
        # patient B is row 2 of the batch; the radiology visit is 10 days away, the stay of A does not count
        self.assertEqual(table[(2, RADIOLOGY)], 10.0)
        self.assertEqual(table[(2, OUTPATIENT)], 10.0)
        self.assertNotIn((2, HOSPITAL), table)
        # and A's rows know nothing of B's visit
        self.assertNotIn((0, RADIOLOGY), table)
        self.assertNotIn((1, RADIOLOGY), table)

    def test_the_questions_that_happen_are_asked_first(self):
        a = self.segment([ANGINA, HOSPITAL], [0.0, 10.0], [0])
        labels = build_batch_question_labels(self.h, self.questions, [a], num_questions=3, positive_fraction=1.0, rng=np.random.default_rng(0))
        asked = set(self.questions[labels.question_rows].tolist())
        # the positives of this patient after the point (the stay and the visit type above it) are always
        # asked; with only two of them the third question is drawn from all the questions
        self.assertTrue({INPATIENT, HOSPITAL} <= asked)
        self.assertEqual(len(asked), 3)
        self.assertEqual(len(labels.event_rows), 2)

    def test_a_batch_without_prediction_points_has_no_events(self):
        a = self.segment([ANGINA, HOSPITAL], [0.0, 10.0], [])
        labels = build_batch_question_labels(self.h, self.questions, [a], num_questions=4, rng=np.random.default_rng(0))
        self.assertEqual(len(labels.event_rows), 0)
        self.assertEqual(len(labels.question_rows), 4)

    def test_the_events_are_ordered_and_match_the_single_patient_builder(self):
        rng = np.random.default_rng(3)
        pool = [HOSPITAL, RADIOLOGY, ANGINA, HOME, 12345]
        segments, singles = [], []
        offset = 0
        for _ in range(4):
            n = int(rng.integers(3, 12))
            concepts = [pool[i] for i in rng.integers(0, len(pool), n)]
            days = np.sort(rng.integers(0, 30, n)).astype(float)
            positions = np.sort(rng.choice(n, int(rng.integers(1, min(n, 4) + 1)), replace=False))
            segments.append(self.segment(concepts, days, positions))
            singles.append((concepts, days, positions))
        labels = build_batch_question_labels(self.h, self.questions, segments, num_questions=10, positive_fraction=0.5, rng=np.random.default_rng(5))
        asked = self.questions[labels.question_rows].tolist()
        got = sorted(zip(labels.event_rows.tolist(), labels.event_questions.tolist(), labels.event_times.tolist()))
        want, offset = [], 0
        for concepts, days, positions in singles:
            single = build_question_labels(self.h, concepts, days, days[positions], days[-1], asked)
            want += list(zip((single.event_rows + offset).tolist(), single.event_questions.tolist(), single.event_times.tolist()))
            offset += len(positions)
        self.assertEqual(got, sorted(want))
        keys = list(zip(labels.event_rows.tolist(), labels.event_questions.tolist()))
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
