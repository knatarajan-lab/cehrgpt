import unittest

import numpy as np
import polars as pl
import torch

from cehrgpt.open_vocab import question_universe as qu
from cehrgpt.open_vocab.head import open_vocab_nll_loss
from cehrgpt.open_vocab.labels import QuestionHierarchy, QuestionLabels, build_question_labels

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


if __name__ == "__main__":
    unittest.main()
