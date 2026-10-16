import random
import re
import unittest

from cehrgpt.qa.question_templates import (
    SECONDS_PER_DAY,
    TIME_TO_EVENT_PHRASINGS,
    WILL_OCCUR_PHRASINGS,
    ClinicalEvents,
    ConceptCoverage,
    PatientTimeline,
    QAConfig,
    generate_qa_pairs,
    time_to_event,
    will_occur,
)

NAMES = {"100": "diabetes", "200": "asthma", "300": "fracture", "400": "migraine"}
INDEX = 1_600_000_000.0


def day(n):
    return INDEX + n * SECONDS_PER_DAY


def timeline(events, age=60):
    return PatientTimeline(
        concept_ids=[c for c, _ in events],
        epoch_times=[t for _, t in events],
        index_time=INDEX,
        age_at_index=age,
    )


def horizon_of(question):
    return int(re.search(r"(\d+) days", question).group(1))


class QuestionTemplatesTest(unittest.TestCase):
    def setUp(self):
        self.timeline = timeline(
            [
                ("[VS]", day(-30)),
                ("100", day(-30)),
                ("100", day(-5)),
                ("D3", day(-2)),
                ("200", day(40)),
                ("[VS]", day(400)),
                ("300", day(400)),
            ]
        )
        self.events = ClinicalEvents(self.timeline, NAMES)

    def test_events_split_at_the_index_date(self):
        self.assertEqual(self.events.prior_concepts, {"100"})
        self.assertEqual(set(self.events.first_future_time), {"200", "300"})
        self.assertAlmostEqual(self.events.days_to_first("200"), 40)
        self.assertIsNone(self.events.days_to_first("100"))
        self.assertAlmostEqual(self.events.follow_up_days, 400)

    def test_non_concept_tokens_are_ignored(self):
        self.assertNotIn("[VS]", self.events.prior_concepts)
        self.assertNotIn("D3", self.events.prior_concepts)

    def test_will_occur_answer_depends_on_the_horizon(self):
        config = QAConfig(will_occur_positive_fraction=1.0, horizons_in_days=(10, 90))
        rng = random.Random(0)
        seen = set()
        for _ in range(80):
            pair = will_occur(self.events, NAMES, ["400"], rng, config)
            if pair is None or "asthma" not in pair.question:
                continue
            horizon = horizon_of(pair.question)
            seen.add(horizon)
            if horizon >= 40:
                self.assertEqual(pair.answer, "Yes.")
            else:
                self.assertEqual(pair.answer, "No.")
        self.assertEqual(seen, {10, 90})

    def test_binary_answers_are_exactly_yes_or_no(self):
        # The timing belongs to the time-to-event question, so the score can be read from "Yes."
        answers = set()
        for seed in range(60):
            for pair in generate_qa_pairs(
                self.timeline, NAMES, ["400"], random.Random(seed), 6,
                QAConfig(time_to_event_weight=0.0),
            ):
                self.assertEqual(pair.qa_type, "will_occur")
                answers.add(pair.answer)
        self.assertEqual(answers, {"Yes.", "No."})

    def test_negatives_are_only_asked_with_enough_follow_up(self):
        # Follow-up ends after 10 days, so "no migraine within 365 days" is unknowable
        short = timeline([("100", day(-5)), ("200", day(10))])
        events = ClinicalEvents(short, NAMES)
        config = QAConfig(will_occur_positive_fraction=0.0)
        rng = random.Random(0)
        for _ in range(100):
            pair = will_occur(events, NAMES, ["400"], rng, config)
            if pair is not None:
                self.assertEqual(pair.answer, "No.")
                self.assertLessEqual(horizon_of(pair.question), 10)

    def test_time_to_event_positive_and_censored(self):
        rng = random.Random(1)
        answers = {
            time_to_event(self.events, NAMES, ["400"], rng, QAConfig()).answer
            for _ in range(80)
        }
        self.assertIn("About 40 days.", answers)
        self.assertIn("About 400 days.", answers)
        self.assertIn("It was not observed in the 400 days of follow-up.", answers)

    def test_no_follow_up_gives_no_time_to_event_negative(self):
        no_future = timeline([("100", day(-5))])
        events = ClinicalEvents(no_future, NAMES)
        config = QAConfig(time_to_event_positive_fraction=0.0)
        self.assertIsNone(time_to_event(events, NAMES, ["400"], random.Random(0), config))

    def test_positive_fraction_is_respected(self):
        for fraction in (0.0, 1.0):
            config = QAConfig(time_to_event_positive_fraction=fraction)
            rng = random.Random(3)
            answers = [
                time_to_event(self.events, NAMES, ["400"], rng, config).answer
                for _ in range(40)
            ]
            observed = [a.startswith("About") for a in answers]
            self.assertEqual(all(observed), fraction == 1.0)
            self.assertEqual(any(observed), fraction == 1.0)

    def test_paraphrases_are_used_and_can_be_turned_off(self):
        rng = random.Random(0)
        questions = {
            time_to_event(self.events, NAMES, ["400"], rng, QAConfig()).question
            for _ in range(80)
        }
        self.assertGreater(len(questions), 1)
        rng = random.Random(0)
        questions = {
            time_to_event(
                self.events, NAMES, ["400"], rng, QAConfig(paraphrase=False)
            ).question.split("next ")[0]
            for _ in range(80)
        }
        self.assertEqual(questions, {"How many days until the patient's "})
        self.assertGreater(len(WILL_OCCUR_PHRASINGS), 1)
        self.assertGreater(len(TIME_TO_EVENT_PHRASINGS), 1)

    def test_generate_is_reproducible_distinct_and_only_two_types(self):
        first = generate_qa_pairs(self.timeline, NAMES, ["400"], random.Random(7), 6)
        second = generate_qa_pairs(self.timeline, NAMES, ["400"], random.Random(7), 6)
        self.assertEqual(first, second)
        questions = [pair.question for pair in first]
        self.assertEqual(len(questions), len(set(questions)))
        self.assertLessEqual(len(first), 6)
        self.assertTrue({p.qa_type for p in first} <= {"will_occur", "time_to_event"})

    def test_weights_select_the_question_type(self):
        for weights, expected in (
            ({"will_occur_weight": 1.0, "time_to_event_weight": 0.0}, "will_occur"),
            ({"will_occur_weight": 0.0, "time_to_event_weight": 1.0}, "time_to_event"),
        ):
            pairs = generate_qa_pairs(
                self.timeline, NAMES, ["400"], random.Random(5), 6, QAConfig(**weights)
            )
            self.assertTrue(pairs)
            self.assertEqual({p.qa_type for p in pairs}, {expected})

    def test_questions_never_ask_about_the_past(self):
        banned = ("before", "ever", "history", "previous", "again", "last ")
        pairs = []
        for seed in range(30):
            pairs += generate_qa_pairs(
                self.timeline, NAMES, ["100", "400"], random.Random(seed), 6
            )
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertIn(pair.qa_type, ("will_occur", "time_to_event"))
            self.assertFalse(any(w in pair.question.lower() for w in banned), pair.question)

    def test_new_onset_only_skips_concepts_from_the_history(self):
        # "100" is in the history and, as a negative candidate, would otherwise be asked about
        config = QAConfig(will_occur_positive_fraction=0.0, time_to_event_positive_fraction=0.0)
        new_only = QAConfig(
            will_occur_positive_fraction=0.0,
            time_to_event_positive_fraction=0.0,
            new_onset_only=True,
        )
        asked = lambda cfg: {  # noqa: E731
            pair.question
            for seed in range(60)
            for pair in generate_qa_pairs(
                self.timeline, NAMES, ["100"], random.Random(seed), 4, cfg
            )
        }
        self.assertTrue(any("diabetes" in q for q in asked(config)))
        self.assertFalse(any("diabetes" in q for q in asked(new_only)))

    def test_pairs_carry_the_concept_they_ask_about(self):
        for seed in range(20):
            for pair in generate_qa_pairs(self.timeline, NAMES, ["400"], random.Random(seed), 4):
                self.assertIn(pair.concept_id, NAMES)
                self.assertIn(NAMES[pair.concept_id], pair.question)

    def test_out_of_scope_fraction_steers_away_from_motor_tasks(self):
        motor_codes = {"200"}  # asthma is a MOTOR task; fracture, migraine are not
        for fraction, expect_motor in ((0.0, True), (1.0, False)):
            config = QAConfig(
                will_occur_positive_fraction=1.0,
                time_to_event_positive_fraction=1.0,
                out_of_scope_fraction=fraction,
            )
            concepts = {
                pair.concept_id
                for seed in range(40)
                for pair in generate_qa_pairs(
                    self.timeline, NAMES, ["400"], random.Random(seed), 4, config, motor_codes
                )
            }
            self.assertEqual(concepts & motor_codes != set(), expect_motor)
            if not expect_motor:
                self.assertEqual(concepts, {"300"})

    def test_out_of_scope_falls_back_when_there_is_nothing_else(self):
        # Only asthma happens, and it is a MOTOR task: ask about it rather than nothing
        only_motor = timeline([("100", day(-5)), ("200", day(10))])
        config = QAConfig(
            will_occur_positive_fraction=1.0,
            time_to_event_positive_fraction=1.0,
            out_of_scope_fraction=1.0,
        )
        pairs = generate_qa_pairs(
            only_motor, NAMES, [], random.Random(0), 3, config, {"200"}
        )
        self.assertTrue(pairs)
        self.assertEqual({p.concept_id for p in pairs}, {"200"})

    def test_out_of_scope_without_motor_codes_changes_nothing(self):
        config = QAConfig(out_of_scope_fraction=1.0)
        plain = generate_qa_pairs(self.timeline, NAMES, ["400"], random.Random(2), 4)
        same = generate_qa_pairs(self.timeline, NAMES, ["400"], random.Random(2), 4, config)
        # The rng is consumed only when there are codes to steer by
        self.assertEqual(plain, same)

    def test_artificial_tokens_never_become_questions(self):
        tokens = timeline(
            [
                ("[VS]", day(1)),
                ("D5", day(2)),
                ("year:2020", day(3)),
                ("Gender/F", day(4)),
                ("200", day(5)),
            ]
        )
        names = {**NAMES, "[VS]": "[VS]", "D5": "5 days", "year:2020": "2020", "Gender/F": "f"}
        config = QAConfig(will_occur_positive_fraction=1.0, time_to_event_positive_fraction=1.0)
        for seed in range(30):
            for pair in generate_qa_pairs(tokens, names, ["400"], random.Random(seed), 4, config):
                self.assertIn(pair.concept_id, NAMES)

    def test_event_occurred_flag(self):
        for seed in range(40):
            for pair in generate_qa_pairs(self.timeline, NAMES, ["400"], random.Random(seed), 4):
                no_event = pair.answer == "No." or pair.answer.startswith("It was not observed")
                self.assertEqual(pair.event_occurred, not no_event, pair)

    def test_coverage_prefers_the_least_asked_concepts(self):
        coverage = ConceptCoverage(num_candidates=4)
        for _ in range(10):
            coverage.record("a")
        rng = random.Random(0)
        self.assertEqual(coverage.choose(["a", "b"], rng), "b")
        self.assertEqual(coverage.choose(["a"], rng), "a")
        with self.assertRaises(ValueError):
            ConceptCoverage(0)

    def test_coverage_reaches_the_whole_vocabulary(self):
        vocabulary = [str(1000 + i) for i in range(60)]
        names = {c: f"concept {c}" for c in vocabulary}
        # A patient whose only event is concept 1000: without coverage the negatives are random
        patient = timeline([("1000", day(5)), ("1000", day(400))])
        for use_coverage in (True, False):
            coverage = ConceptCoverage(8) if use_coverage else None
            asked = {}
            rng = random.Random(0)
            config = QAConfig(will_occur_positive_fraction=0.0, time_to_event_positive_fraction=0.0)
            for _ in range(80):
                for pair in generate_qa_pairs(
                    patient, names, vocabulary, rng, 3, config, coverage=coverage
                ):
                    asked[pair.concept_id] = asked.get(pair.concept_id, 0) + 1
            covered = len(asked)
            if use_coverage:
                self.assertEqual(covered, 59)  # every concept except the one that happens
                self.assertLessEqual(max(asked.values()), 2 * min(asked.values()) + 3)
            else:
                self.assertLessEqual(covered, 59)

    def test_pool_is_not_scanned_per_question(self):
        # A vocabulary of 200k concepts must stay cheap: the picker may not filter the pool
        import time

        vocabulary = [str(i) for i in range(200_000)]
        names = {c: c for c in vocabulary}
        patient = timeline([("5", day(5)), ("5", day(400))])
        rng = random.Random(0)
        config = QAConfig(will_occur_positive_fraction=0.0, time_to_event_positive_fraction=0.0)
        start = time.time()
        for _ in range(200):
            generate_qa_pairs(patient, names, vocabulary, rng, 4, config, coverage=ConceptCoverage(8))
        self.assertLess(time.time() - start, 5.0)

    def test_invalid_config(self):
        for kwargs in (
            {"will_occur_positive_fraction": 1.5},
            {"out_of_scope_fraction": -0.1},
            {"horizons_in_days": ()},
            {"horizons_in_days": (0, 30)},
            {"will_occur_weight": -1.0},
            {"will_occur_weight": 0.0, "time_to_event_weight": 0.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    QAConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
