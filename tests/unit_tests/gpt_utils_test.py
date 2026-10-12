import random
import unittest

from cehrgpt import gpt_utils
from cehrgpt.gpt_utils import (
    ETHOS_TIME_BUCKET_TO_DAYS,
    RandomSampleCache,
    construct_age_sequence,
    convert_time_interval_to_time_tuple,
    extract_time_interval_in_days,
    generate_artificial_time_tokens,
    is_att_token,
    is_ethos_time_bucket_token,
    is_inpatient_att_token,
    is_visit_start,
    random_slice_gpt_sequence,
    truncate_history_keeping_demographics,
)


class TestRandomSampleCache(unittest.TestCase):
    def test_cache_sampling_without_weights(self):
        cache = RandomSampleCache([1, 2, 3], cache_size=2)
        sample_1 = cache.next()
        sample_2 = cache.next()
        self.assertIn(sample_1, [1, 2, 3])
        self.assertIn(sample_2, [1, 2, 3])

    def test_cache_sampling_with_weights(self):
        cache = RandomSampleCache(
            [1, 2, 3], cache_size=2, sample_weights=[0.1, 0.1, 0.8]
        )
        sample = cache.next()
        self.assertIn(sample, [1, 2, 3])

    def test_cache_refill(self):
        cache = RandomSampleCache([1], cache_size=2)
        cache.next()
        self.assertEqual(len(cache._cache), 1)


class TestRandomSliceGPTSequence(unittest.TestCase):
    def test_random_slice_gpt_sequence_with_empty_starting_points(self):
        result = random_slice_gpt_sequence(["year:2020", "age:30", "male", "race"], 10)
        self.assertEqual((0, 0, ["year:2020", "age:30", "male", "race"]), result)

    def test_random_slice_gpt_sequence_with_valid_data(self):

        random.seed(0)

        # The pat sequence is greater than max_seq_len,
        # it should randomly pick a starting point and do not update the demographic prompt
        # because the time interval between the visits is 7 days
        concept_ids = [
            "year:2020",
            "age:30",
            "male",
            "race",
            "VS",
            "C",
            "C",
            "VE",
            "D7",
            "VS",
            "C",
            "VE",
        ]
        result = random_slice_gpt_sequence(concept_ids, 10)
        self.assertTrue(result[0] == 4)
        self.assertTrue(result[1] == 7)
        self.assertEqual(result[2], ["year:2020", "age:30", "male", "race"])

        concept_ids = [
            "year:2020",
            "age:30",
            "male",
            "race",
            "VS",
            "C",
            "C",
            "VE",
            "D500",
            "VS",
            "C",
            "VE",
            "VS",
            "C",
            "C",
            "VE",
            "D500",
            "VS",
            "C",
            "VE",
            "VS",
            "C",
            "C",
            "VE",
            "D500",
            "VS",
            "C",
            "VE",
        ]

        # The pat sequence is greater than max_seq_len,
        # it should randomly pick a starting point and update the demographic prompt
        result = random_slice_gpt_sequence(concept_ids, 20)
        self.assertTrue(result[0] == 9)
        self.assertTrue(result[1] == 23)
        self.assertEqual(result[2], ["year:2021", "age:31", "male", "race"])

        # The pat sequence is shorter than max_seq_len, it should return the original demographic prompt
        concept_ids = ["year:2020", "age:30", "male", "race", "VS", "C", "C", "VE"]
        result = random_slice_gpt_sequence(concept_ids, 10)
        self.assertTrue(result[0] == 0)
        self.assertTrue(result[1] == 0)
        self.assertEqual(result[2], ["year:2020", "age:30", "male", "race"])

    def test_random_slice_gpt_sequence_with_invalid_data(self):
        concept_ids = ["year:2020", "age:30", "male", "race", "invalid"]
        result = random_slice_gpt_sequence(concept_ids, 10)
        self.assertEqual(result, (0, 0, ["year:2020", "age:30", "male", "race"]))


class TestTokenFunctions(unittest.TestCase):
    def test_is_visit_start(self):
        self.assertTrue(is_visit_start("VS"))
        self.assertFalse(is_visit_start("D1"))

    def test_is_att_token(self):
        self.assertTrue(is_att_token("D7"))
        self.assertTrue(is_att_token("VS-D7"))
        self.assertFalse(is_att_token("random"))

    def test_is_inpatient_att_token(self):
        self.assertTrue(is_inpatient_att_token("VS-D7"))
        self.assertTrue(is_inpatient_att_token("VS-D7-VE"))
        self.assertTrue(is_inpatient_att_token("i-D7"))
        self.assertTrue(is_inpatient_att_token("i-D7-VE"))
        self.assertFalse(is_inpatient_att_token("V-D7"))
        self.assertFalse(is_inpatient_att_token("D7-VE"))


class TestTimeIntervalFunctions(unittest.TestCase):
    def test_extract_time_interval_in_days(self):
        self.assertEqual(extract_time_interval_in_days("D7"), 7)
        self.assertEqual(extract_time_interval_in_days("W1"), 7)
        self.assertEqual(extract_time_interval_in_days("M1"), 30)
        self.assertEqual(extract_time_interval_in_days("Y1"), 365)

    def test_extract_time_interval_invalid(self):
        with self.assertRaises(ValueError):
            extract_time_interval_in_days("invalid")

    def test_convert_time_interval_to_time_tuple(self):
        result = convert_time_interval_to_time_tuple(400, is_inpatient=True)
        self.assertEqual(result, ("year:1", "month:1", "i-day:5"))

    def test_convert_time_interval_negative(self):
        with self.assertRaises(AssertionError):
            convert_time_interval_to_time_tuple(-10, is_inpatient=False)


class TestGenerateArtificialTimeTokens(unittest.TestCase):
    def test_generate_artificial_time_tokens(self):
        tokens = generate_artificial_time_tokens()
        self.assertTrue("D1" in tokens)
        self.assertTrue("W1" in tokens)
        self.assertTrue("M1" in tokens)
        self.assertTrue("LT" in tokens)


class TestEthosTimeTokens(unittest.TestCase):
    def test_every_bucket_and_its_inpatient_variant_converts_to_days(self):
        for bucket, days in ETHOS_TIME_BUCKET_TO_DAYS.items():
            for token in (bucket, f"i-{bucket}"):
                self.assertTrue(is_ethos_time_bucket_token(token), token)
                self.assertTrue(is_att_token(token), token)
                self.assertEqual(extract_time_interval_in_days(token), days, token)

    def test_six_month_token_is_180_days_and_repeats_add_up(self):
        self.assertAlmostEqual(extract_time_interval_in_days("=6mt"), 180, delta=1)
        total = sum(extract_time_interval_in_days(t) for t in ["=6mt"] * 3)
        self.assertAlmostEqual(total, 540, delta=3)

    def test_legacy_bucket_labels_still_parse(self):
        self.assertEqual(extract_time_interval_in_days("2w-1mt"), 22)
        self.assertEqual(extract_time_interval_in_days("i-12h-1d"), 1)

    def test_standard_att_tokens_are_unchanged(self):
        self.assertEqual(extract_time_interval_in_days("D7"), 7)
        self.assertEqual(extract_time_interval_in_days("W2"), 14)
        self.assertEqual(extract_time_interval_in_days("i-D3"), 3)
        self.assertEqual(extract_time_interval_in_days("LT"), 365 * 3)
        self.assertFalse(is_ethos_time_bucket_token("D7"))

    def test_non_time_tokens_are_not_time_buckets(self):
        for token in ("ICD9CM/0/311", "Visit/OP", "VALUE_BIN/8", "Gender/F"):
            self.assertFalse(is_ethos_time_bucket_token(token), token)


class TestConstructAgeSequenceWarning(unittest.TestCase):
    def setUp(self):
        gpt_utils._warned_about_missing_age_token = False

    def test_missing_age_token_warns_once_and_still_returns_zeros(self):
        prompt = ["[VS]", "9202", "193782", "43021974", "[VE]"]
        with self.assertLogs(gpt_utils.logger.name, level="WARNING") as logs:
            results = [construct_age_sequence(prompt) for _ in range(50)]
        self.assertEqual(len(logs.output), 1)
        self.assertIn("not a valid age token", logs.output[0])
        self.assertTrue(all(result == [0] * len(prompt) for result in results))

    def test_valid_age_token_never_warns(self):
        with self.assertNoLogs(gpt_utils.logger.name, level="WARNING"):
            ages = construct_age_sequence(["year:2020", "age:40", "8507", "8527", "[VS]", "D400"])
        self.assertEqual(ages, [40, 40, 40, 40, 40, 41])

    def test_stored_ages_are_returned_without_a_warning(self):
        with self.assertNoLogs(gpt_utils.logger.name, level="WARNING"):
            self.assertEqual(construct_age_sequence(["[VS]", "9202"], [3, 4]), [3, 4])


def make_history(num_visits, att="M6", header=("year:2000", "age:30", "8507", "8527")):
    tokens = list(header)
    for visit in range(num_visits):
        if visit:
            tokens.append(att)
        tokens += ["[VS]", "9202", f"c{visit}a", f"c{visit}b", "[VE]"]
    return tokens


class TestTruncateHistoryKeepingDemographics(unittest.TestCase):
    HEADER_SIZE = 4

    def test_history_that_fits_is_unchanged(self):
        history = make_history(3)
        self.assertEqual(truncate_history_keeping_demographics(history, 1000), history)
        # the budget only counts the history, not the 4 header tokens
        budget = len(history) - self.HEADER_SIZE
        self.assertEqual(truncate_history_keeping_demographics(history, budget), history)
        self.assertNotEqual(
            truncate_history_keeping_demographics(history, budget - 1), history
        )

    def test_budget_does_not_count_the_header(self):
        history = make_history(40)
        result = truncate_history_keeping_demographics(history, 60)
        history_tokens = result[self.HEADER_SIZE:]
        self.assertLessEqual(len(history_tokens), 60)
        # the result is the header on top of the budget, not squeezed into it
        self.assertGreater(len(result), 60)
        self.assertLessEqual(len(result), 60 + self.HEADER_SIZE)

    def test_truncated_history_keeps_a_rebuilt_header_and_starts_on_a_visit(self):
        history = make_history(40)  # 40 visits, 6 months apart: 19.5 years
        result = truncate_history_keeping_demographics(history, 60)
        self.assertEqual(result[2:4], ["8507", "8527"])
        self.assertTrue(result[0].startswith("year:") and result[1].startswith("age:"))
        self.assertEqual(result[4], "[VS]")
        # the rest is the most recent part of the original history
        recent = result[4:]
        self.assertEqual(history[-len(recent):], recent)

    def test_year_and_age_move_forward_by_the_dropped_time(self):
        history = make_history(40, att="Y1")  # one year between visits
        result = truncate_history_keeping_demographics(history, 40)
        recent = result[self.HEADER_SIZE:]
        dropped_visits = (len(history) - self.HEADER_SIZE - len(recent)) // 6
        # each dropped visit (with its ATT token) is one year
        self.assertGreater(dropped_visits, 0)
        self.assertEqual(result[0], f"year:{2000 + dropped_visits}")
        self.assertEqual(result[1], f"age:{30 + dropped_visits}")

    def test_rebuilt_header_reproduces_the_ages_of_the_full_history(self):
        history = make_history(40, att="Y1")
        truncated = truncate_history_keeping_demographics(history, 40)
        full_ages = construct_age_sequence(history)
        truncated_ages = construct_age_sequence(truncated)
        # the last tokens are the same tokens of the same patient, so the ages must agree
        n = len(truncated) - self.HEADER_SIZE
        self.assertEqual(truncated_ages[-n:], full_ages[-n:])

    def test_history_without_a_header_is_truncated_to_the_budget(self):
        history = ["[VS]", "9202", "a", "[VE]", "M1"] * 20
        result = truncate_history_keeping_demographics(history, 30)
        self.assertLessEqual(len(result), 30)
        self.assertEqual(result[0], "[VS]")
        self.assertEqual(history[-len(result):], result)

    def test_invalid_age_token_is_treated_as_no_header(self):
        history = make_history(40, header=("year:2000", "age:oops", "8507", "8527"))
        result = truncate_history_keeping_demographics(history, 60)
        self.assertLessEqual(len(result), 60)
        self.assertEqual(result[0], "[VS]")
        self.assertNotIn("year:2000", result)

    def test_a_tiny_budget_still_keeps_the_header(self):
        result = truncate_history_keeping_demographics(make_history(10), 4)
        self.assertTrue(result[0].startswith("year:"))
        self.assertLessEqual(len(result), 4 + self.HEADER_SIZE)


if __name__ == "__main__":
    unittest.main()
