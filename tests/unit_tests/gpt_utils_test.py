import random
import unittest

from cehrgpt.gpt_utils import (
    RandomSampleCache,
    convert_time_interval_to_time_tuple,
    extract_time_interval_in_days,
    generate_artificial_time_tokens,
    is_att_token,
    is_inpatient_att_token,
    is_visit_start,
    random_slice_gpt_sequence,
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


if __name__ == "__main__":
    unittest.main()
