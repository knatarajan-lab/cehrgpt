import unittest

from cehrgpt.time_to_event.time_to_event_model import validate_zero_shot_time_tokens

NON_TIME_TOKENS = [
    "year:2014",
    "age:50",
    "Gender/F",
    "Race/0",
    "Visit/OP",
    "ICD9CM/0/311",
    "ICD10CM/1/41XA",
    "LOINC/14336-2",
    "VALUE_BIN/8",
    "8532",
    "[VS]",
    "[VE]",
    "[END]",
    "LT",
]


class TestValidateZeroShotTimeTokens(unittest.TestCase):
    def test_day_tokens_are_supported(self):
        validate_zero_shot_time_tokens(
            NON_TIME_TOKENS + [f"D{i}" for i in range(30)] + ["i-D1", "i-D14"]
        )

    def test_ethos_comet_tokens_are_supported(self):
        validate_zero_shot_time_tokens(
            NON_TIME_TOKENS + ["5m-15m", "i-5m-15m", "3mt-6mt", "i-3mt-6mt"]
        )

    def test_week_month_cehr_bert_and_mix_tokens_are_rejected(self):
        # what WEEK, MONTH, CEHR_BERT and MIX tokenizers put in the vocabulary
        for time_tokens in (
            ["W0", "W1", "W2"],
            ["M0", "M1", "M11"],
            ["W0", "W3", "M0", "M11", "W-1"],
            ["D0", "D7", "W2", "W5", "M2", "M12"],
            ["D1", "i-W1"],
            ["D1", "Y1"],
        ):
            with self.assertRaises(ValueError, msg=time_tokens) as ctx:
                validate_zero_shot_time_tokens(NON_TIME_TOKENS + time_tokens)
            self.assertIn("DAY", str(ctx.exception))

    def test_rejection_lists_the_offending_tokens(self):
        with self.assertRaises(ValueError) as ctx:
            validate_zero_shot_time_tokens(["D1", "W2", "M3"])
        self.assertIn("W2", str(ctx.exception))
        self.assertIn("M3", str(ctx.exception))

    def test_vocab_without_any_time_tokens_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_zero_shot_time_tokens(NON_TIME_TOKENS)

    def test_accepts_a_vocab_dict(self):
        validate_zero_shot_time_tokens({"D1": 0, "Visit/OP": 1})


if __name__ == "__main__":
    unittest.main()
