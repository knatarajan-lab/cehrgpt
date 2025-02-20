import unittest

from cehrgpt.rl_runners.grpo.rewards import DemographicGroup, reward_co_occurrence


class TestRewardCoOccurrence(unittest.TestCase):
    def setUp(self):
        # Example co_occurrence_matrices setup
        self.co_occurrence_matrices = [
            {DemographicGroup("Adult", "White", "Male"): {("1", "2"): 0.5}}
        ]
        self.kwargs = {
            "co_occurrence_matrices": [
                (
                    0,
                    30,
                    {DemographicGroup("age:40-50", "White", "Male"): {("1", "2"): 0.5}},
                ),
                (
                    30,
                    30,
                    {DemographicGroup("age:40-50", "White", "Male"): {("1", "2"): 0.8}},
                ),
            ]
        }

    def test_valid_input(self):
        # Testing with a hypothetical valid input
        prompts = [["John", "age:45", "Male", "White"]]
        completions = [["[VS]", "1", "[VE]", "D7", "[VS]", "2", "[VE]"]]
        expected_rewards = [
            0.5 / 7
        ]  # Assuming reward logic and normalization by length of completion
        rewards = reward_co_occurrence(prompts, completions, **self.kwargs)
        self.assertEqual(rewards, expected_rewards)

        # Testing with a hypothetical valid input
        prompts = [["John", "age:45", "Male", "White"]]
        completions = [["[VS]", "1", "[VE]", "D56", "[VS]", "2", "[VE]"]]
        expected_rewards = [
            0.8 / 7
        ]  # Assuming reward logic and normalization by length of completion
        rewards = reward_co_occurrence(prompts, completions, **self.kwargs)
        self.assertEqual(rewards, expected_rewards)

    def test_empty_input(self):
        # Testing with empty inputs
        prompts = [[]]
        completions = [[]]
        expected_rewards = [0.0]
        rewards = reward_co_occurrence(prompts, completions, **self.kwargs)
        self.assertEqual(rewards, expected_rewards)

    def test_no_numeric_ids(self):
        # Testing where no numeric concept IDs are present
        prompts = [["Jane", "age:30", "Female", "Black"]]
        completions = [["att7days", "hello", "world"]]
        expected_rewards = [0.0]
        rewards = reward_co_occurrence(prompts, completions, **self.kwargs)
        self.assertEqual(rewards, expected_rewards)


# Run the tests
if __name__ == "__main__":
    unittest.main()
