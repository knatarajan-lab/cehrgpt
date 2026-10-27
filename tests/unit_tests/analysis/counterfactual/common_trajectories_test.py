import unittest

import polars as pl

from cehrgpt.analysis.counterfactual import hazard_ratio_estimation as hr


def trajectories(pairs):
    """Two events for each (subject_id, trajectory_id) pair."""
    rows = [(s, t, code) for s, t in pairs for code in ("a", "b")]
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "trajectory_id": [r[1] for r in rows],
            "code": [r[2] for r in rows],
        }
    ).with_row_index("position")


def pairs_of(frame):
    return sorted(set(zip(frame["subject_id"].to_list(), frame["trajectory_id"].to_list())))


class CommonTrajectoriesTest(unittest.TestCase):
    def test_only_the_pairs_of_both_arms_are_kept(self):
        a = trajectories([(1, 0), (1, 1), (2, 0), (2, 1)])
        b = trajectories([(1, 0), (2, 0), (2, 1), (3, 0)])
        kept_a, kept_b, n_common, n_left_out = hr.restrict_to_common_trajectories(a, b)
        self.assertEqual(pairs_of(kept_a), [(1, 0), (2, 0), (2, 1)])
        self.assertEqual(pairs_of(kept_b), [(1, 0), (2, 0), (2, 1)])
        self.assertEqual((n_common, n_left_out), (3, 2))

    def test_all_the_events_of_a_kept_pair_stay(self):
        a = trajectories([(1, 0), (2, 0)])
        kept_a, _, _, _ = hr.restrict_to_common_trajectories(a, trajectories([(1, 0)]))
        self.assertEqual(kept_a["code"].to_list(), ["a", "b"])
        self.assertEqual(kept_a["position"].to_list(), [0, 1])

    def test_an_arm_that_is_not_generated_yet_leaves_nothing_to_compare(self):
        with self.assertRaisesRegex(ValueError, "generated in both arms"):
            hr.restrict_to_common_trajectories(trajectories([(1, 0)]), trajectories([(2, 0)]))

    def test_arms_with_the_same_pairs_are_unchanged(self):
        a = trajectories([(1, 0), (2, 1)])
        kept_a, kept_b, n_common, n_left_out = hr.restrict_to_common_trajectories(a, a)
        self.assertEqual((n_common, n_left_out), (2, 0))
        self.assertEqual(len(kept_a), len(a))
        self.assertEqual(len(kept_b), len(a))


if __name__ == "__main__":
    unittest.main()
