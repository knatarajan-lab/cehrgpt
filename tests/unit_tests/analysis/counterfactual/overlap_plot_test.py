import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from cehrgpt.analysis.counterfactual import overlap_plot as plot


def scores(preferences, retained=None):
    """A scores table of the swap step: the preference is that for the other drug than the one started."""
    n = len(preferences)
    return pl.DataFrame(
        {
            "person_id": list(range(n)),
            "p_source": [0.01] * n,
            "p_comparator": [0.01] * n,
            "preference_score": preferences,
            "retained": retained if retained is not None else [True] * n,
        },
        schema_overrides={"preference_score": pl.Float64},
    )


class PreferenceTest(unittest.TestCase):
    def test_the_preference_for_the_target_is_flipped_for_those_who_started_it(self):
        started_target = scores([0.2, 0.9])  # preference for the comparator
        started_comparator = scores([0.2, 0.9])  # preference for the target
        np.testing.assert_allclose(plot.preference_for_target(started_target, True), [0.8, 0.1])
        np.testing.assert_allclose(plot.preference_for_target(started_comparator, False), [0.2, 0.9])

    def test_patients_that_could_not_be_scored_are_left_out(self):
        values = plot.preference_for_target(scores([0.5, None, 0.3]), False)
        np.testing.assert_allclose(values, [0.5, 0.3])

    def test_identical_distributions_overlap_completely_and_separate_ones_not_at_all(self):
        rng = np.random.default_rng(0)
        sample = rng.uniform(0, 1, 20000)
        self.assertAlmostEqual(plot.overlap_coefficient(sample, sample), 1.0, places=6)
        self.assertAlmostEqual(plot.overlap_coefficient(sample * 0.2, 0.8 + sample * 0.2), 0.0, places=6)
        half = plot.overlap_coefficient(sample, rng.uniform(0, 0.5, 20000))
        self.assertAlmostEqual(half, 0.5, delta=0.03)


class PlotTest(unittest.TestCase):
    def draw(self, comparator=True, trim=0.05):
        rng = np.random.default_rng(1)
        # patients who started the target tend to prefer it; their score in the file is for the comparator
        target = scores(list(1 - rng.beta(6, 3, 300)), [bool(x) for x in rng.random(300) < 0.9])
        other = scores(list(rng.beta(3, 6, 200)), [bool(x) for x in rng.random(200) < 0.8]) if comparator else None
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        output = Path(self.folder.name) / "plots" / "overlap.png"
        summary = plot.plot_preference_scores(target, other, trim, output, "ACE inhibitors", "Thiazides")
        return output, summary

    def test_the_plot_and_the_summary_are_written(self):
        output, summary = self.draw()
        self.assertGreater(output.stat().st_size, 5000)
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        written = pl.read_csv(output.with_name("overlap_summary.csv"))
        self.assertEqual(written["n_patients"].to_list(), [300, 200])
        self.assertEqual(written["n_retained"].sum(), summary["n_retained"].sum())
        self.assertTrue(0.0 < written["overlap_coefficient"][0] < 1.0)
        self.assertEqual(written["group"].to_list(), ["Started ACE inhibitors", "Started Thiazides"])

    def test_the_groups_are_ordered_by_their_preference_for_the_target(self):
        _, summary = self.draw()
        medians = dict(zip(summary["group"], summary["q50"]))
        self.assertGreater(medians["Started ACE inhibitors"], medians["Started Thiazides"])

    def test_one_group_only_has_no_overlap_coefficient(self):
        output, summary = self.draw(comparator=False)
        self.assertNotIn("overlap_coefficient", summary.columns)
        self.assertTrue(output.exists())

    def test_the_plot_can_be_drawn_without_a_trim(self):
        output, _ = self.draw(trim=None)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
