import unittest

from scipy.interpolate import UnivariateSpline

from cehrgpt.models.tokenization_hf_cehrgpt import create_bins_with_spline


class TestCreateBinsWithSpline(unittest.TestCase):
    def test_bins_count(self):
        samples = list(range(100))
        num_bins = 10
        bins = create_bins_with_spline(samples, num_bins)
        self.assertEqual(len(bins), num_bins)

    def test_bins_edges(self):
        samples = list(range(100))
        num_bins = 10
        bins = create_bins_with_spline(samples, num_bins)
        self.assertEqual(bins[0]["start_val"], float("-inf"))
        self.assertEqual(bins[-1]["end_val"], float("inf"))

    def test_few_samples(self):
        samples = list(
            range(5)
        )  # Not enough samples to form the requested number of bins
        num_bins = 10
        bins = create_bins_with_spline(samples, num_bins)
        self.assertEqual(len(bins), 0)

    def test_spline_fitting(self):
        samples = list(range(20))
        num_bins = 2
        bins = create_bins_with_spline(samples, num_bins)
        self.assertIsInstance(bins[0]["spline"], UnivariateSpline)
        self.assertIsInstance(bins[1]["spline"], UnivariateSpline)

    def test_spline_evaluation(self):
        samples = [x * x for x in range(10)]  # Quadratic relationship
        num_bins = 1
        bins = create_bins_with_spline(samples, num_bins)
        spline = bins[0]["spline"]
        x = 5
        # Check if the spline at mid-point roughly equals the mid-point y-value in samples
        self.assertAlmostEqual(spline(x), samples[x], delta=1.0)


if __name__ == "__main__":
    unittest.main()
