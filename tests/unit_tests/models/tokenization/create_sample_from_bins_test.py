import unittest

import numpy as np
from scipy.interpolate import UnivariateSpline

from cehrgpt.models.tokenization_hf_cehrgpt import create_sample_from_bins


class TestCreateSampleFromBins(unittest.TestCase):
    def setUp(self):
        # Create a set of bins with splines
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        spline = UnivariateSpline(x, y, s=1)
        self.bins = [{"spline": spline} for _ in range(5)]

    def test_sample_size(self):
        sample_size = 10_000
        samples = create_sample_from_bins(self.bins, sample_size)
        self.assertEqual(len(samples), sample_size)

    def test_empty_bins(self):
        samples = create_sample_from_bins([], 10_000)
        self.assertEqual(len(samples), 0)

    def test_zero_sample_size(self):
        samples = create_sample_from_bins(self.bins, 0)
        self.assertEqual(len(samples), 0)

    def test_sample_distribution(self):
        """Check if samples respect the distribution from the spline by checking mean close to expected."""
        x = np.linspace(0, 10, 10)
        y = np.sin(x)
        spline = UnivariateSpline(x, y, s=1)
        bins = [{"spline": spline}]
        samples = create_sample_from_bins(bins, 1000)
        expected_mean = np.mean(spline(x))
        samples_mean = np.mean(samples)
        self.assertAlmostEqual(samples_mean, expected_mean, delta=0.1)


if __name__ == "__main__":
    unittest.main()
