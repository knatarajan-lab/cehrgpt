from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline

from cehrgpt.tokenization.tokenization_constants import UNKNOWN_BIN


def truncated_sample(sample, standard_deviation):
    lower_quantile = stats.norm.cdf(-standard_deviation)
    upper_quantile = stats.norm.cdf(standard_deviation)
    lower_bound = np.quantile(sample, lower_quantile)
    upper_bound = np.quantile(sample, upper_quantile)
    return [x for x in sample if lower_bound <= x <= upper_bound]


def is_valid_valid_bin(token: str) -> bool:
    return token.startswith("BIN:")


def create_value_bin(bin_index: int) -> str:
    return "BIN:" + str(bin_index)


def create_sample_from_bins(bins, sample_size: int = 10_000) -> List[float]:
    """
    Generates a specified number of samples from a list of bins, each containing a fitted spline.

    This function iterates over each bin, extracts the spline, and uses it to generate a set of samples
    uniformly distributed along the x-axis defined by the spline's knots. It ensures that the total number
    of samples generated matches the specified sample size by distributing the number of samples evenly
    across the bins.

    Parameters:
        bins (List[Dict[str, UnivariateSpline]]): A list of dictionaries, each containing a 'spline' key
            with a UnivariateSpline object as its value. These splines define the data distribution within
            each bin from which samples are to be generated.
        sample_size (int, optional): The total number of samples to generate from all bins combined.
            Defaults to 10,000.

    Returns:
        List[float]: A list of sampled values, where each value is generated based on the spline functions
            provided in the bins. The total number of samples in the list will be equal to `sample_size`.

    Raises:
        ValueError: If `sample_size` is less than the number of bins, as it would not be possible to generate
            at least one sample per bin.

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> spline = UnivariateSpline(x, y, s=1)
        >>> bins = [{'spline': spline} for _ in range(5)]
        >>> samples = create_sample_from_bins(bins, 1000)
        >>> len(samples)
        1000

    Note:
        The function assumes that each bin's spline has a sufficient range of x-values (knots) to allow for
        meaningful sampling. If the range of x-values is too narrow, the uniformity of the sample distribution
        may be affected.
    """
    sample = []
    num_of_bins = len(bins)
    if num_of_bins > 0:
        sample_per_bin = sample_size // num_of_bins
        for value_bin in bins:
            bin_spline = value_bin["spline"]
            x = np.random.uniform(
                bin_spline.get_knots()[0], bin_spline.get_knots()[-1], sample_per_bin
            )
            y = bin_spline(x)
            sample.extend(y)
    return sample


def create_bins_with_spline(samples, num_bins, d_freedom=3) -> List[Dict[str, Any]]:
    """
    Divides a list of numeric samples into a specified number of bins and fits a spline to the data in each bin.

    This function first sorts the list of samples, then partitions the sorted list into `num_bins` bins. For each bin,
    a UnivariateSpline is fitted to the data within the bin, using the specified degrees of freedom. The function
    handles edge cases by assigning infinity to the bounds of the first and last bins.

    Parameters:
        samples (List[float]): A list of sample data points, which are real numbers.
        num_bins (int): The number of bins to divide the sample data into. It is assumed that there are enough samples to at least fill the bins to the minimum required for spline fitting.
        d_freedom (int, optional): The degree of freedom for the spline. Default is 1, which fits a linear spline.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a bin. Each dictionary contains:
            - 'bin_index' (int): The index of the bin.
            - 'start_val' (float): The starting value of the bin, with the first bin starting at negative infinity.
            - 'end_val' (float): The ending value of the bin, with the last bin ending at positive infinity.
            - 'spline' (UnivariateSpline): The spline object fitted to the data within the bin.

    Raises:
        ValueError: If `num_bins` is less than 1 or if there are insufficient samples to create the specified number of bins with the required minimum data points per bin.

    Example:
        >>> samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> bins = create_bins_with_spline(samples, 2)
        >>> for b in bins:
        ...     print(b['bin_index'], b['start_val'], b['end_val'])
        ...
        0 -inf 5.5
        1 5.5 inf

    Note:
        The function assumes that each bin will have at least `d_freedom + 1` samples to fit the spline. If the total number of samples is less than `num_bins * (d_freedom + 1)`, no bins will be created.
    """
    samples.sort()
    bins = []
    if len(samples) >= num_bins * (d_freedom + 1):
        samples_per_bin = len(samples) // num_bins
        for bin_index in range(0, num_bins):
            if bin_index == 0:
                start_val = float("-inf")
            else:
                start_val = samples[bin_index * samples_per_bin]

            if bin_index == num_bins - 1:
                end_val = float("inf")
            else:
                end_val = samples[(bin_index + 1) * samples_per_bin]
            x = range(bin_index * samples_per_bin, (bin_index + 1) * samples_per_bin)
            y = samples[bin_index * samples_per_bin: (bin_index + 1) * samples_per_bin]
            spline = UnivariateSpline(x, y, k=d_freedom)
            bins.append(
                {
                    "bin_index": bin_index,
                    "start_val": start_val,
                    "end_val": end_val,
                    "spline": spline,
                }
            )
    return bins
