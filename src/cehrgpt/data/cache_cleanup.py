"""
Removing the cache files that a saved dataset no longer needs.

Two things clean up after a dataset is built, and the order they run in matters:

  * ``CacheFileCollector.remove_cache_files()`` deletes the cache files that were recorded while the dataset was built,
    and with them the whole folder of the raw dataset that they came from (the parquet files turned into Arrow, which
    is about ten times their size). It finds that folder from the path of a recorded file that still exists.
  * ``Dataset.cleanup_cache_files()`` deletes every ``cache-*.arrow`` of the cache folder of a dataset that the dataset
    itself does not use. Splits that share a cache folder delete each other's files this way.

Run in this order, the second leaves the first without a file to find the raw folder from, and the raw dataset stays on
disk. The collector goes first here, and the second is only for what is left.
"""

from typing import Dict, Optional, Union

from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from datasets import Dataset, DatasetDict
from transformers.utils import logging

LOG = logging.get_logger("transformers")


def _cleanup_cache_files(dataset: Dataset) -> int:
    try:
        return dataset.cleanup_cache_files()
    except FileNotFoundError:
        # the folder of the cache files has been removed together with the raw dataset
        return 0


def clean_up_dataset_caches(
    dataset: Union[Dataset, DatasetDict],
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> Union[int, Dict[str, int]]:
    """
    Remove the cache files of a dataset that has been saved to disk, and the raw datasets that it was built from.

    The dataset must not be used from its cache files after this: load it again from where it was saved.
    Returns the number of files that Dataset.cleanup_cache_files removed after the collector (per split for a
    DatasetDict).
    """
    if cache_file_collector is not None:
        cache_file_collector.remove_cache_files()
    if isinstance(dataset, DatasetDict):
        return {split: _cleanup_cache_files(split_dataset) for split, split_dataset in dataset.items()}
    return _cleanup_cache_files(dataset)
