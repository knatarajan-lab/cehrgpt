import glob
import os
import tempfile
import unittest

import pandas as pd
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from datasets import (
    Dataset,
    DatasetDict,
    disable_caching,
    enable_caching,
    is_caching_enabled,
    load_from_disk,
)

from cehrgpt.data.cache_cleanup import clean_up_dataset_caches


def double(batch):
    return {"person_id": batch["person_id"], "twice": [2 * p for p in batch["person_id"]]}


class CleanUpDatasetCachesTest(unittest.TestCase):
    """The raw datasets and the cache files of the splits are all removed once the tokenized dataset is saved."""

    def setUp(self):
        # Other tests of the session switch the caching of datasets off. With it off the results of a map are not
        # cache files of the cache folder, which is not what a run of the runners has.
        was_enabled = is_caching_enabled()
        enable_caching()
        self.addCleanup(enable_caching if was_enabled else disable_caching)
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        # the cache folder has a datasets folder in its path, as the cache of Hugging Face does
        self.cache = os.path.join(self.root.name, "hf_home", "datasets")
        for name, ids in (("train", range(0, 400)), ("test", range(1000, 1100))):
            os.makedirs(os.path.join(self.root.name, name))
            pd.DataFrame({"person_id": list(ids)}).to_parquet(os.path.join(self.root.name, name, "part.parquet"))

    def build(self):
        """The splits of the pretraining runner: train and validation from one folder, test from another."""
        train = Dataset.from_parquet(os.path.join(self.root.name, "train", "*.parquet"), cache_dir=self.cache)
        test = Dataset.from_parquet(os.path.join(self.root.name, "test", "*.parquet"), cache_dir=self.cache)
        split = train.train_test_split(test_size=0.2, seed=1)
        splits = DatasetDict({"train": split["train"], "validation": split["test"], "test": test})
        collector = CacheFileCollector()
        mapped = DatasetDict(
            {name: ds.map(double, batched=True, batch_size=50, num_proc=2) for name, ds in splits.items()}
        )
        collector.add_cache_files(mapped)
        saved = os.path.join(self.root.name, "saved")
        mapped.save_to_disk(saved)
        return mapped, collector, saved

    def arrow_files(self):
        return sorted(
            os.path.relpath(path, self.cache)
            for path in glob.glob(os.path.join(self.cache, "**", "*.arrow"), recursive=True)
        )

    def test_cleaning_up_each_split_first_leaves_a_raw_dataset_behind(self):
        # what the runners did: Dataset.cleanup_cache_files of the splits, and then the collector
        mapped, collector, _ = self.build()
        mapped.cleanup_cache_files()
        collector.remove_cache_files()
        self.assertTrue(
            any("parquet-train.arrow" in path for path in self.arrow_files()), "the raw Arrow dataset was removed"
        )

    def test_nothing_is_left_after_the_helper(self):
        mapped, collector, saved = self.build()
        self.assertTrue(self.arrow_files())
        clean_up_dataset_caches(mapped, collector)
        self.assertEqual(self.arrow_files(), [])
        # the saved dataset does not depend on the cache
        reloaded = load_from_disk(saved)
        self.assertEqual({k: len(v) for k, v in reloaded.items()}, {"train": 320, "validation": 80, "test": 100})
        self.assertEqual(reloaded["test"]["twice"][:3], [2000, 2002, 2004])

    def test_without_a_collector_the_cache_files_that_are_not_used_are_removed(self):
        mapped, _, _ = self.build()
        counts = clean_up_dataset_caches(mapped)
        self.assertEqual(set(counts), {"train", "validation", "test"})
        self.assertGreater(counts["train"] + counts["validation"], 0)

    def test_a_single_dataset(self):
        train = Dataset.from_parquet(os.path.join(self.root.name, "train", "*.parquet"), cache_dir=self.cache)
        collector = CacheFileCollector()
        mapped = train.map(double, batched=True, batch_size=50, num_proc=2)
        collector.add_cache_files(mapped)
        clean_up_dataset_caches(mapped, collector)
        self.assertEqual(self.arrow_files(), [])


if __name__ == "__main__":
    unittest.main()
