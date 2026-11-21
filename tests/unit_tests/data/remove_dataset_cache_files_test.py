import glob
import os
import tempfile
import unittest

from datasets import (
    Dataset,
    DatasetDict,
    disable_caching,
    enable_caching,
    is_caching_enabled,
    load_from_disk,
)

from cehrgpt.data.cache_cleanup import clean_up_dataset_caches, remove_dataset_cache_files


def _slice(batch):
    # Like the cohort slicing: every person gives two sequences
    return {
        "person_id": [p for p in batch["person_id"] for _ in range(2)],
        "input_ids": [ids[: i + 1] for ids in batch["input_ids"] for i in range(2)],
    }


class RemoveDatasetCacheFilesTest(unittest.TestCase):
    """
    Slicing a cohort out of the tokenized dataset writes cache files next to that dataset.

    The slice's own cache files count as used, so clean_up_dataset_caches kept them, and every linear probing
    task left a copy of its slice (80 to 125 GB for MIMIC) in the folder of the tokenized dataset.
    """

    def setUp(self):
        was_enabled = is_caching_enabled()
        enable_caching()
        self.addCleanup(enable_caching if was_enabled else disable_caching)
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.tokenized_path = os.path.join(self.root.name, "tokenized")
        DatasetDict(
            {
                split: Dataset.from_dict(
                    {"person_id": list(ids), "input_ids": [[p, p + 1, p + 2] for p in ids]}
                )
                for split, ids in (("train", range(0, 300)), ("test", range(1000, 1100)))
            }
        ).save_to_disk(self.tokenized_path)
        self.original_files = self._files(self.tokenized_path)

    @staticmethod
    def _files(folder):
        return sorted(os.path.relpath(f, folder) for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True))

    def _sliced(self):
        tokenized = load_from_disk(self.tokenized_path)
        cohort = set(range(0, 300, 3)) | set(range(1000, 1100, 2))
        filtered = DatasetDict(
            {
                split: ds.filter(lambda batch: [p in cohort for p in batch["person_id"]], batched=True)
                for split, ds in tokenized.items()
            }
        )
        return DatasetDict(
            {
                split: ds.map(_slice, batched=True, remove_columns=ds.column_names)
                for split, ds in filtered.items()
            }
        )

    def test_the_tokenized_dataset_is_left_as_it_was(self):
        sliced = self._sliced()
        self.assertTrue(glob.glob(os.path.join(self.tokenized_path, "*", "cache-*.arrow")))
        saved_path = os.path.join(self.root.name, "prepared")
        sliced.save_to_disk(saved_path)
        clean_up_dataset_caches(sliced)
        # what was left before this fix: the cache files of the slice itself
        self.assertTrue(glob.glob(os.path.join(self.tokenized_path, "*", "cache-*.arrow")))

        self.assertGreater(remove_dataset_cache_files(sliced), 0)
        self.assertEqual(self._files(self.tokenized_path), self.original_files)
        self.assertEqual(len(load_from_disk(self.tokenized_path)["train"]), 300)

        reloaded = load_from_disk(saved_path)
        self.assertEqual(len(reloaded["train"]), 200)
        self.assertEqual(len(reloaded["test"]), 100)
        self.assertEqual(reloaded["train"][1], {"person_id": 0, "input_ids": [0, 1]})

    def test_a_dataset_loaded_from_disk_keeps_its_files(self):
        # load_from_disk gives data-*.arrow files, which are not cache files
        self.assertEqual(remove_dataset_cache_files(load_from_disk(self.tokenized_path)), 0)
        self.assertEqual(self._files(self.tokenized_path), self.original_files)
        self.assertEqual(remove_dataset_cache_files(Dataset.from_dict({"person_id": [1, 2]})), 0)


if __name__ == "__main__":
    unittest.main()
