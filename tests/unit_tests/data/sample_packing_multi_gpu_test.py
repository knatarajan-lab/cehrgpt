import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from accelerate.data_loader import BatchSamplerShard
from datasets import Dataset

import cehrgpt.runners.sample_packing_trainer as sample_packing_trainer
from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler


def _lengths(n=1001, seed=0):
    rng = random.Random(seed)
    return [rng.randint(5, 400) for _ in range(n)]


def _sampler(lengths, **kwargs):
    # How SamplePackingTrainer builds it: the same global batches on every rank, sharded by accelerate
    kwargs.setdefault("max_tokens_per_batch", 1024)
    kwargs.setdefault("max_position_embeddings", 512)
    return SamplePackingBatchSampler(lengths, num_replicas=1, rank=0, seed=42, **kwargs)


def _ranks(lengths, epoch, num_processes=2, **kwargs):
    """The batches of each rank, as accelerator.prepare shards them with even_batches."""
    shards = []
    for rank in range(num_processes):
        sampler = _sampler(lengths, **kwargs)
        sampler.sampler.set_epoch(epoch)
        shards.append(BatchSamplerShard(sampler, num_processes=num_processes, process_index=rank, even_batches=True))
    return shards


class SamplePackingMultiGpuTest(unittest.TestCase):
    def test_every_rank_gets_the_same_number_of_batches(self):
        # A rank that runs out of batches first leaves the others waiting in the gradient all-reduce forever
        lengths = _lengths()
        for epoch in range(4):
            shards = _ranks(lengths, epoch)
            counts = [len(list(shard)) for shard in shards]
            self.assertEqual(len(set(counts)), 1, f"epoch {epoch}: {counts}")
            self.assertEqual([len(shard) for shard in shards], counts)

    def test_an_epoch_covers_every_sample(self):
        # Sharding in the sampler and again in accelerate gave the ranks only half of the samples
        lengths = _lengths()
        for epoch in range(3):
            seen = {i for shard in _ranks(lengths, epoch) for batch in shard for i in batch}
            self.assertEqual(seen, set(range(len(lengths))))

    def test_negative_sampling_skips_the_same_samples_on_every_rank(self):
        lengths = _lengths()
        labels = [i % 5 == 0 for i in range(len(lengths))]
        for epoch in range(3):
            batches = []
            for _ in range(2):
                sampler = _sampler(lengths, negative_sampling_probability=0.3, labels=labels)
                sampler.sampler.set_epoch(epoch)
                batches.append(list(sampler))
            self.assertEqual(batches[0], batches[1])
            counts = [len(list(shard)) for shard in _ranks(lengths, epoch, negative_sampling_probability=0.3, labels=labels)]
            self.assertEqual(counts[0], counts[1])

    def test_length_is_the_number_of_batches(self):
        lengths = _lengths()
        sampler = _sampler(lengths)
        for epoch in range(3):
            sampler.sampler.set_epoch(epoch)
            self.assertEqual(len(sampler), len(list(sampler)))
        dropping = _sampler(lengths, drop_last=True)
        self.assertEqual(len(dropping), len(list(dropping)))
        self.assertEqual(len(_sampler([])), 0)


class SamplePackingTrainerShardingTest(unittest.TestCase):
    """The trainer leaves the sharding to accelerate: one global sequence of batches, and even batches."""

    def _trainer(self):
        trainer = object.__new__(sample_packing_trainer.SamplePackingTrainer)
        dataset = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5]], "num_of_concepts": [2, 3]})
        trainer.train_dataset = dataset
        trainer.eval_dataset = dataset
        trainer.train_lengths = None
        trainer.validation_lengths = None
        trainer.data_collator = None
        trainer.negative_sampling_probability = None
        trainer.max_tokens_per_batch = 16
        trainer.max_position_embeddings = 8
        trainer.args = SimpleNamespace(
            seed=1,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
        )
        trainer.accelerator = SimpleNamespace(prepare=lambda dataloader: dataloader)
        trainer._remove_unused_columns = lambda dataset, description=None: dataset
        return trainer

    def test_train_and_eval_samplers_do_not_shard(self):
        trainer = self._trainer()
        built = []

        def record(*args, **kwargs):
            built.append(kwargs)
            return SamplePackingBatchSampler(*args, **kwargs)

        with patch.object(sample_packing_trainer, "SamplePackingBatchSampler", side_effect=record):
            trainer.get_train_dataloader()
            trainer.get_eval_dataloader()
        self.assertEqual(len(built), 2)
        for kwargs in built:
            self.assertEqual((kwargs.get("num_replicas"), kwargs.get("rank")), (1, 0))

    def test_accelerate_evens_the_batches(self):
        source = open(sample_packing_trainer.__file__).read()
        self.assertIn("self.accelerator.even_batches = True", source)
        self.assertNotIn("self.accelerator.even_batches = False", source)


if __name__ == "__main__":
    unittest.main()
