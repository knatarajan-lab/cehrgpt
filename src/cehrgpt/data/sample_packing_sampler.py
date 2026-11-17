from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import logging

LOG = logging.get_logger("transformers")


class SamplePlacerHolder:
    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch


class SamplePackingBatchSampler(Sampler[List[int]]):
    """
    A batch sampler that creates batches by packing samples together.

    to maximize GPU utilization, ensuring the total tokens per batch
    doesn't exceed max_tokens.
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens_per_batch: int,
        max_position_embeddings: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        negative_sampling_probability: Optional[float] = None,
        labels: Optional[List[int]] = None,
        num_batches_multiple_of: int = 1,
    ):
        """
        Args:

            lengths: List of sequence lengths for each sample
            max_tokens: Maximum number of tokens in a batch
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__()

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                LOG.info(
                    "torch.distributed is initialized and there are %s of replicas",
                    num_replicas,
                )
            else:
                num_replicas = 1
                LOG.info(
                    "torch.dist is not initialized and therefore default to 1 for num_replicas"
                )

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                LOG.info(
                    "torch.distributed is initialized and the current rank is %s", rank
                )
            else:
                rank = 0
                LOG.info(
                    "torch.distributed is not initialized and therefore default to 0 for rank"
                )

        if not (0 <= rank < num_replicas):
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        if negative_sampling_probability is not None and labels is None:
            raise ValueError(
                f"When the negative sampling probability is provide, the labels must be provided as well"
            )

        self.lengths = lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_position_embeddings = max_position_embeddings
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.negative_sampling_probability = negative_sampling_probability
        self.labels = labels
        self.num_batches_multiple_of = num_batches_multiple_of
        # Trainer https://github.com/huggingface/transformers/blame/main/src/transformers/trainer.py#L2470
        # http://github.com/huggingface/accelerate/blob/v0.31.0/src/accelerate/data_loader.py#L482
        # the huggingface trainer will call the accelerate.data_loader.DataLoaderShard.set_epoch,
        # which will call batch_sampler.sample.set_epoch
        self.sampler = SamplePlacerHolder()
        self._length_cache = None

    def __iter__(self) -> Iterator[List[int]]:
        batches = list(self._pack())
        # accelerate gives batch i to process i % num_processes and cannot even out batches without a fixed size,
        # so repeat the first batches until every process gets the same number
        remainder = len(batches) % self.num_batches_multiple_of
        if batches and remainder:
            batches += [batches[i % len(batches)] for i in range(self.num_batches_multiple_of - remainder)]
        return iter(batches)

    def _pack(self) -> Iterator[List[int]]:
        if self.shuffle:
            # Deterministically shuffle training samples based on epoch and seed.
            g = torch.Generator()
            g.manual_seed(self.seed + self.sampler.epoch)
            indices = torch.randperm(len(self.lengths), generator=g).tolist()
        else:
            # Evaluation must visit samples in the same order on every epoch.
            indices = list(range(len(self.lengths)))

        # Partition indices for this rank
        indices = indices[self.rank :: self.num_replicas]

        # The negative sampling draws depend only on the seed and the epoch, so that every rank skips the same
        # samples and yields the same batches when the batches are sharded across the ranks afterwards
        negative_sampling_rng = np.random.default_rng(self.seed + self.sampler.epoch)

        batch = []
        current_batch_tokens = 0

        for idx in indices:
            # There is a chance to skip the negative samples to account for the class imbalance
            # in the fine-tuning dataset
            if self.negative_sampling_probability:
                if (
                    negative_sampling_rng.random() > self.negative_sampling_probability
                    and self.labels[idx] == 0
                ):
                    continue
            # We take the minimum of the two because each sequence will be truncated to fit
            # the context window of the model
            sample_length = min(self.lengths[idx], self.max_position_embeddings)
            # If adding this sample would exceed max_tokens_per_batch, yield the current batch
            if (
                current_batch_tokens + sample_length + 2 > self.max_tokens_per_batch
                and batch
            ):
                yield batch
                batch = []
                current_batch_tokens = 0

            # Add the sample to the current batch
            batch.append(idx)
            # plus extract one for the [END] and [PAD] tokens to separate samples
            current_batch_tokens += sample_length + 2

        # Yield the last batch if it's not empty and we're not dropping it
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """
        The number of batches that __iter__ yields in the current epoch.

        The batches depend only on the lengths, the seed and the epoch, so they are counted exactly by packing them.
        An estimate from the average length was off by over 10%, which gave the trainer a wrong number of steps per
        epoch and a wrong learning-rate schedule, and the distributed trainer needs the real number of batches.
        """
        epoch = self.sampler.epoch
        if self._length_cache is None or self._length_cache[0] != epoch:
            self._length_cache = (epoch, sum(1 for _ in self.__iter__()))
        return self._length_cache[1]
