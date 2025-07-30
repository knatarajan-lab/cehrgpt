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
        drop_last: bool = False,
        negative_sampling_probability: Optional[float] = None,
        labels: Optional[List[int]] = None,
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
        self.drop_last = drop_last
        self.negative_sampling_probability = negative_sampling_probability
        self.labels = labels
        # Trainer https://github.com/huggingface/transformers/blame/main/src/transformers/trainer.py#L2470
        # http://github.com/huggingface/accelerate/blob/v0.31.0/src/accelerate/data_loader.py#L482
        # the huggingface trainer will call the accelerate.data_loader.DataLoaderShard.set_epoch,
        # which will call batch_sampler.sample.set_epoch
        self.sampler = SamplePlacerHolder()

    def __iter__(self) -> Iterator[List[int]]:

        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.sampler.epoch)
        indices = torch.randperm(len(self.lengths), generator=g).tolist()

        # Partition indices for this rank
        indices = indices[self.rank :: self.num_replicas]

        batch = []
        current_batch_tokens = 0

        for idx in indices:
            # There is a chance to skip the negative samples to account for the class imbalance
            # in the fine-tuning dataset
            if self.negative_sampling_probability:
                if (
                    np.random.random() > self.negative_sampling_probability
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
        Estimates the number of batches that will be generated.

        This is an approximation since the exact number depends on the specific
        sequence lengths and their order.
        """
        if len(self.lengths) == 0:
            return 0

        # There is a chance to skip the negative samples to account for the class imbalance
        # in the fine-tuning dataset
        if self.negative_sampling_probability:
            truncated_lengths = []
            for length, label in zip(self.lengths, self.labels):
                if (
                    np.random.random() > self.negative_sampling_probability
                    and label == 0
                ):
                    continue
                truncated_lengths.append(length)
        else:
            # We need to truncate the lengths due to the context window limit imposed by the model
            truncated_lengths = [
                min(self.max_position_embeddings, length + 2) for length in self.lengths
            ]

        # Calculate average sequence length
        avg_seq_length = sum(truncated_lengths) // len(truncated_lengths)

        # Estimate average number of sequences per batch
        seqs_per_batch = self.max_tokens_per_batch // avg_seq_length

        # Estimate total number of batches
        if self.drop_last:
            # If dropping last incomplete batch
            return len(truncated_lengths) // seqs_per_batch
        else:
            # If keeping last incomplete batch, ensure at least 1 batch
            return max(1, len(truncated_lengths) // seqs_per_batch)
