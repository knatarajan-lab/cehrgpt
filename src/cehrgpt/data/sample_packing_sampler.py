from typing import Iterator, List

import torch
from torch.utils.data import RandomSampler, Sampler
from transformers import logging

logger = logging.get_logger(__name__)


class SamplePackingSampler(Sampler):
    """
    A batch sampler that creates batches by packing samples together.

    to maximize GPU utilization, ensuring the total tokens per batch
    doesn't exceed max_tokens.
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens: int,
        world_size: int,
        drop_last: bool = False,
    ):
        """
        Args:

            lengths: List of sequence lengths for each sample
            max_tokens: Maximum number of tokens in a batch
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__()
        self.lengths = lengths
        self.world_size = max(1, world_size)
        self.max_tokens = max_tokens
        self.drop_last = drop_last
        self.sampler = RandomSampler(lengths)

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        current_batch_tokens = 0
        for idx in self.sampler:
            sample_length = self.lengths[idx]
            # If adding this sample would exceed max_tokens, yield the current batch
            if (
                current_batch_tokens + sample_length > self.max_tokens * self.world_size
                and batch
            ):
                yield batch
                batch = []
                current_batch_tokens = 0

            # Add the sample to the current batch
            batch.append(idx)
            # plus extract one for the PAD token to separate samples
            current_batch_tokens += sample_length + 1

        # Yield the last batch if it's not empty and we're not dropping it
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """
        Estimates the number of batches that will be generated.

        This is an approximation since the exact number depends on the specific
        sequence lengths and their order.
        """
        # Calculate average sequence length
        avg_seq_length = sum(self.lengths) // len(self.lengths)

        # Estimate average number of sequences per batch
        seqs_per_batch = self.max_tokens * self.world_size // avg_seq_length

        # Estimate total number of batches
        if self.drop_last:
            # If dropping last incomplete batch
            return len(self.lengths) // seqs_per_batch
        else:
            # If keeping last incomplete batch, ensure at least 1 batch
            return max(1, len(self.lengths) // seqs_per_batch)
