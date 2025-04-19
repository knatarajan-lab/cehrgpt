from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import has_length
from transformers.utils import import_utils, logging

from cehrgpt.data.sample_packing_sampler import SamplePackingSampler

DEFAULT_MAX_TOKENS_PER_BATCH = 16384

LOG = logging.get_logger(__name__)


class SamplePackingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if "max_tokens_per_batch" in kwargs:
            self.max_tokens_per_batch = kwargs.pop("max_tokens_per_batch")
            LOG.info("max_tokens_per_batch: %s", self.max_tokens_per_batch)
        else:
            self.max_tokens_per_batch = DEFAULT_MAX_TOKENS_PER_BATCH
            LOG.info(
                "max_tokens_per_batch is not provided to SamplePackingTrainer and will default to %s",
                DEFAULT_MAX_TOKENS_PER_BATCH,
            )
        super().__init__(*args, **kwargs)

    def num_examples(self, dataloader: DataLoader) -> int:
        if has_length(dataloader):
            return len(dataloader)
        raise RuntimeError("DataLoader in SamplePackingTrainer must have length")

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training dataloader with our custom batch sampler."""
        train_dataset = self.train_dataset

        # Calculate lengths of all sequences in dataset
        lengths = [len(sample["input_ids"]) for sample in train_dataset]

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if import_utils.is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )
        # Create our custom batch sampler
        batch_sampler = SamplePackingSampler(
            lengths=lengths,
            max_tokens=self.max_tokens_per_batch,
            world_size=self.args.world_size,
            drop_last=self.args.dataloader_drop_last,
        )
        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": batch_sampler,
        }
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
