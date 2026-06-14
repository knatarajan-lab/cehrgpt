"""
Standalone script to tokenize a raw CEHR-GPT dataset using an existing tokenizer.

Tokenizes once, saves to disk, and the output can be reused for repeated training runs
by passing the output path as --data_folder (the runner detects dataset_dict.json).

Usage:
    python scripts/tokenize_dataset.py \
        --tokenizer_name_or_path /path/to/tokenizer \
        --data_folder /path/to/raw/parquet \
        --output_dir /path/to/save/tokenized \
        --validation_split_percentage 0.05 \
        --preprocessing_num_workers 8 \
        --preprocessing_batch_size 1000
"""

import argparse
import os
from pathlib import Path

from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert.runners.runner_util import load_parquet_as_dataset
from datasets import DatasetDict
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_pretraining_dataset
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

LOG = logging.get_logger("transformers")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize a raw CEHR-GPT dataset using an existing tokenizer and save to disk."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="Path to an existing CehrGptTokenizer directory.",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to the folder containing raw parquet patient sequence files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where the tokenized HuggingFace DatasetDict will be saved.",
    )
    parser.add_argument(
        "--test_data_folder",
        type=str,
        default=None,
        help="Path to a separate folder of parquet files to use as the test split.",
    )
    parser.add_argument(
        "--tokenized_dataset_name",
        type=str,
        default=None,
        help="Sub-directory name under --output_dir where the dataset is saved "
             "(e.g. 'my_dataset'). Mirrors the --tokenized_dataset_name used by the training runner.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=None,
        help="Fraction of data to use as validation set (e.g. 0.05 for 5%%).",
    )
    parser.add_argument(
        "--validation_split_num",
        type=int,
        default=None,
        help="Number of examples to use as validation set (used when --streaming is enabled).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for dataset mapping.",
    )
    parser.add_argument(
        "--preprocessing_batch_size",
        type=int,
        default=1000,
        help="Batch size for dataset mapping.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Load data in streaming mode (cannot be saved to disk with this flag).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer_path = os.path.expanduser(args.tokenizer_name_or_path)
    data_folder = os.path.expanduser(args.data_folder)
    output_dir = os.path.expanduser(args.output_dir)
    if args.tokenized_dataset_name:
        save_dir = os.path.join(output_dir, args.tokenized_dataset_name)
    else:
        save_dir = output_dir

    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found at: {data_folder}")

    if os.path.exists(os.path.join(save_dir, "dataset_dict.json")):
        LOG.warning(
            "Tokenized dataset already exists at %s — skipping to avoid overwriting. "
            "Delete the directory first if you want to re-tokenize.",
            save_dir,
        )
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    LOG.info("Loading tokenizer from %s", tokenizer_path)
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_path)

    LOG.info("Loading raw dataset from %s", data_folder)
    dataset = load_parquet_as_dataset(
        data_folder,
        split="train",
        streaming=args.streaming,
    )

    # Split into train/validation
    if args.streaming:
        if not args.validation_split_num:
            raise ValueError("--validation_split_num is required when --streaming is enabled.")
        dataset = dataset.shuffle(buffer_size=10_000, seed=args.seed)
        dataset = DatasetDict({
            "train": dataset.skip(args.validation_split_num),
            "validation": dataset.take(args.validation_split_num),
        })
    else:
        if args.validation_split_percentage is None:
            raise ValueError("--validation_split_percentage is required when streaming is disabled.")
        split = dataset.train_test_split(
            test_size=args.validation_split_percentage, seed=args.seed
        )
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Add a test split from a separate folder if provided
    if args.test_data_folder:
        test_data_folder = os.path.expanduser(args.test_data_folder)
        if not os.path.isdir(test_data_folder):
            raise FileNotFoundError(f"Test data folder not found at: {test_data_folder}")
        LOG.info("Loading test dataset from %s", test_data_folder)
        test_dataset = load_parquet_as_dataset(
            test_data_folder,
            split="train",
            streaming=args.streaming,
        )
        dataset["test"] = test_dataset

    # Build a minimal DataTrainingArguments for the tokenization pipeline
    data_args = DataTrainingArguments(
        data_folder=data_folder,
        dataset_prepared_path=output_dir,
        preprocessing_num_workers=args.preprocessing_num_workers,
        preprocessing_batch_size=args.preprocessing_batch_size,
        streaming=args.streaming,
    )

    split_sizes = {k: (len(v) if not args.streaming else -1) for k, v in dataset.items()}
    LOG.info("Tokenizing dataset %s", split_sizes)

    tokenized = create_cehrgpt_pretraining_dataset(
        dataset=dataset,
        cehrgpt_tokenizer=cehrgpt_tokenizer,
        data_args=data_args,
    )

    if args.streaming:
        LOG.warning(
            "Streaming mode is enabled — cannot save to disk. "
            "Run without --streaming to produce a reusable tokenized dataset."
        )
        return

    LOG.info("Saving tokenized dataset to %s", save_dir)
    tokenized.save_to_disk(save_dir)
    stats = tokenized.cleanup_cache_files()
    LOG.info("Cleaned up intermediate cache files: %s", stats)
    LOG.info("Done. Pass --data_folder %s to the training runner to reuse this dataset.", save_dir)


if __name__ == "__main__":
    main()
