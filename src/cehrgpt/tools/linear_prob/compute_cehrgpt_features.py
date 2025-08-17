import datetime
import glob
import os
import shutil
import uuid
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.distributed as dist
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import concatenate_datasets, load_from_disk
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import (
    batched_powerSGD_hook,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import is_main_process
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)
from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2Model,
    extract_features_from_packed_sequence,
)
from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import (
    extract_cohort_sequences,
    prepare_finetune_dataset,
)
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import tokenizer_exists

LOG = logging.get_logger("transformers")


def get_torch_dtype(torch_dtype: Optional[str] = None) -> Union[torch.dtype, str]:
    if torch_dtype and hasattr(torch, torch_dtype):
        return getattr(torch, torch_dtype)
    return torch.float32


def extract_averaged_embeddings_from_packed_sequence(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    ve_token_indicators: torch.BoolTensor,
) -> torch.Tensor:
    """
    Args:

        hidden_states: (batch_size=1, seq_len, hidden_dim) tensor
        attention_mask: (batch_size=1, seq_len) tensor, where 0 indicates padding
        ve_token_indicators: (batch_size=1, seq_len) bool tensor, True if token is VE token
    Returns:
        (num_samples, hidden_dim) tensor: averaged embeddings over VE tokens for each sample
    """
    # Step 1: Create segment IDs
    mask = attention_mask[0]  # (seq_len,)
    segment_ids = (mask == 0).cumsum(dim=0) + 1  # start segment IDs from 1
    segment_ids = (segment_ids * mask).to(torch.int32)  # set PAD positions back to 0

    # Step 2: Only keep tokens that are both valid and VE tokens
    valid = (segment_ids > 0) & (ve_token_indicators[0])
    valid_embeddings = hidden_states[0, valid].to(
        torch.float32
    )  # (num_valid_ve_tokens, hidden_dim)
    valid_segments = segment_ids[valid]  # (num_valid_ve_tokens,)

    # Step 3: Group by segment id and average
    num_segments = int(segment_ids.max().item())

    sample_embeddings = torch.zeros(
        num_segments, hidden_states.size(-1), device=hidden_states.device
    )
    counts = torch.zeros(num_segments, device=hidden_states.device)

    sample_embeddings.index_add_(0, valid_segments - 1, valid_embeddings)
    counts.index_add_(
        0, valid_segments - 1, torch.ones_like(valid_segments, dtype=counts.dtype)
    )

    # Avoid divide-by-zero (if some segments have no VE tokens, set their embeddings to zero)
    counts = counts.masked_fill(counts == 0, 1.0)

    sample_embeddings = sample_embeddings / counts.unsqueeze(-1)

    return sample_embeddings


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    torch_dtype = get_torch_dtype(model_args.torch_dtype)
    cehrgpt_model = (
        CEHRGPT2Model.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=torch_dtype,
        )
        .eval()
        .to(device)
    )

    if LINEAR_PROB_TOKEN not in cehrgpt_tokenizer.get_vocab():
        cehrgpt_tokenizer.add_tokens(LINEAR_PROB_TOKEN)
        cehrgpt_model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)

    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )
    cache_file_collector = CacheFileCollector()
    processed_dataset = None
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        if cehrgpt_args.expand_tokenizer:
            if tokenizer_exists(training_args.output_dir):
                cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
                    training_args.output_dir
                )
            else:
                LOG.warning(
                    f"CehrGptTokenizer must exist in {training_args.output_dir} "
                    f"when the dataset has been processed and expand_tokenizer is set to True. "
                    f"Please delete the processed dataset at {prepared_ds_path}."
                )
                processed_dataset = None
                shutil.rmtree(prepared_ds_path)

    if processed_dataset is None:
        if is_main_process(training_args.local_rank):
            # If the full dataset has been tokenized, we don't want to tokenize the cohort containing
            # the subset of the data. We should slice out the portion of the tokenized sequences for each sample
            if cehrgpt_args.tokenized_full_dataset_path is not None:
                processed_dataset = extract_cohort_sequences(
                    data_args, cehrgpt_args, cache_file_collector
                )
            else:
                # Organize them into a single DatasetDict
                final_splits = prepare_finetune_dataset(
                    data_args, training_args, cehrgpt_args, cache_file_collector
                )
                # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
                if not data_args.streaming:
                    all_columns = final_splits["train"].column_names
                    if "visit_concept_ids" in all_columns:
                        final_splits = final_splits.remove_columns(
                            ["visit_concept_ids"]
                        )

                processed_dataset = create_cehrgpt_finetuning_dataset(
                    dataset=final_splits,
                    cehrgpt_tokenizer=cehrgpt_tokenizer,
                    data_args=data_args,
                    cache_file_collector=cache_file_collector,
                )
            if not data_args.streaming:
                processed_dataset.save_to_disk(prepared_ds_path)
                processed_dataset.cleanup_cache_files()

            # Remove all the cached files if processed_dataset.cleanup_cache_files() did not remove them already
            cache_file_collector.remove_cache_files()

        # After main-process-only operations, synchronize all processes to ensure consistency
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load the dataset from disk again to in torch distributed training
        processed_dataset = load_from_disk(str(prepared_ds_path))

    # Getting the existing features
    feature_folders = glob.glob(
        os.path.join(training_args.output_dir, "*", "features", "*.parquet")
    )
    if feature_folders:
        existing_features = pd.concat(
            [
                pd.read_parquet(f, columns=["subject_id", "prediction_time_posix"])
                for f in feature_folders
            ],
            ignore_index=True,
        )
        subject_prediction_tuples = set(
            existing_features.apply(
                lambda row: f"{int(row['subject_id'])}-{int(row['prediction_time_posix'])}",
                axis=1,
            ).tolist()
        )
        processed_dataset = processed_dataset.filter(
            lambda _batch: [
                f"{int(subject)}-{int(time)}" not in subject_prediction_tuples
                for subject, time in zip(_batch["person_id"], _batch["index_date"])
            ],
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            batched=True,
        )
        LOG.info(
            "The datasets after filtering (train: %s, validation: %s, test: %s)",
            len(processed_dataset["train"]),
            len(processed_dataset["validation"]),
            len(processed_dataset["test"]),
        )

    if (
        cehrgpt_model.config.max_position_embeddings
        < model_args.max_position_embeddings
    ):
        LOG.info(
            f"Increase model.config.max_position_embeddings to {model_args.max_position_embeddings}"
        )
        cehrgpt_model.config.max_position_embeddings = (
            model_args.max_position_embeddings
        )
        cehrgpt_model.resize_position_embeddings(model_args.max_position_embeddings)

    train_set = concatenate_datasets(
        [processed_dataset["train"], processed_dataset["validation"]]
    )

    if cehrgpt_args.sample_packing:
        per_device_eval_batch_size = 1
        data_collator_fn = partial(
            SamplePackingCehrGptDataCollator,
            cehrgpt_args.max_tokens_per_batch,
            cehrgpt_model.config.max_position_embeddings,
        )
        train_batch_sampler = SamplePackingBatchSampler(
            lengths=train_set["num_of_concepts"],
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=cehrgpt_model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
        test_batch_sampler = SamplePackingBatchSampler(
            lengths=processed_dataset["test"]["num_of_concepts"],
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=cehrgpt_model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
    else:
        data_collator_fn = CehrGptDataCollator
        train_batch_sampler = None
        test_batch_sampler = None
        per_device_eval_batch_size = training_args.per_device_eval_batch_size

    # We suppress the additional learning objectives in fine-tuning
    data_collator = data_collator_fn(
        tokenizer=cehrgpt_tokenizer,
        max_length=(
            cehrgpt_args.max_tokens_per_batch
            if cehrgpt_args.sample_packing
            else model_args.max_position_embeddings
        ),
        include_values=cehrgpt_model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
        add_linear_prob_token=True,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=train_batch_sampler,
    )

    test_dataloader = DataLoader(
        dataset=processed_dataset["test"],
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=test_batch_sampler,
    )

    if data_args.is_data_in_meds:
        demographics_dict = dict()
    else:
        # Loading demographics
        print("Loading demographics as a dictionary")
        demographics_df = pd.concat(
            [
                pd.read_parquet(
                    data_dir,
                    columns=[
                        "person_id",
                        "index_date",
                        "gender_concept_id",
                        "race_concept_id",
                    ],
                )
                for data_dir in [data_args.data_folder, data_args.test_data_folder]
            ]
        )

        demographics_df["index_date"] = (
            demographics_df["index_date"].dt.tz_localize("UTC")
            - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        ).dt.total_seconds()

        demographics_dict = {
            (row["person_id"], row["index_date"]): {
                "gender_concept_id": row["gender_concept_id"],
                "race_concept_id": row["race_concept_id"],
            }
            for _, row in demographics_df.iterrows()
        }

    data_loaders = [("train", train_loader), ("test", test_dataloader)]

    ve_token_id = cehrgpt_tokenizer.ve_token_id
    for split, data_loader in data_loaders:
        # Ensure prediction folder exists
        feature_output_folder = (
            Path(training_args.output_dir) / "features_with_label" / f"{split}_features"
        )
        feature_output_folder.mkdir(parents=True, exist_ok=True)

        LOG.info("Generating features for %s set at %s", split, feature_output_folder)

        with torch.no_grad():
            for index, batch in enumerate(
                tqdm(data_loader, desc="Generating features")
            ):
                prediction_time_ages = (
                    batch.pop("age_at_index").numpy().astype(float).squeeze()
                )
                if prediction_time_ages.ndim == 0:
                    prediction_time_ages = np.asarray([prediction_time_ages])

                person_ids = batch.pop("person_id").numpy().astype(int).squeeze()
                if person_ids.ndim == 0:
                    person_ids = np.asarray([person_ids])
                prediction_time_posix = batch.pop("index_date").numpy().squeeze()
                if prediction_time_posix.ndim == 0:
                    prediction_time_posix = np.asarray([prediction_time_posix])

                prediction_time = list(
                    map(
                        lambda posix_time: datetime.datetime.utcfromtimestamp(
                            posix_time
                        ).replace(tzinfo=None),
                        prediction_time_posix,
                    )
                )

                labels = (
                    batch.pop("classifier_label")
                    .float()
                    .cpu()
                    .numpy()
                    .astype(bool)
                    .squeeze()
                )
                if labels.ndim == 0:
                    labels = np.asarray([labels])

                # Right now the model does not support this column, we need to pop it
                if "epoch_times" in batch:
                    batch.pop("epoch_times")

                if "ages" in batch:
                    batch.pop("ages")

                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward pass
                cehrgpt_output = cehrgpt_model(
                    **batch, output_attentions=False, output_hidden_states=False
                )
                if cehrgpt_args.sample_packing:
                    if cehrgpt_args.average_over_sequence:
                        ve_token_indicators: torch.BoolTensor = (
                            batch["input_ids"] == ve_token_id
                        )
                        features = (
                            extract_averaged_embeddings_from_packed_sequence(
                                cehrgpt_output.last_hidden_state,
                                batch["attention_mask"],
                                ve_token_indicators,
                            )
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                        )
                    else:
                        features = (
                            extract_features_from_packed_sequence(
                                cehrgpt_output.last_hidden_state,
                                batch["attention_mask"],
                            )
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                            .squeeze(axis=0)
                        )
                else:
                    if cehrgpt_args.average_over_sequence:
                        features = torch.where(
                            batch["attention_mask"].unsqueeze(dim=-1).to(torch.bool),
                            cehrgpt_output.last_hidden_state,
                            0,
                        )
                        # Average across the sequence
                        features = features.mean(dim=1)
                    else:
                        last_end_token = any(
                            [
                                cehrgpt_tokenizer.end_token_id == input_id
                                for input_id in batch.pop("input_ids")
                                .cpu()
                                .numpy()
                                .squeeze()
                                .tolist()
                            ]
                        )
                        last_token_index = -2 if last_end_token else -1
                        LOG.debug(
                            "The last token is [END], we need to use the token index before that: %s",
                            last_token_index,
                        )
                        features = (
                            cehrgpt_output.last_hidden_state[..., last_token_index, :]
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                        )

                # Flatten features or handle them as a list of arrays (one array per row)
                features_list = [feature for feature in features]
                race_concept_ids = []
                gender_concept_ids = []
                for person_id, index_date in zip(person_ids, prediction_time):
                    key = (person_id, index_date.date())
                    if key in demographics_dict:
                        demographics = demographics_dict[key]
                        gender_concept_ids.append(demographics["gender_concept_id"])
                        race_concept_ids.append(demographics["race_concept_id"])
                    else:
                        gender_concept_ids.append(0)
                        race_concept_ids.append(0)

                features_pd = pd.DataFrame(
                    {
                        "subject_id": person_ids,
                        "prediction_time": prediction_time,
                        "prediction_time_posix": prediction_time_posix,
                        "boolean_value": labels,
                        "age_at_index": prediction_time_ages,
                    }
                )
                # Adding features as a separate column where each row contains a feature array
                features_pd["features"] = features_list
                features_pd["race_concept_id"] = race_concept_ids
                features_pd["gender_concept_id"] = gender_concept_ids
                features_pd.to_parquet(
                    feature_output_folder / f"{uuid.uuid4()}.parquet"
                )


if __name__ == "__main__":
    main()
