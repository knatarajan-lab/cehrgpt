import glob
import os
import shutil
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import DatasetDict, concatenate_datasets, load_from_disk
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
    CEHRGPT2LMHeadModel,
    extract_features_from_packed_sequence,
)
from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN, RANDOM_TOKEN
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


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    """Promote a 0-d numpy array to 1-d; leave higher-rank arrays unchanged."""
    return np.asarray([arr]) if arr.ndim == 0 else arr


def _create_feature_dataset(data_args, training_args, cehrgpt_args, tokenizer, cache_file_collector):
    """Process the raw data into a tokenized DatasetDict for feature extraction."""
    if cehrgpt_args.tokenized_full_dataset_path is not None:
        return extract_cohort_sequences(data_args, cehrgpt_args, tokenizer)

    final_splits = prepare_finetune_dataset(
        data_args, training_args, cehrgpt_args, cache_file_collector
    )
    # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
    if not data_args.streaming:
        if "visit_concept_ids" in final_splits["train"].column_names:
            final_splits = final_splits.remove_columns(["visit_concept_ids"])

    return create_cehrgpt_finetuning_dataset(
        dataset=final_splits,
        cehrgpt_tokenizer=tokenizer,
        data_args=data_args,
        cache_file_collector=cache_file_collector,
    )


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
        CEHRGPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=torch_dtype,
        )
        .eval()
        .to(device)
    )
    for additional_token in [LINEAR_PROB_TOKEN, RANDOM_TOKEN]:
        if additional_token not in cehrgpt_tokenizer.get_vocab():
            cehrgpt_tokenizer.add_tokens(additional_token)
        if cehrgpt_tokenizer.vocab_size > cehrgpt_model.config.vocab_size:
            cehrgpt_model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)

    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )
    cache_file_collector = CacheFileCollector()
    processed_dataset = None

    if cehrgpt_args.refresh_processed_dataset and prepared_ds_path.exists():
        LOG.info("Refreshing prepared dataset: removing cache at %s", prepared_ds_path)
        shutil.rmtree(prepared_ds_path)

    if any(prepared_ds_path.glob("*")):
        LOG.info("Loading prepared dataset from disk at %s...", prepared_ds_path)
        processed_dataset = load_from_disk(str(prepared_ds_path))
        if cehrgpt_args.expand_tokenizer:
            if tokenizer_exists(training_args.output_dir):
                cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
                    training_args.output_dir
                )
            else:
                LOG.warning(
                    "CehrGptTokenizer must exist in %s when the dataset has been processed "
                    "and expand_tokenizer is set to True. "
                    "Please delete the processed dataset at %s.",
                    training_args.output_dir,
                    prepared_ds_path,
                )
                processed_dataset = None
                shutil.rmtree(prepared_ds_path)

    if processed_dataset is None:
        if is_main_process(training_args.local_rank):
            processed_dataset = _create_feature_dataset(
                data_args, training_args, cehrgpt_args, cehrgpt_tokenizer, cache_file_collector
            )
            if not data_args.streaming:
                # Drop any splits that ended up empty — save_to_disk raises
                # SchemaInferenceError when asked to write a zero-example split.
                empty_splits = [k for k, v in processed_dataset.items() if len(v) == 0]
                if empty_splits:
                    LOG.warning("Dropping empty splits before saving: %s", empty_splits)
                    processed_dataset = DatasetDict(
                        {k: v for k, v in processed_dataset.items() if len(v) > 0}
                    )
                processed_dataset.save_to_disk(prepared_ds_path)
                stats = processed_dataset.cleanup_cache_files()
                LOG.info(
                    "Clean up the cached files for the cehrgpt feature dataset: %s",
                    stats,
                )
            cache_file_collector.remove_cache_files()

        # After main-process-only operations, synchronize all processes to ensure consistency
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load the dataset from disk again in torch distributed training
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
            "Increase model.config.max_position_embeddings to %s",
            model_args.max_position_embeddings,
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
        add_linear_prob_token=cehrgpt_args.add_random_token,
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
            - datetime(1970, 1, 1, tzinfo=timezone.utc)
        ).dt.total_seconds()

        demographics_dict = {
            (row["person_id"], row["index_date"]): {
                "gender_concept_id": row["gender_concept_id"],
                "race_concept_id": row["race_concept_id"],
            }
            for _, row in demographics_df.iterrows()
        }

    data_loaders = [("train", train_loader), ("test", test_dataloader)]

    for split, data_loader in data_loaders:
        feature_output_folder = (
            Path(training_args.output_dir) / "features_with_label" / f"{split}_features"
        )
        feature_output_folder.mkdir(parents=True, exist_ok=True)

        LOG.info("Generating features for %s set at %s", split, feature_output_folder)

        with torch.no_grad():
            for index, batch in enumerate(
                tqdm(data_loader, desc="Generating features")
            ):
                prediction_time_ages = _ensure_1d(
                    batch.pop("age_at_index").numpy().astype(float).squeeze()
                )
                person_ids = _ensure_1d(
                    batch.pop("person_id").numpy().astype(int).squeeze()
                )
                prediction_time_posix = _ensure_1d(
                    batch.pop("index_date").numpy().squeeze()
                )
                prediction_time = [
                    datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)
                    for t in prediction_time_posix
                ]
                labels = _ensure_1d(
                    batch.pop("classifier_label")
                    .float()
                    .cpu()
                    .numpy()
                    .astype(bool)
                    .squeeze()
                )

                # Right now the model does not support this column, we need to pop it
                if "epoch_times" in batch:
                    batch.pop("epoch_times")

                if "ages" in batch:
                    batch.pop("ages")

                batch = {k: v.to(device) for k, v in batch.items()}
                cehrgpt_output = cehrgpt_model(
                    **batch, output_attentions=False, output_hidden_states=True
                )
                # When the model was trained without MOTOR (include_motor_time_to_event=False),
                # linear_prob_hidden_states is None.  Fall back to the last transformer
                # hidden state in that case.
                linear_prob_hs = cehrgpt_output.linear_prob_hidden_states
                if linear_prob_hs is None:
                    linear_prob_hs = cehrgpt_output.hidden_states[-1]

                if cehrgpt_args.sample_packing:
                    features = (
                        extract_features_from_packed_sequence(
                            linear_prob_hs,
                            batch["attention_mask"],
                        )
                        .cpu()
                        .float()
                        .detach()
                        .numpy()
                        .squeeze(axis=0)
                    )
                    if cehrgpt_args.combine_global_local_features:
                        last_features = (
                            extract_features_from_packed_sequence(
                                cehrgpt_output.hidden_states[-1],
                                batch["attention_mask"],
                            )
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                            .squeeze(axis=0)
                        )
                        features = np.concatenate([features, last_features], axis=-1)
                else:
                    features = (
                        linear_prob_hs[..., -1, :]
                        .cpu()
                        .float()
                        .detach()
                        .numpy()
                    )
                    if cehrgpt_args.combine_global_local_features:
                        last_features = (
                            cehrgpt_output.hidden_states[-1][..., -1, :]
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                        )
                        features = np.concatenate([features, last_features], axis=-1)

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
                features_pd["features"] = features_list
                features_pd["race_concept_id"] = race_concept_ids
                features_pd["gender_concept_id"] = gender_concept_ids
                features_pd.to_parquet(
                    feature_output_folder / f"{uuid.uuid4()}.parquet"
                )


if __name__ == "__main__":
    main()
