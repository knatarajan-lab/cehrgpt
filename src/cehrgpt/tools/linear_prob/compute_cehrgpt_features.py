import glob
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.hf_cehrgpt import CEHRGPT2Model
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import tokenizer_exists

LOG = logging.get_logger("transformers")


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    cehrgpt_model = (
        CEHRGPT2Model.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=(
                torch.bfloat16 if is_flash_attn_2_available() else torch.float32
            ),
        )
        .eval()
        .to(device)
    )
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
        # Organize them into a single DatasetDict
        final_splits = prepare_finetune_dataset(
            data_args, training_args, cehrgpt_args, cache_file_collector
        )
        if cehrgpt_args.expand_tokenizer:
            new_tokenizer_path = os.path.expanduser(training_args.output_dir)
            if tokenizer_exists(new_tokenizer_path):
                cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(new_tokenizer_path)
            else:
                cehrgpt_tokenizer = CehrGptTokenizer.expand_trained_tokenizer(
                    cehrgpt_tokenizer=cehrgpt_tokenizer,
                    dataset=final_splits["train"],
                    data_args=data_args,
                    concept_name_mapping={},
                )
                cehrgpt_tokenizer.save_pretrained(
                    os.path.expanduser(training_args.output_dir)
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

    # Getting the existing features
    feature_folders = glob.glob(
        os.path.join(training_args.output_dir, "*", "features", "*.parquet")
    )
    if feature_folders:
        existing_features = pd.concat(
            [
                pd.read_parquet(f, columns=["subject_id", "prediction_time"])
                for f in feature_folders
            ],
            ignore_index=True,
        )
        subject_prediction_tuples = list(
            existing_features.apply(
                lambda row: (row["subject_id"], row["prediction_time"]), axis=1
            )
        )
        processed_dataset = processed_dataset.filter(
            lambda _batch: [
                (subject, datetime.fromtimestamp(time)) not in subject_prediction_tuples
                for subject, time in zip(_batch["person_id"], _batch["index_date"])
            ],
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            batched=True,
        )

    LOG.info(f"cehrgpt_model.config.vocab_size: {cehrgpt_model.config.vocab_size}")
    LOG.info(f"cehrgpt_tokenizer.vocab_size: {cehrgpt_tokenizer.vocab_size}")
    if cehrgpt_model.config.vocab_size < cehrgpt_tokenizer.vocab_size:
        cehrgpt_model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)
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

    data_collator = CehrGptDataCollator(
        tokenizer=cehrgpt_tokenizer,
        max_length=(
            cehrgpt_model.config.max_position_embeddings - 1
            if cehrgpt_model.config.causal_sfm
            else cehrgpt_model.config.max_position_embeddings
        ),
        include_values=model_args.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
    )

    train_loader = DataLoader(
        dataset=concatenate_datasets(
            [processed_dataset["train"], processed_dataset["validation"]]
        ),
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
    )

    test_dataloader = DataLoader(
        dataset=processed_dataset["test"],
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
    )

    # Loading demographics
    print("Loading demographics as a dictionary")
    demographics_df = pd.read_parquet(
        data_args.data_folder,
        columns=["person_id", "index_date", "gender_concept_id", "race_concept_id"],
    )
    demographics_df["index_date"] = demographics_df.index_date.dt.date
    demographics_dict = {
        (row["person_id"], row["index_date"]): {
            "gender_concept_id": row["gender_concept_id"],
            "race_concept_id": row["race_concept_id"],
        }
        for _, row in demographics_df.iterrows()
    }

    data_loaders = [("train", train_loader), ("test", test_dataloader)]

    for split, data_loader in data_loaders:

        # Ensure prediction folder exists
        feature_output_folder = Path(training_args.output_dir) / split / "features"
        feature_output_folder.mkdir(parents=True, exist_ok=True)

        LOG.info("Generating features for %s set at %s", split, feature_output_folder)

        with torch.no_grad():
            for index, batch in enumerate(
                tqdm(data_loader, desc="Generating features")
            ):
                prediction_time_ages = (
                    batch.pop("age_at_index").numpy().squeeze().astype(float)
                )
                person_ids = list(batch.pop("person_id").numpy().squeeze().astype(int))
                index_dates = list(
                    map(
                        datetime.fromtimestamp,
                        batch.pop("index_date").numpy().squeeze(axis=-1).tolist(),
                    )
                    if "index_date" in batch
                    else None
                )
                labels = (
                    batch.pop("classifier_label")
                    .float()
                    .cpu()
                    .numpy()
                    .squeeze()
                    .astype(bool)
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward pass
                cehrgpt_output = cehrgpt_model(
                    **batch, output_attentions=False, output_hidden_states=False
                )
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
                for person_id, index_date in zip(person_ids, index_dates):
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
                        "prediction_time": index_dates,
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
