import datetime
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import load_from_disk
from meds import held_out_split, train_split, tuning_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import is_main_process
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.generation.generate_batch_hf_gpt_sequence import (
    generate_single_batch,
    normalize_value,
)
from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    extract_time_interval_in_hours,
    is_att_token,
    is_inpatient_hour_token,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import (
    extract_cohort_sequences,
    prepare_finetune_dataset,
)
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import tokenizer_exists

LOG = logging.get_logger("transformers")


def map_data_split_name(split: str) -> str:
    if split == "train":
        return train_split
    elif split == "validation":
        return tuning_split
    elif split == "test":
        return held_out_split
    raise ValueError(f"Unknown split: {split}")


def seed_all(seed: int = 42):
    """Set seed for Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs

    # For reproducibility in dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_trajectories_per_batch(
    batch: Dict[str, Any],
    cehrgpt_tokenizer: CehrGptTokenizer,
    cehrgpt_model: CEHRGPT2LMHeadModel,
    device,
    data_output_path: Path,
    max_length: int,
):
    subject_ids = batch["person_id"].squeeze().detach().cpu().tolist()
    prediction_times = batch["index_date"].squeeze().detach().cpu().tolist()
    batched_epoch_times = batch["epoch_times"].detach().cpu().tolist()
    batched_input_ids = batch["input_ids"]
    batched_ages = batch["ages"]
    batched_value_indicators = batch["value_indicators"]
    batched_values = batch["values"]
    # Make sure the batch does not exceed batch_size
    batch_sequences = generate_single_batch(
        cehrgpt_model,
        cehrgpt_tokenizer,
        batched_input_ids,
        ages=batched_ages,
        values=batched_values,
        value_indicators=batched_value_indicators,
        max_length=max_length,
        top_p=1.0,
        top_k=cehrgpt_tokenizer.vocab_size,
        device=device,
    )
    # Clear the cache
    torch.cuda.empty_cache()

    trajectories = []
    for sample_i, (concept_ids, value_indicators, values) in enumerate(
        zip(
            batch_sequences["sequences"],
            batch_sequences["value_indicators"],
            batch_sequences["values"],
        )
    ):
        (
            concept_ids,
            is_numeric_types,
            number_as_values,
            concept_as_values,
            units,
        ) = normalize_value(concept_ids, values, cehrgpt_tokenizer)

        epoch_times = batched_epoch_times[sample_i]
        input_length = len(epoch_times)
        # Getting the last observed event time from the token before the prediction time
        window_last_observed = epoch_times[input_length - 1]
        current_cursor = epoch_times[-1]
        generated_epoch_times = []
        valid_indices = []

        for i in range(input_length, len(concept_ids)):
            concept_id = concept_ids[i]
            # We use the left padding strategy in the data collator
            if concept_id in [cehrgpt_tokenizer.pad_token, cehrgpt_tokenizer.end_token]:
                continue
            # We need to construct the time stamp
            if is_att_token(concept_id):
                current_cursor += extract_time_interval_in_days(concept_id) * 24 * 3600
            elif is_inpatient_hour_token(concept_id):
                current_cursor += extract_time_interval_in_hours(concept_id) * 3600
            elif is_visit_start(concept_id) or is_visit_end(concept_id):
                continue
            else:
                valid_indices.append(i)
                generated_epoch_times.append(
                    datetime.datetime.utcfromtimestamp(current_cursor).replace(
                        tzinfo=None
                    )
                )

        trajectories.append(
            {
                "subject_id": subject_ids[sample_i],
                "prediction_time": datetime.datetime.utcfromtimestamp(
                    prediction_times[sample_i]
                ).replace(tzinfo=None),
                "window_last_observed_time": datetime.datetime.utcfromtimestamp(
                    window_last_observed
                ).replace(tzinfo=None),
                "times": generated_epoch_times,
                "concept_ids": np.asarray(concept_ids)[valid_indices].tolist(),
                "numeric_values": np.asarray(number_as_values)[valid_indices].tolist(),
                "text_value": np.asarray(concept_as_values)[valid_indices].tolist(),
                "units": np.asarray(units)[valid_indices].tolist(),
            }
        )

    trajectories = (
        pl.DataFrame(trajectories)
        .explode(["times", "concept_ids", "numeric_values", "text_value", "units"])
        .rename(
            {
                "times": "time",
                "concept_ids": "code",
                "numeric_values": "numeric_value",
                "units": "unit",
            }
        )
        .select(
            "subject_id",
            "prediction_time",
            "window_last_observed_time",
            "time",
            "code",
            "numeric_value",
            "text_value",
            "unit",
        )
    )
    trajectories.write_parquet(data_output_path)


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
        CEHRGPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
        )
        .eval()
        .to(device)
    )
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )

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

    if processed_dataset is None and is_main_process(training_args.local_rank):
        # If the full dataset has been tokenized, we don't want to tokenize the cohort containing
        # the subset of the data. We should slice out the portion of the tokenized sequences for each sample
        if cehrgpt_args.tokenized_full_dataset_path is not None:
            processed_dataset = extract_cohort_sequences(data_args, cehrgpt_args)
        else:
            # Organize them into a single DatasetDict
            final_splits = prepare_finetune_dataset(
                data_args, training_args, cehrgpt_args
            )
            # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
            if not data_args.streaming:
                all_columns = final_splits["train"].column_names
                if "visit_concept_ids" in all_columns:
                    final_splits = final_splits.remove_columns(["visit_concept_ids"])

            processed_dataset = create_cehrgpt_finetuning_dataset(
                dataset=final_splits,
                cehrgpt_tokenizer=cehrgpt_tokenizer,
                data_args=data_args,
            )
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)
            processed_dataset.cleanup_cache_files()

    # After main-process-only operations, synchronize all processes to ensure consistency
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # We suppress the additional learning objectives in fine-tuning
    data_collator = CehrGptDataCollator(
        tokenizer=cehrgpt_tokenizer,
        max_length=cehrgpt_args.generation_input_length,
        include_values=cehrgpt_model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=False,
        add_linear_prob_token=False,
    )

    LOG.info(
        "Generating %s trajectories per sample",
        cehrgpt_args.num_of_trajectories_per_sample,
    )
    for sample_i in range(cehrgpt_args.num_of_trajectories_per_sample):
        for split, dataset in processed_dataset.items():
            meds_split = map_data_split_name(split)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=training_args.per_device_eval_batch_size,
                num_workers=training_args.dataloader_num_workers,
                collate_fn=data_collator,
                pin_memory=training_args.dataloader_pin_memory,
            )
            sample_output_dir = (
                Path(training_args.output_dir) / meds_split / f"{sample_i}"
            )
            sample_output_dir.mkdir(exist_ok=True, parents=True)
            for batch_i, batch in tqdm(
                enumerate(dataloader),
                desc=f"Generating Trajectories for split {meds_split} with trajectory {sample_i + 1}",
            ):
                output_parquet_file = sample_output_dir / f"{batch_i}.parquet"
                if output_parquet_file.exists():
                    LOG.info("%s already exists, skip...", output_parquet_file)
                    continue

                generate_trajectories_per_batch(
                    batch,
                    cehrgpt_tokenizer,
                    cehrgpt_model,
                    device,
                    sample_output_dir / f"{batch_i}.parquet",
                    cehrgpt_args.generation_max_new_tokens
                    + cehrgpt_args.generation_input_length,
                )


if __name__ == "__main__":
    # ✅ Call first thing inside main()
    seed_all(42)
    main()
