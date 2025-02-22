import os
from collections import defaultdict
from functools import partial
from typing import Dict, Tuple

import polars as pl
from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
from cehrbert.runners.runner_util import get_last_hf_checkpoint, load_parquet_as_dataset
from transformers.utils import logging
from trl import GRPOConfig, GRPOTrainer

from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.omop.vocab_utils import generate_concept_maps
from cehrgpt.rl_runners.grpo.compute_patient_sequence_co_occurrence import (
    is_co_occurrence_folder,
    temporal_co_occurrence_stats_name,
)
from cehrgpt.rl_runners.grpo.compute_patient_sequence_concept_prevalence import (
    concept_prevalence_stats_name,
)
from cehrgpt.rl_runners.grpo.compute_patient_sequence_length_stats import (
    patient_sequence_length_stats_name,
)
from cehrgpt.rl_runners.grpo.rewards import (
    DemographicGroup,
    reward_co_occurrence_information_content,
    reward_concept_information_content,
    reward_length,
    reward_valid_sequences,
)
from cehrgpt.runners.gpt_runner_util import parse_dynamic_arguments
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTGRPOArguments

logger = logging.get_logger("transformers")


def create_concept_prevalence(
    concept_prevalence_stats: pl.DataFrame,
) -> Dict[DemographicGroup, Dict[str, float]]:
    result_dict = defaultdict(lambda: defaultdict(float))
    for row in concept_prevalence_stats.to_dicts():
        demographic_group = DemographicGroup(
            row["age_group"], row["race"], row["gender"]
        )
        result_dict[demographic_group][str(row["concept_id"])] = row["prob"]
    return result_dict


def create_co_occurrence_matrix(
    matrix: pl.DataFrame,
    threshold: int = 20,
) -> Dict[DemographicGroup, Dict[Tuple[str, str], float]]:
    result = (
        matrix.filter(pl.col("count") >= threshold)
        .group_by(["age_group", "race", "gender"])
        .agg(
            [
                pl.col("concept_id_1").alias("concept_id_1_list"),
                pl.col("concept_id_2").alias("concept_id_2_list"),
                pl.col("prob").alias("prob_list"),
            ]
        )
    )

    # Transform DataFrame to the desired dictionary
    result_dict = {}
    for row in result.to_dicts():
        demographic_group = DemographicGroup(
            row["age_group"], row["race"], row["gender"]
        )
        # Prepare the inner dictionary
        inner_dict = {}
        for concept_id_1, concept_id_2, prob in zip(
            row["concept_id_1_list"], row["concept_id_2_list"], row["prob_list"]
        ):
            inner_dict[(str(concept_id_1), str(concept_id_2))] = prob
        result_dict[demographic_group] = inner_dict
    return result_dict


def create_length_stats(patient_seq_length_stats):
    length_stats = {}
    for row in patient_seq_length_stats.to_dicts():
        demographic_group = DemographicGroup(
            row["age_group"], row["race"], row["gender"]
        )
        length_stats[demographic_group] = {
            "log_mean": row["log_mean"],
            "log_std": row["log_std"],
        }
    return length_stats


def main():
    cehrgpt_grpo_args, model_args, training_args = parse_dynamic_arguments(
        (CehrGPTGRPOArguments, ModelArguments, GRPOConfig)
    )

    dataset = load_parquet_as_dataset(cehrgpt_grpo_args.demographics_prompt_dir)
    dataset = dataset.map(
        lambda batch: {
            "prompt": [concept_ids[:4] for concept_ids in batch["concept_ids"]]
        },
        batch_size=1024,
        batched=True,
        remove_columns=dataset.column_names,
    )
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    cehrgpt_model = CEHRGPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    co_occurrence_list = []
    co_occurrence_dir = str(
        os.path.join(
            cehrgpt_grpo_args.aggregate_data_dir, temporal_co_occurrence_stats_name
        )
    )
    for co_occurrence_folder in [
        f.path for f in os.scandir(co_occurrence_dir) if f.is_dir()
    ]:
        folder_split_parts = os.path.basename(co_occurrence_folder).split("_")
        if is_co_occurrence_folder(co_occurrence_folder):
            time_window_start = folder_split_parts[-2]
            time_window_end = folder_split_parts[-1]
            logger.info(
                "Load co-occurrence using a time window that starts at %s and ends at %s",
                time_window_start,
                time_window_end,
            )
            co_occurrence_matrix = pl.read_parquet(
                os.path.join(co_occurrence_folder, "*.parquet")
            )
            result_dict = create_co_occurrence_matrix(
                co_occurrence_matrix, threshold=10
            )
            if time_window_start.isnumeric():
                time_window_start = int(time_window_start)
            else:
                raise RuntimeError(
                    f"time_window_start must be numeric, but received {time_window_start}"
                )
            if time_window_end.isnumeric():
                time_window_end = int(time_window_end)
            else:
                time_window_end = 1_000_000_000
            co_occurrence_list.append((time_window_start, time_window_end, result_dict))

    reward_co_occurrence_with_time_window = partial(
        reward_co_occurrence_information_content,
        co_occurrence_matrices=co_occurrence_list,
    )
    reward_co_occurrence_with_time_window.__name__ = f"reward_co_occurrence"

    patient_seq_length_stats = pl.read_parquet(
        os.path.join(
            cehrgpt_grpo_args.aggregate_data_dir,
            patient_sequence_length_stats_name,
            "*.parquet",
        )
    )
    length_stats = create_length_stats(patient_seq_length_stats)
    reward_length_func = partial(reward_length, length_stats=length_stats)
    reward_length_func.__name__ = f"reward_length"

    concept = pl.read_parquet(
        os.path.join(cehrgpt_grpo_args.vocabulary_dir, "concept", "*.parquet")
    )
    _, concept_domain_map = generate_concept_maps(concept)
    reward_valid_sequence_func = partial(
        reward_valid_sequences, concept_domain_map=concept_domain_map
    )
    reward_valid_sequence_func.__name__ = f"reward_valid_sequence"

    concept_prevalence_stats = pl.read_parquet(
        os.path.join(
            cehrgpt_grpo_args.aggregate_data_dir,
            concept_prevalence_stats_name,
            "*.parquet",
        )
    )
    concept_prevalence = create_concept_prevalence(concept_prevalence_stats)
    reward_concept_information_content_func = partial(
        reward_concept_information_content, concept_prevalence=concept_prevalence
    )
    reward_concept_information_content_func.__name__ = (
        f"reward_concept_information_content"
    )

    # Detecting last checkpoint.
    checkpoint = get_last_hf_checkpoint(training_args)
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    logger.info("The existing checkpoint is %s", checkpoint)
    trainer = GRPOTrainer(
        model=cehrgpt_model,
        processing_class=cehrgpt_tokenizer,
        reward_funcs=[
            reward_valid_sequence_func,
            reward_length_func,
            reward_co_occurrence_with_time_window,
            reward_concept_information_content_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.generation_config.return_dict_in_generate = False
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
