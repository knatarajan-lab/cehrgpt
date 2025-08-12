import os

import polars as pl
import torch
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.generation.generate_batch_hf_gpt_sequence import (
    generate_single_batch,
    normalize_value,
)
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import prepare_finetune_dataset
from cehrgpt.runners.gpt_runner_util import parse_runner_args

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
    max_new_tokens = cehrgpt_model.config.n_positions

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Organize them into a single DatasetDict
    final_splits = prepare_finetune_dataset(data_args, training_args, cehrgpt_args)
    batch_size = training_args.per_device_eval_batch_size
    dataset = final_splits["test"]

    for row_index, row in enumerate(dataset):
        current_person_id = row["person_id"]
        prediction_time = row["index_date"]
        prompts = [cehrgpt_tokenizer.encode(row["concept_ids"]) * batch_size]
        # Make sure the batch does not exceed batch_size
        batch_sequences = generate_single_batch(
            cehrgpt_model,
            cehrgpt_tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            top_p=1.0,
            top_k=cehrgpt_tokenizer.vocab_size,
            device=device,
        )
        # Clear the cache
        torch.cuda.empty_cache()

        trajectories = []
        for i, (concept_ids, value_indicators, values) in enumerate(
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

            trajectories.append(
                {
                    "subject_id": current_person_id,
                    "prediction_time": prediction_time,
                    "concept_ids": concept_ids,
                    "numeric_values": number_as_values,
                    "text_value": concept_as_values,
                    "units": units,
                    "trajectory_id": i + 1,
                }
            )

        trajectories = (
            pl.DataFrame(trajectories)
            .with_columns(
                pl.struct(
                    [
                        pl.col("concept_ids").alias("code"),
                        pl.col("numeric_values").alias("numeric_value"),
                        pl.col("text_value").alias("text_value"),
                        pl.col("units").alias("unit"),
                    ]
                ).alias("event")
            )
            .explode("event")
            .unnest("event")
            .select(
                "subject_id",
                "prediction_time",
                "trajectory_id",
                "code",
                "numeric_value",
                "text_value",
                "unit",
            )
        )
        trajectories.write_parquet(
            os.path.join(training_args.output_dir, f"{row_index}.parquet")
        )


if __name__ == "__main__":
    main()
