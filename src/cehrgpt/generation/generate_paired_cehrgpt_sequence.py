import datetime
import os
import random
import uuid

import pandas as pd
import torch
from cehrbert.runners.runner_util import load_parquet_as_dataset
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.cehrgpt_args import create_inference_base_arg_parser
from cehrgpt.generation.generate_batch_hf_gpt_sequence import (
    generate_single_batch,
    normalize_value,
)
from cehrgpt.gpt_utils import get_cehrgpt_output_folder
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

LOG = logging.get_logger("transformers")


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_folder,
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
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    folder_name = get_cehrgpt_output_folder(args, cehrgpt_tokenizer)
    output_folder_name = os.path.join(
        args.output_folder, folder_name, "generated_sequences"
    )

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    LOG.info(f"Loading tokenizer at {args.model_folder}")
    LOG.info(f"Loading model at {args.model_folder}")
    LOG.info(f"Write sequences to {output_folder_name}")
    LOG.info(f"Context window {args.context_window}")
    LOG.info(f"Temperature {args.temperature}")
    LOG.info(f"Repetition Penalty {args.repetition_penalty}")
    LOG.info(f"Sampling Strategy {args.sampling_strategy}")
    LOG.info(f"Num beam {args.num_beams}")
    LOG.info(f"Num beam groups {args.num_beam_groups}")
    LOG.info(f"Epsilon cutoff {args.epsilon_cutoff}")
    LOG.info(f"Top P {args.top_p}")
    LOG.info(f"Top K {args.top_k}")
    LOG.info(f"Loading sequence_data_path at {args.sequence_data_path}")

    data = load_parquet_as_dataset(args.sequence_data_path)
    total_rows = len(data)
    float(args.batch_size) / total_rows
    num_of_batches = args.num_of_patients // args.batch_size + 1
    sequence_to_flush = []
    for i in range(num_of_batches):
        LOG.info(f"{datetime.datetime.now()}: Batch {i} started")
        random_indices = random.sample(range(total_rows), k=args.batch_size)
        sample_data = data.select(random_indices)
        prompts = []
        chosen_responses = []
        cutoff_frac = random.uniform(0, args.cutoff_frac_max)
        for row in sample_data:
            seq_len = len(row["concept_ids"])
            prompt_len = max(4, int(seq_len * cutoff_frac))
            prompts.append(cehrgpt_tokenizer.encode(row["concept_ids"][:prompt_len]))
            chosen_responses.append(
                {
                    "person_id": row["person_id"],
                    "chosen_concept_ids": row["concept_ids"],
                    "chosen_concept_values": row["concept_values"],
                    "chosen_concept_value_masks": row["concept_value_masks"],
                    "chosen_units": row["units"],
                    "prompt_length": prompt_len,
                }
            )

        batch_sequences = generate_single_batch(
            cehrgpt_model,
            cehrgpt_tokenizer,
            prompts=prompts,
            max_new_tokens=args.context_window,
            mini_num_of_concepts=args.min_num_of_concepts,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            epsilon_cutoff=args.epsilon_cutoff,
            device=device,
        )

        # Clear the cache
        torch.cuda.empty_cache()

        for seq, value_indicator, value, chosen_response in zip(
            batch_sequences["sequences"],
            batch_sequences["value_indicators"],
            batch_sequences["values"],
            chosen_responses,
        ):
            output = {"rejected_concept_ids": seq}
            normalized_values, units = normalize_value(
                seq, value_indicator, value, cehrgpt_tokenizer
            )
            if normalized_values is not None:
                output["rejected_concept_values"] = normalized_values
            if value_indicator is not None:
                output["rejected_concept_value_masks"] = value_indicator
            if units is not None:
                output["rejected_units"] = units
            output.update(chosen_response)
            sequence_to_flush.append(output)

        if len(sequence_to_flush) >= args.buffer_size:
            LOG.info(f"{datetime.datetime.now()}: Flushing to the Disk at Batch {i}")
            pd.DataFrame(
                sequence_to_flush,
                columns=[
                    "person_id",
                    "chosen_concept_ids",
                    "chosen_concept_values",
                    "chosen_concept_value_masks",
                    "chosen_units",
                    "rejected_concept_ids",
                    "rejected_concept_values",
                    "rejected_concept_value_masks",
                    "rejected_units",
                ],
            ).to_parquet(os.path.join(output_folder_name, f"{uuid.uuid4()}.parquet"))
            sequence_to_flush.clear()

    if len(sequence_to_flush) > 0:
        LOG.info(f"{datetime.datetime.now()}: Flushing to the Disk at Final Batch")
        pd.DataFrame(
            sequence_to_flush,
            columns=[
                "person_id",
                "chosen_concept_ids",
                "chosen_concept_values",
                "chosen_concept_value_masks",
                "chosen_units",
                "rejected_concept_ids",
                "rejected_concept_values",
                "rejected_concept_value_masks",
                "rejected_units",
            ],
        ).to_parquet(os.path.join(output_folder_name, f"{uuid.uuid4()}-last.parquet"))


def create_arg_parser():
    base_arg_parser = create_inference_base_arg_parser(
        description="Arguments for generating paired patient sequences"
    )
    base_arg_parser.add_argument(
        "--num_of_patients",
        dest="num_of_patients",
        action="store",
        type=int,
        help="The number of patients that will be generated",
        required=True,
    )
    base_arg_parser.add_argument(
        "--sequence_data_path",
        dest="sequence_data_path",
        action="store",
        help="The path for your sequence data",
        required=True,
    )
    base_arg_parser.add_argument(
        "--cutoff_frac_max",
        dest="cutoff_frac_max",
        action="store",
        type=float,
        help="The max fraction of the patient sequences that will be used for prompting",
        required=False,
        default=0.5,
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
