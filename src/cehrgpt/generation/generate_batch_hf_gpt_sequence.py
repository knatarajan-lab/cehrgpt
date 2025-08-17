import datetime
import os
import random
import uuid
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from cehrbert.runners.runner_util import load_parquet_as_dataset
from transformers import GenerationConfig
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.cehrgpt_args import create_inference_base_arg_parser
from cehrgpt.generation.omop_converter_batch import START_TOKEN_SIZE
from cehrgpt.gpt_utils import construct_age_sequence, get_cehrgpt_output_folder
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.special_tokens import END_TOKEN
from cehrgpt.models.tokenization_hf_cehrgpt import (
    NA,
    CehrGptTokenizer,
    is_valid_valid_bin,
)

LOG = logging.get_logger("transformers")


def normalize_value(
    seq: Sequence[str],
    values: Sequence[str],
    tokenizer: CehrGptTokenizer,
) -> Tuple[
    Sequence[str],
    Optional[Sequence[Optional[int]]],
    Optional[Sequence[Optional[float]]],
    Optional[Sequence[Optional[str]]],
    Optional[Sequence[str]],
]:
    concepts = []
    number_as_values = []
    concept_as_values = []
    is_numeric_types = []
    units = []
    for concept, value in zip(seq, values):
        if concept == END_TOKEN:
            break
        number_as_value = None
        concept_as_value = value if value and value.isnumeric() else None
        is_numeric_type = 0
        unit = NA
        # If concept is numeric, we expect the next token to be a value bin
        if is_valid_valid_bin(value):
            converted_value, unit = tokenizer.denormalize(concept, value)
            if isinstance(converted_value, float):
                number_as_value = converted_value
                is_numeric_type = 1

        concepts.append(concept)
        number_as_values.append(number_as_value)
        concept_as_values.append(concept_as_value)
        is_numeric_types.append(is_numeric_type)
        units.append(unit)

    return (
        concepts,
        is_numeric_types,
        number_as_values,
        concept_as_values,
        units,
    )


def generate_single_batch(
    model: CEHRGPT2LMHeadModel,
    cehrgpt_tokenizer: CehrGptTokenizer,
    prompts: torch.Tensor,
    max_length: int,
    ages: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    value_indicators: Optional[torch.Tensor] = None,
    max_new_tokens: Optional[int] = None,
    mini_num_of_concepts=1,
    top_p=0.95,
    top_k=50,
    temperature=1.0,
    repetition_penalty=1.0,
    num_beams=1,
    num_beam_groups=1,
    epsilon_cutoff=0.0,
    device: Any = "cpu",
) -> Dict[str, Any]:
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            min_length=mini_num_of_concepts,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            bos_token_id=model.generation_config.bos_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            renormalize_logits=True,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            epsilon_cutoff=epsilon_cutoff,
        )

        batched_prompts = prompts.to(device)
        if ages is not None:
            ages = ages.to(device)
        if values is not None:
            values = values.to(device)
        if value_indicators is not None:
            value_indicators = value_indicators.to(device)

        results = model.generate(
            inputs=batched_prompts,
            ages=ages,
            values=values,
            value_indicators=value_indicators,
            generation_config=generation_config,
            cehrgpt_tokenizer=cehrgpt_tokenizer,
        )

    sequences = [
        cehrgpt_tokenizer.decode(seq.cpu().numpy(), skip_special_tokens=False)
        for seq in results.sequences
    ]
    if results.sequence_vals is not None:
        values = [
            cehrgpt_tokenizer.decode_value(
                values.cpu().numpy(), skip_special_tokens=False
            )
            for values in results.sequence_vals
        ]
    else:
        values = np.zeros_like(sequences)
        values.fill(NA)
    if results.sequence_val_masks is not None:
        value_indicators = results.sequence_val_masks.cpu().numpy()
    else:
        value_indicators = np.zeros_like(sequences, dtype=np.int32).astype(bool)
    return {
        "sequences": sequences,
        "values": values,
        "value_indicators": value_indicators,
    }


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

    # Determine whether we will use the demographics with the long sequences
    max_seq_allowed = (
        cehrgpt_model.config.n_positions
        if args.drop_long_sequences
        else np.iinfo(np.int32).max
    )

    LOG.info(f"Loading tokenizer at {args.model_folder}")
    LOG.info(f"Loading model at {args.model_folder}")
    LOG.info(f"Write sequences to {output_folder_name}")
    LOG.info(f"Context window {args.context_window}")
    LOG.info(f"Max sequence allowed {max_seq_allowed}")
    LOG.info(f"Temperature {args.temperature}")
    LOG.info(f"Repetition Penalty {args.repetition_penalty}")
    LOG.info(f"Sampling Strategy {args.sampling_strategy}")
    LOG.info(f"Num beam {args.num_beams}")
    LOG.info(f"Num beam groups {args.num_beam_groups}")
    LOG.info(f"Epsilon cutoff {args.epsilon_cutoff}")
    LOG.info(f"Top P {args.top_p}")
    LOG.info(f"Top K {args.top_k}")
    LOG.info(f"Loading demographic_info at {args.demographic_data_path}")
    LOG.info(f"MEDS format: {args.meds_format}")

    dataset = load_parquet_as_dataset(args.demographic_data_path)
    total_rows = len(dataset)

    num_of_batches = args.num_of_patients // args.batch_size + 1
    sequence_to_flush = []
    current_person_id = 1
    prompt_size = 2 if args.meds_format else START_TOKEN_SIZE
    for i in range(num_of_batches):
        LOG.info(f"{datetime.datetime.now()}: Batch {i} started")

        # Randomly pick demographics from the existing population
        random_prompts = []
        random_prompt_ages = []
        iter = 0
        while len(random_prompts) < args.batch_size:
            for row in dataset.select(
                random.sample(range(total_rows), k=args.batch_size)
            ):
                if (
                    args.min_num_of_concepts
                    <= len(row["concept_ids"])
                    <= max_seq_allowed
                ):
                    prompt = row["concept_ids"][:prompt_size]
                    random_prompts.append(cehrgpt_tokenizer.encode(prompt))
                    random_prompt_ages.append(construct_age_sequence(prompt))
                iter += 1
                if not random_prompts and iter > 10:
                    raise RuntimeError(
                        f"The length of concept_ids in {args.demographic_data_path} does not qualify!"
                    )

        # Make sure the batch does not exceed batch_size
        batch_sequences = generate_single_batch(
            cehrgpt_model,
            cehrgpt_tokenizer,
            torch.tensor(random_prompts[: args.batch_size]),
            ages=torch.tensor(random_prompt_ages[: args.batch_size]),
            max_length=args.context_window,
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

        for concept_ids, value_indicators, values in zip(
            batch_sequences["sequences"],
            batch_sequences["value_indicators"],
            batch_sequences["values"],
        ):
            (
                concept_ids,
                is_numeric_types,
                number_as_values,
                concept_as_values,
                units,
            ) = normalize_value(concept_ids, values, cehrgpt_tokenizer)
            output = {"concept_ids": concept_ids, "person_id": current_person_id}
            if is_numeric_types is not None:
                output["is_numeric_types"] = is_numeric_types
            if number_as_values is not None:
                output["number_as_values"] = number_as_values
            if concept_as_values is not None:
                output["concept_as_values"] = concept_as_values
            if value_indicators is not None:
                output["concept_value_masks"] = value_indicators
            if units is not None:
                output["units"] = units

            sequence_to_flush.append(output)
            current_person_id += 1

        if len(sequence_to_flush) >= args.buffer_size:
            LOG.info(f"{datetime.datetime.now()}: Flushing to the Disk at Batch {i}")
            pd.DataFrame(
                sequence_to_flush,
                columns=[
                    "concept_ids",
                    "person_id",
                    "is_numeric_types",
                    "number_as_values",
                    "concept_as_values",
                    "concept_value_masks",
                    "units",
                ],
            ).to_parquet(os.path.join(output_folder_name, f"{uuid.uuid4()}.parquet"))
            sequence_to_flush.clear()

    if len(sequence_to_flush) > 0:
        LOG.info(f"{datetime.datetime.now()}: Flushing to the Disk at Final Batch")
        pd.DataFrame(
            sequence_to_flush,
            columns=[
                "concept_ids",
                "person_id",
                "is_numeric_types",
                "number_as_values",
                "concept_as_values",
                "concept_value_masks",
                "units",
            ],
        ).to_parquet(os.path.join(output_folder_name, f"{uuid.uuid4()}-last.parquet"))


def create_arg_parser():
    base_arg_parser = create_inference_base_arg_parser(
        description="Arguments for generating patient sequences"
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
        "--demographic_data_path",
        dest="demographic_data_path",
        action="store",
        help="The path for your concept_path",
        required=True,
    )
    base_arg_parser.add_argument(
        "--drop_long_sequences",
        dest="drop_long_sequences",
        action="store_true",
    )
    base_arg_parser.add_argument(
        "--meds_format",
        dest="meds_format",
        action="store_true",
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
