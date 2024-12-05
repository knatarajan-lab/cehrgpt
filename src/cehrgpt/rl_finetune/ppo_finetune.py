import datetime
import os
import pickle
import random
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from cehrbert.models.hf_models.tokenization_utils import agg_helper
from cehrbert.runners.runner_util import load_parquet_as_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.utils import is_flash_attn_2_available, logging
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

from cehrgpt.cehrgpt_args import create_inference_base_arg_parser
from cehrgpt.gpt_utils import get_cehrgpt_output_folder
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

LOG = logging.get_logger("transformers")


def extract_demographics_info(
    records: Dict[str, Any]
) -> Dict[Tuple[str, str, str, str], Dict[str, int]]:
    batched_concept_ids = records["concept_ids"]
    outputs = defaultdict(dict)
    for concept_ids in batched_concept_ids:
        start_year, start_age, gender, race = concept_ids[:4]
        existing_stats = outputs[(start_year, start_age, gender, race)]
        for concept_id, cnt in dict(Counter(concept_ids[4:])).items():
            if concept_id in existing_stats:
                existing_stats[concept_id] += cnt
            else:
                existing_stats[concept_id] = cnt
    return outputs


def generate_single_batch(
    model,
    tokenizer,
    batched_prompts,
    max_new_tokens=512,
    mini_num_of_concepts=1,
    top_p=0.95,
    top_k=50,
    temperature=1.0,
    repetition_penalty=1.0,
    num_beams=1,
    num_beam_groups=1,
    epsilon_cutoff=0.0,
) -> List[List[str]]:
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=repetition_penalty,
            max_length=max_new_tokens,
            min_length=mini_num_of_concepts,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            pad_token_id=tokenizer.pad_token_id,
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
        results = model.generate(
            inputs=batched_prompts, generation_config=generation_config
        )

    return [tokenizer.decode(seq.cpu().numpy()) for seq in results.sequences]


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = CEHRGPT2LMHeadModel.from_pretrained(
        args.model_folder,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else "eager"
        ),
        torch_dtype=(torch.bfloat16 if is_flash_attn_2_available() else torch.float32),
    )
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id
    model = AutoModelForCausalLMWithValueHead(cehrgpt_model).to(device)
    model.is_peft_model = False
    ref_model = create_reference_model(model).to(device)

    # create a ppo trainer
    ppo_trainer = PPOTrainer(
        config=PPOConfig(
            batch_size=args.batch_size, mini_batch_size=args.mini_batch_size
        ),
        model=model,
        ref_model=ref_model,
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
    )

    model_folder_name = os.path.join(
        args.output_folder, get_cehrgpt_output_folder(args, cehrgpt_tokenizer), "model"
    )

    if not os.path.exists(model_folder_name):
        os.makedirs(model_folder_name)

    LOG.info(f"Loading tokenizer at {args.model_folder}")
    LOG.info(f"Loading model at {args.model_folder}")
    LOG.info(f"Will save the fine-tuned model at {model_folder_name}")
    LOG.info(f"Context window {args.context_window}")
    LOG.info(f"Temperature {args.temperature}")
    LOG.info(f"Repetition Penalty {args.repetition_penalty}")
    LOG.info(f"Sampling Strategy {args.sampling_strategy}")
    LOG.info(f"Num beam {args.num_beams}")
    LOG.info(f"Num beam groups {args.num_beam_groups}")
    LOG.info(f"Epsilon cutoff {args.epsilon_cutoff}")
    LOG.info(f"Top P {args.top_p}")
    LOG.info(f"Top K {args.top_k}")
    LOG.info(f"Loading demographic_info at {args.demographic_data_path}")

    dataset = load_parquet_as_dataset(args.demographic_data_path)
    parts = dataset.filter(
        lambda batched: [
            num_of_concepts > 4 for num_of_concepts in batched["num_of_concepts"]
        ],
        batched=True,
    ).map(
        partial(agg_helper, map_func=extract_demographics_info),
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )
    prompts_and_concept_stats = defaultdict(dict)
    for stat in tqdm(parts, desc="Aggregating the concept counts"):
        fixed_stat = pickle.loads(stat["data"])
        for prompt, concept_stats in fixed_stat.items():
            for concept_id, count in concept_stats.items():
                if concept_id not in prompts_and_concept_stats[prompt]:
                    prompts_and_concept_stats[prompt][concept_id] = count
                else:
                    prompts_and_concept_stats[prompt][concept_id] += count

    for prompt, concept_stats in prompts_and_concept_stats.items():
        total_count = sum(concept_stats.values())
        for concept_id in concept_stats.keys():
            concept_stats[concept_id] = concept_stats[concept_id] / total_count

    prompts = list(prompts_and_concept_stats.keys())
    total = len(prompts)
    device = ppo_trainer.current_device
    num_of_batches = args.num_of_patients // args.batch_size + 1
    for i in range(num_of_batches):
        LOG.info(f"{datetime.datetime.now()}: Batch {i} started")
        random_index = random.randint(0, total - 1)
        expected_concept_dist = prompts_and_concept_stats[prompts[random_index]]
        batched_prompt = cehrgpt_tokenizer.encode(
            [prompts[random_index] for _ in range(args.batch_size)]
        ).to(device)
        batched_sequences = generate_single_batch(
            cehrgpt_model,
            cehrgpt_tokenizer,
            batched_prompt,
            max_new_tokens=args.context_window,
            mini_num_of_concepts=args.min_num_of_concepts,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            epsilon_cutoff=args.epsilon_cutoff,
        )
        # Clear the cache
        torch.cuda.empty_cache()

        reward = calculate_reward(
            batched_sequences, expected_concept_dist, cehrgpt_tokenizer
        )
        query_tensors = []
        response_tensors = []
        for sequence in batched_sequences:
            query_tensors.append(cehrgpt_tokenizer.encode(sequence[:4]))
            response_tensors.append(cehrgpt_tokenizer.encode(sequence[4:]))
        rewards = [torch.FloatTensor(reward) for _ in range(len(batched_sequences))]
        train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats=train_stats, batch={}, rewards=rewards)

    ppo_trainer.save_pretrained(model_folder_name)


def calculate_reward(
    batched_sequences: List[List[str]],
    expected_concept_dist: Dict[str, float],
    tokenizer: CehrGptTokenizer,
) -> torch.Tensor:
    actual_concept_dist = dict(
        Counter(
            [
                concept_id
                for sequence in batched_sequences
                for concept_id in sequence[4:]
            ]
        )
    )
    total_count = sum(actual_concept_dist.values())
    for concept_id in actual_concept_dist.keys():
        actual_concept_dist[concept_id] /= total_count
    # Translate the concept ids to token ids
    actual_dist = np.zeros(tokenizer.vocab_size)
    actual_dist[tokenizer.encode(actual_concept_dist.keys())] = list(
        actual_concept_dist.values()
    )
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    logprob_dist = torch.tensor(np.log(actual_dist + epsilon))
    # Translate the concept ids to token ids
    ref_dist = np.zeros(tokenizer.vocab_size)
    ref_dist[tokenizer.encode(expected_concept_dist.keys())] = list(
        expected_concept_dist.values()
    )
    ref_logprob_dist = torch.tensor(np.log(ref_dist + epsilon))
    return F.kl_div(logprob_dist, ref_logprob_dist, log_target=True, reduction="sum")


def create_arg_parser():
    base_arg_parser = create_inference_base_arg_parser(
        description="Arguments for finetuning cehr-gpt using PPO"
    )
    base_arg_parser.add_argument(
        "--mini_batch_size",
        dest="mini_batch_size",
        action="store",
        type=int,
        required=True,
    )
    base_arg_parser.add_argument(
        "--demographic_data_path",
        dest="demographic_data_path",
        action="store",
        help="The path for your concept_path",
        required=True,
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
