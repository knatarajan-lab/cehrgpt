import datetime
import os
import pathlib
import pickle

import numpy as np
import polars as pl
import torch
from cehrbert.runners.runner_util import load_parquet_as_dataset
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.utils import logging
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, create_reference_model

from cehrgpt.data.encoder_decoder.rl_ppo_instruct_cehrgpt_data_collator import (
    InstructCehrGptPPODataCollator,
)
from cehrgpt.generation.cehrgpt_patient.clinical_statement_generator import (
    ClinicalStatementGenerator,
    ConditionDrugKnowledgeGraph,
)
from cehrgpt.generation.encoder_decoder.instruct_model_cli import (
    generate_responses,
    setup_model,
)
from cehrgpt.models.rl.cehrgpt_ppo_trainer import CehrGptPPOTrainer
from cehrgpt.models.rl.rewards.reward_model import CEHRGPTRewardModel
from cehrgpt.omop.vocab_utils import (
    create_drug_ingredient_to_brand_drug_map,
    generate_ancestor_descendant_map,
    generate_concept_maps,
)
from cehrgpt.rl_finetune.ppo_finetune_v2 import (
    create_arg_parser as rl_create_arg_parser,
)

LOG = logging.get_logger("transformers")


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder_tokenizer, cehrgpt_tokenizer, encoder_decoder_model = setup_model(
        tokenizer_path=args.tokenizer_folder,
        model_path=args.model_folder,
        device=device,
    )
    model = AutoModelForCausalLMWithValueHead(encoder_decoder_model).to(device)
    model.is_peft_model = False
    ref_model = create_reference_model(model).to(device)

    # create a ppo trainer
    ppo_trainer = CehrGptPPOTrainer(
        config=PPOConfig(
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            init_kl_coef=args.init_kl_coef,
            vf_coef=args.vf_coef,
            kl_penalty=args.kl_penalty,
            gamma=args.gamma,
            use_score_scaling=args.use_score_scaling,
        ),
        model=model,
        ref_model=ref_model,
        tokenizer=cehrgpt_tokenizer,
        training_data_collator=InstructCehrGptPPODataCollator(
            encoder_tokenizer=encoder_tokenizer,
            cehrgpt_tokenizer=cehrgpt_tokenizer,
            max_length=args.context_window,
        ),
    )

    LOG.info("%s: Loading tokenizer at %s", datetime.datetime.now(), args.model_folder)
    LOG.info("%s: Loading model at %s", datetime.datetime.now(), args.model_folder)
    LOG.info("%s: Context window %s", datetime.datetime.now(), args.context_window)
    LOG.info("%s: Temperature %s", datetime.datetime.now(), args.temperature)
    LOG.info(
        "%s: Repetition Penalty %s", datetime.datetime.now(), args.repetition_penalty
    )
    LOG.info(
        "%s: Sampling Strategy %s", datetime.datetime.now(), args.sampling_strategy
    )
    LOG.info("%s: Num beam %s", datetime.datetime.now(), args.num_beams)
    LOG.info("%s: Num beam groups %s", datetime.datetime.now(), args.num_beam_groups)
    LOG.info("%s: Epsilon cutoff %s", datetime.datetime.now(), args.epsilon_cutoff)
    LOG.info("%s: Top P %s", datetime.datetime.now(), args.top_p)
    LOG.info("%s: Top K %s", datetime.datetime.now(), args.top_k)
    LOG.info(
        "%s: Loading demographic_info at %s",
        datetime.datetime.now(),
        args.demographic_data_path,
    )

    # Configure model generation settings
    generation_config = GenerationConfig(
        max_length=args.context_window,
        mini_num_of_concepts=args.min_num_of_concepts,
        bos_token_id=cehrgpt_tokenizer.start_token_id,
        eos_token_id=cehrgpt_tokenizer.end_token_id,
        pad_token_id=cehrgpt_tokenizer.pad_token_id,
        decoder_start_token_id=cehrgpt_tokenizer.start_token_id,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        epsilon_cutoff=args.epsilon_cutoff,
        do_sample=True,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        renormalize_logits=True,
    )

    LOG.info(
        "%s: Loading the knowledge graph from %s",
        datetime.datetime.now(),
        args.knowledge_graph_path,
    )
    with open(args.knowledge_graph_path, "rb") as f:
        knowledge_graph = pickle.load(f)

    LOG.info(
        "%s: Loading the concept and concept ancestor from %s",
        datetime.datetime.now(),
        args.vocabulary_dir,
    )
    concept = pl.read_parquet(os.path.join(args.vocabulary_dir, "concept", "*parquet"))
    concept_ancestor = pl.read_parquet(
        os.path.join(args.vocabulary_dir, "concept_ancestor", "*parquet")
    )

    LOG.info(
        "%s: Constructing the ingredient to branded drug mapping",
        datetime.datetime.now(),
    )
    drug_ingredient_to_brand_drug_map = create_drug_ingredient_to_brand_drug_map(
        concept, concept_ancestor
    )

    LOG.info(
        "%s: Constructing the ancestor to descendent mapping", datetime.datetime.now()
    )
    ancestor_descendant_map = generate_ancestor_descendant_map(
        concept_ancestor_pl=concept_ancestor,
        concept_ids=cehrgpt_tokenizer.get_vocab().keys(),
    )
    LOG.info(
        "%s: Constructing the concept mapping dictionaries", datetime.datetime.now()
    )
    concept_name_map, concept_domain_map = generate_concept_maps(concept)
    clinical_statement_generator = ClinicalStatementGenerator(
        condition_drug_knowledge_graph=ConditionDrugKnowledgeGraph(
            knowledge_graph=knowledge_graph,
            drug_ingredient_to_brand_drug_map=drug_ingredient_to_brand_drug_map,
        ),
        # This allows all conditions to be included
        allowed_clinical_conditions=None,
    )

    reward_model = CEHRGPTRewardModel(
        ancestor_descendent_map=ancestor_descendant_map,
        ingredient_to_drug_map=drug_ingredient_to_brand_drug_map,
        concept_name_map=concept_name_map,
    )

    LOG.info(
        "%s: Loading the training data from %s",
        datetime.datetime.now(),
        args.demographic_data_path,
    )
    dataset = load_parquet_as_dataset(args.demographic_data_path).filter(
        lambda batched: [
            model.config.n_positions >= num_of_concepts > args.min_num_tokens
            for num_of_concepts in batched["num_of_concepts"]
        ],
        batched=True,
    )

    logs = []
    device = ppo_trainer.current_device
    total_rows = len(dataset)
    num_of_micro_batches = args.batch_size // args.mini_batch_size
    for i in tqdm(range(args.num_of_steps)):
        LOG.info(f"{datetime.datetime.now()}: Batch {i} started")
        batched_queries = []
        batched_sequences = []
        batched_values = []
        batched_value_indicators = []
        batched_encoder_age_concept_prompt_tuples = []
        for _ in range(num_of_micro_batches):
            random_patient_sequences = [
                record["concept_ids"]
                for record in dataset.select(
                    np.random.randint(0, total_rows, args.mini_batch_size)
                )
            ]
            queries = []
            for patient_sequence in random_patient_sequences:
                clinical_statement, prompt_tuples = (
                    clinical_statement_generator.generate_clinical_statement(
                        patient_sequence,
                        concept_name_mapping=concept_name_map,
                        concept_domain_mapping=concept_domain_map,
                        return_seed_concepts=True,
                    )
                )
                queries.append(clinical_statement)
                batched_encoder_age_concept_prompt_tuples.append(prompt_tuples)

            micro_batched_sequences = generate_responses(
                queries=queries,
                encoder_tokenizer=encoder_tokenizer,
                cehrgpt_tokenizer=cehrgpt_tokenizer,
                model=encoder_decoder_model,
                device=device,
                generation_config=generation_config,
            )

            # Clear the cache
            torch.cuda.empty_cache()
            batched_queries.extend(queries)
            batched_sequences.extend(micro_batched_sequences["sequences"])
            batched_values.extend(micro_batched_sequences["values"])
            batched_value_indicators.extend(micro_batched_sequences["value_indicators"])

        LOG.info("%s: Batch %s sequence generated", datetime.datetime.now(), i)
        query_tensors = []
        response_tensors = []
        value_tensors = []
        value_indicator_tensors = []
        rewards = []
        for query, sequence, sequence_val, sequence_val_indicator in zip(
            batched_queries, batched_sequences, batched_values, batched_value_indicators
        ):
            # Convert sequence to a NumPy array if it's not already one
            sequence_array = np.asarray(sequence)
            # Find the end token
            condition_array = sequence_array == cehrgpt_tokenizer.end_token
            end_index = (
                np.argmax(condition_array)
                if condition_array.any()
                else len(sequence_array) - 1
            )
            sequence = sequence[: end_index + 1]
            encoder_input = encoder_tokenizer(query, return_tensors="pt").to(device)
            query_tensors.extend(encoder_input["input_ids"])
            response_tensors.append(
                torch.LongTensor(cehrgpt_tokenizer.encode(sequence))
            )
            value_tensors.append(
                torch.LongTensor(cehrgpt_tokenizer.encode_value(sequence_val))
            )
            value_indicator_tensors.append(torch.BoolTensor(sequence_val_indicator))
            reward = reward_model.get_reward(
                query,
                sequence,
                batched_encoder_age_concept_prompt_tuples,
                concept_name_map=concept_name_map,
                concept_domain_map=concept_domain_map,
            )
            LOG.info(
                "%s: Batch %s Reward: %s}",
                {datetime.datetime.now()},
                i,
                reward,
            )
            rewards.append(torch.FloatTensor([reward]))

        train_stats = ppo_trainer.step(
            query_tensors,
            response_tensors,
            rewards,
            value_tensors,
            value_indicator_tensors,
        )
        LOG.info("%s: Batch %s stats: %s}", datetime.datetime.now(), i, train_stats)
        if i != 0 and i % args.save_step == 0:
            checkpoint_folder = pathlib.Path(args.output_folder) / f"checkpoint-{i}"
            checkpoint_folder.mkdir(exist_ok=True)
            ppo_trainer.log_stats(stats=train_stats, batch={}, rewards=rewards)
            ppo_trainer.save_pretrained(checkpoint_folder)

    ppo_trainer.save_pretrained(args.output_folder)
    with open(os.path.join(args.output_folder, "ppo_finetune_stats.pkl"), "wb") as f:
        pickle.dump(logs, f)


def create_arg_parser():
    parser = rl_create_arg_parser()
    parser.add_argument(
        "--vocabulary_dir",
        action="store",
        help="The directory that contains both concept and concept_ancestor data",
        required=True,
    )
    parser.add_argument(
        "--knowledge_graph_path",
        action="store",
        help="Knowledge graph path",
        required=True,
    )
    parser.add_argument(
        "--save_step",
        action="store",
        type=int,
        required=True,
    )
    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
