import datetime
import os

import yaml
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
import pandas as pd
import torch

from transformers.utils import logging, is_flash_attn_2_available

from .time_to_event_model import TimeToEventModel
from ..models.tokenization_hf_cehrgpt import CehrGptTokenizer
from ..models.hf_cehrgpt import CEHRGPT2LMHeadModel
from ..cehrgpt_args import create_inference_base_arg_parser
from ..gpt_utils import is_visit_end, get_cehrgpt_output_folder
from cehrbert.runners.runner_util import load_parquet_as_dataset

LOG = logging.get_logger("transformers")


@dataclass
class TaskConfig:
    task_name: str
    outcome_events: List[str]
    include_descendants: bool = False
    n_future_visits: int = 1
    future_visit_offset: int = 0


def load_task_config_from_yaml(task_config_yaml_file_path: str) -> TaskConfig:
    # Read YAML file
    try:
        with open(task_config_yaml_file_path, 'r') as stream:
            task_definition = yaml.safe_load(stream)
            return TaskConfig(**task_definition)
    except yaml.YAMLError | OSError as e:
        raise ValueError(f"Could not open the task_config yaml file from {task_config_yaml_file_path}") from e


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(
        args
):
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = CEHRGPT2LMHeadModel.from_pretrained(
        args.model_folder,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager",
        torch_dtype=torch.bfloat16 if is_flash_attn_2_available() else "auto"
    ).eval().to(get_device())
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    folder_name = get_cehrgpt_output_folder(args, cehrgpt_tokenizer)

    task_config = load_task_config_from_yaml(args.task_config)
    task_name = task_config.task_name
    outcome_events = task_config.outcome_events

    prediction_output_folder_name = os.path.join(
        folder_name,
        task_name
    )
    os.makedirs(prediction_output_folder_name, exist_ok=True)

    LOG.info(f'Loading tokenizer at {args.model_folder}')
    LOG.info(f'Loading model at {args.model_folder}')
    LOG.info(f'Loading dataset_folder at {args.dataset_folder}')
    LOG.info(f'Write time sensitive predictions to {prediction_output_folder_name}')
    LOG.info(f'Context window {args.context_window}')
    LOG.info(f'Temperature {args.temperature}')
    LOG.info(f'Repetition Penalty {args.repetition_penalty}')
    LOG.info(f'Sampling Strategy {args.sampling_strategy}')
    LOG.info(f'Epsilon cutoff {args.epsilon_cutoff}')
    LOG.info(f'Top P {args.top_p}')
    LOG.info(f'Top K {args.top_k}')

    generation_config = TimeToEventModel.get_generation_config(
        tokenizer=cehrgpt_tokenizer,
        max_length=cehrgpt_model.config.n_positions,
        num_return_sequences=args.num_return_sequences,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        epsilon_cutoff=args.epsilon_cutoff
    )
    ts_pred_model = TimeToEventModel(
        tokenizer=cehrgpt_tokenizer,
        model=cehrgpt_model,
        outcome_events=outcome_events,
        generation_config=generation_config,
        batch_size=args.batch_size,
        device=get_device()
    )
    dataset = load_parquet_as_dataset(
        args.dataset_folder
    )

    def filter_func(examples):
        return [cehrgpt_model.config.n_positions >= _ >= args.min_num_of_concepts for _ in examples['num_of_concepts']]

    test_dataset = dataset.filter(
        filter_func,
        batched=True,
        batch_size=1000
    )

    for record in tqdm(test_dataset, total=len(test_dataset)):
        partial_history = record["concept_ids"]
        label = record["label"]
        time_to_event = record["time_to_event"] if "time_to_event" in record else None
        concept_time_to_events = ts_pred_model.predict_time_to_events(
            partial_history,
            task_config.n_future_visits,
            task_config.future_visit_offset
        )
        visit_counter = sum([int(is_visit_end(_)) for _ in partial_history])
        tte_output = [{
            "person_id": record["person_id"],
            "visit_counter": visit_counter,
            "label": label,
            "time_to_event": time_to_event,
            "predictions": concept_time_to_events
        }]
        LOG.info(f'{datetime.datetime.now()}: Flushing time to visit predictions to disk')
        pd.DataFrame(
            tte_output,
            columns=[
                "person_id", "visit_counter", "label", "time_to_event", "predictions"
            ]
        ).to_parquet(os.path.join(prediction_output_folder_name, f'{record["person_id"]}.parquet'))


def create_arg_parser():
    base_arg_parser = create_inference_base_arg_parser(description='Arguments for time sensitive prediction')
    base_arg_parser.add_argument(
        '--dataset_folder',
        dest='dataset_folder',
        action='store',
        help='The path for your dataset',
        required=True
    )
    base_arg_parser.add_argument(
        '--num_return_sequences',
        dest='num_return_sequences',
        action='store',
        type=int,
        required=True
    )
    base_arg_parser.add_argument(
        '--task_config',
        dest='task_config',
        action='store',
        required=True
    )
    base_arg_parser.add_argument(
        '--concept_ancestor',
        dest='concept_ancestor',
        action='store',
        required=False
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
