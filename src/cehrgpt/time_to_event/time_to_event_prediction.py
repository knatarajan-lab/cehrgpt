import datetime
import glob
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml
from cehrbert.runners.runner_util import load_parquet_as_dataset
from datasets import Dataset
from tqdm import tqdm
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.cehrgpt_args import create_inference_base_arg_parser
from cehrgpt.gpt_utils import get_cehrgpt_output_folder, is_visit_end, is_visit_start
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel

LOG = logging.get_logger("transformers")


@dataclass
class TaskConfig:
    task_name: str
    outcome_events: List[str]
    include_descendants: bool = False
    future_visit_start: int = 0
    future_visit_end: int = -1
    prediction_window_start: int = 0
    prediction_window_end: int = 365
    max_new_tokens: int = 128


def load_task_config_from_yaml(task_config_yaml_file_path: str) -> TaskConfig:
    # Read YAML file
    try:
        with open(task_config_yaml_file_path, "r") as stream:
            task_definition = yaml.safe_load(stream)
            return TaskConfig(**task_definition)
    except yaml.YAMLError | OSError as e:
        raise ValueError(
            f"Could not open the task_config yaml file from {task_config_yaml_file_path}"
        ) from e


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_folder,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=(
                torch.bfloat16
                if is_flash_attn_2_available() and args.use_bfloat16
                else "auto"
            ),
        )
        .eval()
        .to(get_device())
    )
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

    folder_name = get_cehrgpt_output_folder(args, cehrgpt_tokenizer)

    task_config = load_task_config_from_yaml(args.task_config)
    task_name = task_config.task_name
    outcome_events = task_config.outcome_events

    if task_config.include_descendants:
        if not args.concept_ancestor:
            raise RuntimeError(
                "When include_descendants is set to True, the concept_ancestor data needs to be provided."
            )
        concept_ancestor = pd.read_parquet(args.concept_ancestor)
        ancestor_concept_ids = [int(_) for _ in outcome_events if _.isnumeric()]
        descendant_concept_ids = (
            concept_ancestor[
                concept_ancestor.ancestor_concept_id.isin(ancestor_concept_ids)
            ]
            .descendant_concept_id.unique()
            .astype(str)
            .tolist()
        )
        descendant_concept_ids = [
            _ for _ in descendant_concept_ids if _ not in outcome_events
        ]
        outcome_events += descendant_concept_ids

    prediction_output_folder_name = os.path.join(
        args.output_folder, folder_name, task_name
    )
    temp_folder = os.path.join(args.output_folder, folder_name, "temp")
    os.makedirs(prediction_output_folder_name, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    LOG.info(f"Loading tokenizer at {args.model_folder}")
    LOG.info(f"Loading model at {args.model_folder}")
    LOG.info(f"Loading dataset_folder at {args.dataset_folder}")
    LOG.info(f"Write time sensitive predictions to {prediction_output_folder_name}")
    LOG.info(f"Context window {args.context_window}")
    LOG.info(f"Number of new tokens {task_config.max_new_tokens}")
    LOG.info(f"Temperature {args.temperature}")
    LOG.info(f"Repetition Penalty {args.repetition_penalty}")
    LOG.info(f"Sampling Strategy {args.sampling_strategy}")
    LOG.info(f"Epsilon cutoff {args.epsilon_cutoff}")
    LOG.info(f"Top P {args.top_p}")
    LOG.info(f"Top K {args.top_k}")

    # cehrgpt_model.resize_position_embeddings(
    #     cehrgpt_model.config.max_position_embeddings + task_config.max_new_tokens
    # )

    generation_config = TimeToEventModel.get_generation_config(
        tokenizer=cehrgpt_tokenizer,
        max_length=cehrgpt_model.config.n_positions,
        num_return_sequences=args.num_return_sequences,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        epsilon_cutoff=args.epsilon_cutoff,
        max_new_tokens=task_config.max_new_tokens,
    )
    ts_pred_model = TimeToEventModel(
        tokenizer=cehrgpt_tokenizer,
        model=cehrgpt_model,
        outcome_events=outcome_events,
        generation_config=generation_config,
        batch_size=args.batch_size,
        device=get_device(),
    )
    dataset = load_parquet_as_dataset(args.dataset_folder)

    def filter_func(examples):
        return [_ >= args.min_num_of_concepts for _ in examples["num_of_concepts"]]

    test_dataset = dataset.filter(filter_func, batched=True, batch_size=1000)
    test_dataset = test_dataset.shuffle(seed=42)

    # Filter out the records for which the predictions have been generated previously
    test_dataset = filter_out_existing_results(
        test_dataset, prediction_output_folder_name
    )
    tte_outputs = []
    for record in tqdm(test_dataset, total=len(test_dataset)):
        sample_identifier = (
            f"{record['person_id']}_{record['index_date'].strftime('%Y_%m_%d')}"
        )
        if acquire_lock_or_skip_if_already_exist(
            output_folder=temp_folder, sample_id=sample_identifier
        ):
            continue
        partial_history = record["concept_ids"]
        label = record["label"]
        time_to_event = record["time_to_event"] if "time_to_event" in record else None
        seq_length = len(partial_history)
        if (
            generation_config.max_length
            <= seq_length + generation_config.max_new_tokens
        ):
            start_index = seq_length - (
                generation_config.max_length - generation_config.max_new_tokens
            )
            # Make sure the first token starts on VS
            for i, token in enumerate(partial_history[start_index:]):
                if is_visit_start(token):
                    start_index += i
                    break
            partial_history = partial_history[start_index:]

        concept_time_to_event = ts_pred_model.predict_time_to_events(
            partial_history,
            task_config.future_visit_start,
            task_config.future_visit_end,
            task_config.prediction_window_start,
            task_config.prediction_window_end,
            args.debug,
            args.max_n_trial,
        )
        visit_counter = sum([int(is_visit_end(_)) for _ in partial_history])
        predicted_boolean_probability = (
            sum([event != "0" for event in concept_time_to_event.outcome_events])
            / len(concept_time_to_event.outcome_events)
            if concept_time_to_event
            else 0.0
        )
        tte_outputs.append(
            {
                "subject_id": record["person_id"],
                "prediction_time": record["index_date"],
                "visit_counter": visit_counter,
                "boolean_value": label,
                "predicted_boolean_probability": predicted_boolean_probability,
                "predicted_boolean_value": None,
                "time_to_event": time_to_event,
                "trials": (
                    asdict(concept_time_to_event) if concept_time_to_event else None
                ),
            }
        )
        delete_lock_create_processed_flag(
            output_folder=temp_folder, sample_id=sample_identifier
        )
        flush_to_disk_if_full(
            tte_outputs, prediction_output_folder_name, args.buffer_size
        )

    # Final flush
    flush_to_disk_if_full(tte_outputs, prediction_output_folder_name, args.buffer_size)
    # Remove the temp folder
    shutil.rmtree(temp_folder)


def delete_lock_create_processed_flag(output_folder: str, sample_id: str):
    processed_flag_file = os.path.join(output_folder, f"{sample_id}.done")
    # Obtain the lock for this example by creating an empty lock file
    try:
        # Using 'x' mode for exclusive creation; fails if the file already exists
        with open(processed_flag_file, "x"):
            pass  # The file is created; nothing is written to it
    except FileExistsError as e:
        raise FileExistsError(
            f"The processed flag file {processed_flag_file} already exists."
        ) from e

    lock_file = os.path.join(output_folder, f"{sample_id}.lock")
    # Clean up the lock file
    # Safely attempt to delete the lock file
    try:
        os.remove(lock_file)
    except OSError as e:
        raise OSError(f"Can not remove the lock file at {lock_file}") from e


def acquire_lock_or_skip_if_already_exist(output_folder: str, sample_id: str):
    lock_file = os.path.join(output_folder, f"{sample_id}.lock")
    if os.path.exists(lock_file):
        LOG.info(f"Other process acquired the lock --> %s. Skipping...", sample_id)
        return True
    processed_flag_file = os.path.join(output_folder, f"{sample_id}.done")
    if os.path.exists(processed_flag_file):
        LOG.info(f"The sample has been processed --> %s. Skipping...", sample_id)
        return True

    # Obtain the lock for this example by creating an empty lock file
    try:
        # Using 'x' mode for exclusive creation; fails if the file already exists
        with open(lock_file, "x"):
            pass  # The file is created; nothing is written to it
    except FileExistsError:
        LOG.info(f"Other process acquired the lock --> %s. Skipping...", sample_id)
        return True
    return False


def filter_out_existing_results(
    test_dataset: Dataset, prediction_output_folder_name: str
):
    parquet_files = glob.glob(os.path.join(prediction_output_folder_name, "*parquet"))
    if parquet_files:
        cohort_members = set()
        results_dataframe = pd.read_parquet(parquet_files)[
            ["subject_id", "prediction_time"]
        ]
        for row in results_dataframe.itertuples():
            cohort_members.add(
                (row.subject_id, row.prediction_time.strftime("%Y-%m-%d"))
            )

        def filter_func(batched):
            return [
                (person_id, index_date.strftime("%Y-%m-%d")) not in cohort_members
                for person_id, index_date in zip(
                    batched["person_id"], batched["index_date"]
                )
            ]

        test_dataset = test_dataset.filter(filter_func, batched=True, batch_size=1000)
    return test_dataset


def flush_to_disk_if_full(
    tte_outputs: List[Dict[str, Any]], prediction_output_folder_name, buffer_size: int
) -> None:
    if len(tte_outputs) >= buffer_size:
        LOG.info(
            f"{datetime.datetime.now()}: Flushing time to visit predictions to disk"
        )
        output_parquet_file = os.path.join(
            prediction_output_folder_name, f"{uuid.uuid4()}.parquet"
        )
        pd.DataFrame(
            tte_outputs,
            columns=[
                "subject_id",
                "prediction_time",
                "visit_counter",
                "boolean_value",
                "predicted_boolean_probability",
                "predicted_boolean_value",
                "time_to_event",
                "trials",
            ],
        ).to_parquet(output_parquet_file)
        tte_outputs.clear()


def create_arg_parser():
    base_arg_parser = create_inference_base_arg_parser(
        description="Arguments for time sensitive prediction"
    )
    base_arg_parser.add_argument(
        "--dataset_folder",
        dest="dataset_folder",
        action="store",
        help="The path for your dataset",
        required=True,
    )
    base_arg_parser.add_argument(
        "--num_return_sequences",
        dest="num_return_sequences",
        action="store",
        type=int,
        required=True,
    )
    base_arg_parser.add_argument(
        "--task_config", dest="task_config", action="store", required=True
    )
    base_arg_parser.add_argument(
        "--concept_ancestor", dest="concept_ancestor", action="store", required=False
    )
    base_arg_parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
    )
    base_arg_parser.add_argument(
        "--max_n_trial",
        dest="max_n_trial",
        action="store",
        type=int,
        default=2,
        required=False,
    )
    return base_arg_parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
