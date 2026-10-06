import datetime
import glob
import os
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import yaml
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert.runners.runner_util import load_parquet_as_dataset
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.cehrgpt_args import create_inference_base_arg_parser
from cehrgpt.gpt_utils import get_cehrgpt_output_folder, is_visit_end, is_visit_start
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import extract_cohort_sequences
from cehrgpt.runners.gpt_runner_util import (
    read_backbone,
    resolve_attn_implementation,
)
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments
from cehrgpt.time_to_event.time_to_event_model import TimeToEventModel

LOG = logging.get_logger("transformers")


@dataclass
class TaskConfig:
    task_name: str
    # concept ids, or tokens as they are in the tokenizer's vocabulary (e.g. "CPT4/33510")
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


def resolve_gpu_ids(gpu_ids: str) -> List[str]:
    """
    Resolve --gpu_ids ("all" or comma separated indices) into the CUDA_VISIBLE_DEVICES values.

    The indices are relative to the GPUs visible to this process, so the option composes
    with an existing CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=1,3 with --gpu_ids all
    runs on physical GPUs 1 and 3.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = [_.strip() for _ in visible.split(",") if _.strip()] if visible else None
    num_visible = len(visible_ids) if visible_ids is not None else torch.cuda.device_count()
    if num_visible == 0:
        raise RuntimeError("--gpu_ids was provided but no CUDA device is visible.")
    if gpu_ids.strip().lower() == "all":
        indices = list(range(num_visible))
    else:
        indices = [int(_) for _ in gpu_ids.split(",") if _.strip()]
    if len(set(indices)) != len(indices):
        raise ValueError(f"--gpu_ids contains duplicates: {gpu_ids}")
    out_of_range = [_ for _ in indices if not 0 <= _ < num_visible]
    if out_of_range:
        raise ValueError(
            f"--gpu_ids {out_of_range} out of range, {num_visible} GPU(s) are visible."
        )
    return [visible_ids[_] if visible_ids is not None else str(_) for _ in indices]


def strip_cli_option(argv: List[str], option: str) -> List[str]:
    """Drop `option` and its value (both `--opt value` and `--opt=value`) from argv."""
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
        elif arg == option:
            skip_next = True
        elif not arg.startswith(option + "="):
            stripped.append(arg)
    return stripped


def load_time_to_event_dataset(args) -> Dataset:
    if args.tokenized_full_dataset_path and args.cohort_folder:
        LOG.info(
            "Extracting cohort sequences from the pre-tokenized dataset at %s "
            "using cohort_folder %s",
            args.tokenized_full_dataset_path,
            args.cohort_folder,
        )
        data_args = DataTrainingArguments(
            data_folder=args.cohort_folder,
            dataset_prepared_path=args.output_folder,
            cohort_folder=args.cohort_folder,
            observation_window=args.observation_window,
        )
        cehrgpt_args = CehrGPTArguments(
            tokenized_full_dataset_path=args.tokenized_full_dataset_path,
            allow_missing_tokenized_persons=args.allow_missing_tokenized_persons,
        )
        processed_dataset = extract_cohort_sequences(
            data_args, cehrgpt_args, CacheFileCollector()
        )
        dataset = concatenate_datasets(list(processed_dataset.values()))
        dataset = dataset.rename_column("classifier_label", "label")
        # num_of_concepts (if present) reflects the patient's full history length before
        # slicing to the observation window; recompute it from the sliced concept_ids so
        # the min_num_of_concepts filter below reflects the actual available context.
        dataset = dataset.map(
            lambda batch: {"num_of_concepts": [len(_) for _ in batch["concept_ids"]]},
            batched=True,
        )
        return dataset
    return load_parquet_as_dataset(args.dataset_folder)


def prepare_test_dataset(args, prediction_output_folder_name: str) -> Dataset:
    dataset = load_time_to_event_dataset(args)

    def filter_func(examples):
        return [_ >= args.min_num_of_concepts for _ in examples["num_of_concepts"]]

    test_dataset = dataset.filter(filter_func, batched=True, batch_size=1000)
    test_dataset = test_dataset.shuffle(seed=42)

    # Filter out the records for which the predictions have been generated previously
    return filter_out_existing_results(test_dataset, prediction_output_folder_name)


def run_on_multiple_gpus(args, gpu_ids: List[str]) -> int:
    """
    Split the pending predictions into disjoint shards and run one worker process per GPU.

    The dataset is prepared once here (so the workers don't race on the datasets cache),
    each worker re-runs this module on its own shard with CUDA_VISIBLE_DEVICES pinned to
    one GPU and writes its parquet files into <prediction folder>/shard_<i>. Samples that
    already have predictions in the prediction folder (including any shard_<i> subfolder
    from an earlier, possibly interrupted, run) are excluded before sharding, so rerunning
    the same command resumes where it stopped.
    """
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    folder_name = get_cehrgpt_output_folder(args, cehrgpt_tokenizer)
    task_name = load_task_config_from_yaml(args.task_config).task_name
    prediction_output_folder_name = os.path.join(
        args.output_folder, folder_name, task_name
    )
    shard_root = os.path.join(args.output_folder, folder_name, "temp", "shards")
    log_folder = os.path.join(args.output_folder, folder_name, "worker_logs")
    # Shards left over from an interrupted run are stale, finished samples are skipped
    # through the existing parquet files instead
    shutil.rmtree(shard_root, ignore_errors=True)
    os.makedirs(prediction_output_folder_name, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    test_dataset = prepare_test_dataset(args, prediction_output_folder_name)
    num_workers = min(len(gpu_ids), len(test_dataset))
    if num_workers == 0:
        LOG.info("There are no pending samples to predict.")
        return 0
    LOG.info(
        "Splitting %s samples across %s GPUs: %s",
        len(test_dataset),
        num_workers,
        gpu_ids[:num_workers],
    )

    worker_argv = strip_cli_option(sys.argv[1:], "--gpu_ids")
    processes: List[subprocess.Popen] = []
    log_paths = []
    # Make sure the workers don't outlive us when we are asked to stop
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for worker_index in range(num_workers):
            shard_path = os.path.join(shard_root, f"shard_{worker_index}")
            test_dataset.shard(
                num_shards=num_workers, index=worker_index, contiguous=False
            ).save_to_disk(shard_path)
            log_path = os.path.join(
                log_folder, f"worker_{worker_index}_gpu_{gpu_ids[worker_index]}.log"
            )
            log_paths.append(log_path)
            with open(log_path, "w") as log_file:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "cehrgpt.time_to_event.time_to_event_prediction",
                            *worker_argv,
                            "--prepared_dataset_path",
                            shard_path,
                        ],
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": gpu_ids[worker_index],
                        },
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                )
            LOG.info(
                "Started worker %s on GPU %s, log: %s",
                worker_index,
                gpu_ids[worker_index],
                log_path,
            )
        exit_codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    failed = [i for i, code in enumerate(exit_codes) if code != 0]
    if failed:
        for i in failed:
            LOG.error(
                "Worker %s exited with code %s, see %s", i, exit_codes[i], log_paths[i]
            )
        return 1
    shutil.rmtree(shard_root, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(shard_root))
    except OSError:
        pass
    LOG.info("All %s workers finished", num_workers)
    return 0


def split_outcome_events(
    outcome_events: List[Union[str, int]]
) -> Tuple[List[int], List[str]]:
    """Separates the concept ids from the tokens of the outcome_events of a task config.

    An outcome event is either a concept id (numeric), or a token written exactly as it is in the
    tokenizer's vocabulary, e.g. "CPT4/33510". A config uses one of the two."""
    concept_ids, tokens = [], []
    for event in outcome_events:
        if str(event).isnumeric():
            concept_ids.append(int(event))
        elif isinstance(event, str) and event:
            tokens.append(event)
        else:
            raise ValueError(
                f"The outcome event {event!r} is neither a concept id nor a token"
            )
    if concept_ids and tokens:
        raise ValueError(
            "outcome_events cannot mix concept ids and tokens, use one of them"
        )
    if not concept_ids and not tokens:
        raise ValueError("outcome_events is empty")
    return concept_ids, tokens


def main(args):
    uses_tokenized_full_dataset = bool(
        args.tokenized_full_dataset_path and args.cohort_folder
    )
    if not uses_tokenized_full_dataset and not args.dataset_folder:
        raise RuntimeError(
            "Either --dataset_folder, or both --tokenized_full_dataset_path and "
            "--cohort_folder, must be provided."
        )

    if args.gpu_ids and not args.prepared_dataset_path:
        gpu_ids = resolve_gpu_ids(args.gpu_ids)
        if len(gpu_ids) > 1:
            exit_code = run_on_multiple_gpus(args, gpu_ids)
            if exit_code:
                sys.exit(exit_code)
            return
        # CUDA has not been initialized yet, so this pins the single requested GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            args.model_folder,
            attn_implementation=resolve_attn_implementation(
                backbone=read_backbone(args.model_folder)
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
    outcome_events, tokens = split_outcome_events(task_config.outcome_events)
    if tokens and task_config.include_descendants:
        raise ValueError(
            "include_descendants expands concept ids, list the tokens of the descendants instead"
        )

    if task_config.include_descendants:
        if not args.concept_ancestor:
            raise RuntimeError(
                "When include_descendants is set to True, the concept_ancestor data needs to be provided."
            )
        concept_ancestor = pd.read_parquet(args.concept_ancestor)
        descendant_concept_ids = (
            concept_ancestor[
                concept_ancestor.ancestor_concept_id.isin(outcome_events)
            ]
            .descendant_concept_id.unique()
            .astype(str)
            .tolist()
        )
        descendant_concept_ids = [
            _ for _ in descendant_concept_ids if _ not in outcome_events
        ]
        outcome_events += descendant_concept_ids

    if tokens:
        outcome_events = tokens

    prediction_output_folder_name = os.path.join(
        args.output_folder, folder_name, task_name
    )
    if args.prepared_dataset_path:
        # A --gpu_ids worker owns its shard: it writes into its own shard_<i> output folder
        # and must not touch the other workers' locks
        shard_name = os.path.basename(args.prepared_dataset_path.rstrip("/"))
        prediction_output_folder_name = os.path.join(
            prediction_output_folder_name, shard_name
        )
        temp_folder = f"{args.prepared_dataset_path.rstrip('/')}_locks"
    else:
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
    if args.prepared_dataset_path:
        test_dataset = load_from_disk(args.prepared_dataset_path)
    else:
        test_dataset = prepare_test_dataset(args, prediction_output_folder_name)
    vocab = cehrgpt_tokenizer.get_vocab()
    tte_outputs = []
    for record in tqdm(test_dataset, total=len(test_dataset)):
        index_date = record["index_date"]
        # extract_cohort_sequences produces index_date as a POSIX timestamp rather than
        # a datetime object
        if isinstance(index_date, (int, float)):
            index_date = datetime.datetime.fromtimestamp(
                index_date, datetime.timezone.utc
            ).replace(tzinfo=None)
        sample_identifier = f"{record['person_id']}_{index_date.strftime('%Y_%m_%d')}"
        if acquire_lock_or_skip_if_already_exist(
            output_folder=temp_folder, sample_id=sample_identifier
        ):
            continue
        partial_history = record["concept_ids"]
        # Filter out the tokens that are not in the vocabulary
        partial_history = [
            concept_id for concept_id in partial_history
            if concept_id in vocab
        ]
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
                "prediction_time": index_date,
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
    # Recursive so predictions written by previous --gpu_ids runs (shard_<i> folders) count
    parquet_files = glob.glob(
        os.path.join(prediction_output_folder_name, "**", "*.parquet"), recursive=True
    )
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
            keys = []
            for person_id, index_date in zip(
                batched["person_id"], batched["index_date"]
            ):
                # extract_cohort_sequences produces index_date as a POSIX timestamp
                # rather than a datetime object
                if isinstance(index_date, (int, float)):
                    index_date = datetime.datetime.fromtimestamp(
                        index_date, datetime.timezone.utc
                    ).replace(tzinfo=None)
                keys.append((person_id, index_date.strftime("%Y-%m-%d")))
            return [key not in cohort_members for key in keys]

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
        help="The path for your dataset. Required unless both --tokenized_full_dataset_path "
        "and --cohort_folder are provided.",
        required=False,
        default=None,
    )
    base_arg_parser.add_argument(
        "--tokenized_full_dataset_path",
        dest="tokenized_full_dataset_path",
        action="store",
        help="Path to a fully tokenized dataset. When provided together with "
        "--cohort_folder, cohort sequences are sliced out of it (up to each patient's "
        "index date) via extract_cohort_sequences instead of loading --dataset_folder.",
        required=False,
        default=None,
    )
    base_arg_parser.add_argument(
        "--cohort_folder",
        dest="cohort_folder",
        action="store",
        help="Directory containing the cohort's parquet files (person_id/index_date/label, "
        "or MEDS subject_id/prediction_time/boolean_value). Used with "
        "--tokenized_full_dataset_path.",
        required=False,
        default=None,
    )
    base_arg_parser.add_argument(
        "--observation_window",
        dest="observation_window",
        action="store",
        type=int,
        help="Observation window in days before the index date to extract the sequence "
        "from. Used with --tokenized_full_dataset_path; defaults to the patient's full "
        "history up to the index date.",
        required=False,
        default=None,
    )
    base_arg_parser.add_argument(
        "--allow_missing_tokenized_persons",
        dest="allow_missing_tokenized_persons",
        action="store_true",
        help="Skip cohort persons that are missing from the tokenized dataset (with a "
        "warning) instead of failing. Used with --tokenized_full_dataset_path.",
    )
    base_arg_parser.add_argument(
        "--gpu_ids",
        dest="gpu_ids",
        action="store",
        help="Split the predictions across multiple GPUs: either 'all' or comma separated "
        "indices of the visible GPUs, e.g. 0,1,3. One worker process is started per GPU, "
        "each loading its own copy of the model and predicting a disjoint shard of the "
        "samples. Indices are relative to CUDA_VISIBLE_DEVICES when it is set. By default "
        "a single process is used.",
        required=False,
        default=None,
    )
    base_arg_parser.add_argument(
        "--prepared_dataset_path",
        dest="prepared_dataset_path",
        action="store",
        help="Internal, set by --gpu_ids: the shard of pending samples this worker predicts.",
        required=False,
        default=None,
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
