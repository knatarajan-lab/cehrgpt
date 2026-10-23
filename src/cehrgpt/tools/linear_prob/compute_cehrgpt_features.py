import glob
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector
from cehrbert.runners.runner_util import generate_prepared_ds_path
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer_utils import is_main_process
from transformers.utils import logging

from cehrgpt.data.cache_cleanup import clean_up_dataset_caches
from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_finetuning_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import (
    CehrGptDataCollator,
    SamplePackingCehrGptDataCollator,
)
from cehrgpt.data.sample_packing_sampler import SamplePackingBatchSampler
from cehrgpt.models.hf_cehrgpt import (
    CEHRGPT2LMHeadModel,
    extract_features_from_packed_sequence,
)
from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN, RANDOM_TOKEN
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.data_utils import (
    extract_cohort_sequences,
    prepare_finetune_dataset,
)
from cehrgpt.runners.gpt_runner_util import (
    parse_runner_args,
    read_backbone,
    resolve_attn_implementation,
)
from cehrgpt.runners.hf_cehrgpt_pretrain_runner import tokenizer_exists
from cehrgpt.runners.multi_gpu_util import (
    expand_gpu_workers,
    resolve_gpu_ids,
    run_workers,
    strip_cli_option,
)
from cehrgpt.tools.linear_prob.feature_io import (
    AsyncParquetWriter,
    DemographicsLookup,
    write_feature_info,
)

LOG = logging.get_logger("transformers")


def get_torch_dtype(torch_dtype: Optional[str] = None) -> Union[torch.dtype, str]:
    if torch_dtype and hasattr(torch, torch_dtype):
        return getattr(torch, torch_dtype)
    return torch.float32


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    """Promote a 0-d numpy array to 1-d; leave higher-rank arrays unchanged."""
    return np.asarray([arr]) if arr.ndim == 0 else arr


def _create_feature_dataset(data_args, training_args, cehrgpt_args, tokenizer, cache_file_collector):
    """Process the raw data into a tokenized DatasetDict for feature extraction."""
    if cehrgpt_args.tokenized_full_dataset_path is not None:
        return extract_cohort_sequences(data_args, cehrgpt_args, tokenizer)

    final_splits = prepare_finetune_dataset(
        data_args, training_args, cehrgpt_args, cache_file_collector
    )
    # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
    if not data_args.streaming:
        if "visit_concept_ids" in final_splits["train"].column_names:
            final_splits = final_splits.remove_columns(["visit_concept_ids"])

    return create_cehrgpt_finetuning_dataset(
        dataset=final_splits,
        cehrgpt_tokenizer=tokenizer,
        data_args=data_args,
        cache_file_collector=cache_file_collector,
    )


def prepare_processed_dataset(
    cehrgpt_args,
    data_args,
    model_args,
    training_args,
    cehrgpt_tokenizer: CehrGptTokenizer,
) -> Tuple[DatasetDict, CehrGptTokenizer, str]:
    """
    Load the tokenized cohort dataset from disk, or create it, and drop already extracted rows.

    Returns the dataset, the tokenizer to use with it (it is reloaded from the output folder
    when the dataset was prepared earlier with expand_tokenizer), and that tokenizer's path.
    """
    prepared_ds_path = generate_prepared_ds_path(
        data_args, model_args, data_folder=data_args.cohort_folder
    )
    cache_file_collector = CacheFileCollector()
    tokenizer_path = model_args.tokenizer_name_or_path
    processed_dataset = None

    if cehrgpt_args.refresh_processed_dataset and prepared_ds_path.exists():
        LOG.info("Refreshing prepared dataset: removing cache at %s", prepared_ds_path)
        shutil.rmtree(prepared_ds_path)

    if any(prepared_ds_path.glob("*")):
        LOG.info("Loading prepared dataset from disk at %s...", prepared_ds_path)
        processed_dataset = load_from_disk(str(prepared_ds_path))
        if cehrgpt_args.expand_tokenizer:
            if tokenizer_exists(training_args.output_dir):
                cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
                    training_args.output_dir
                )
                tokenizer_path = training_args.output_dir
            else:
                LOG.warning(
                    "CehrGptTokenizer must exist in %s when the dataset has been processed "
                    "and expand_tokenizer is set to True. "
                    "Please delete the processed dataset at %s.",
                    training_args.output_dir,
                    prepared_ds_path,
                )
                processed_dataset = None
                shutil.rmtree(prepared_ds_path)

    if processed_dataset is None:
        if is_main_process(training_args.local_rank):
            processed_dataset = _create_feature_dataset(
                data_args, training_args, cehrgpt_args, cehrgpt_tokenizer, cache_file_collector
            )
            if not data_args.streaming:
                # Drop any splits that ended up empty — save_to_disk raises
                # SchemaInferenceError when asked to write a zero-example split.
                empty_splits = [k for k, v in processed_dataset.items() if len(v) == 0]
                if empty_splits:
                    LOG.warning("Dropping empty splits before saving: %s", empty_splits)
                    processed_dataset = DatasetDict(
                        {k: v for k, v in processed_dataset.items() if len(v) > 0}
                    )
                processed_dataset.save_to_disk(prepared_ds_path)
                stats = clean_up_dataset_caches(processed_dataset, cache_file_collector)
                LOG.info(
                    "Clean up the cached files for the cehrgpt feature dataset: %s",
                    stats,
                )
            else:
                cache_file_collector.remove_cache_files()

        # After main-process-only operations, synchronize all processes to ensure consistency
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load the dataset from disk again in torch distributed training
        processed_dataset = load_from_disk(str(prepared_ds_path))

    # Getting the existing features
    feature_folders = glob.glob(
        os.path.join(training_args.output_dir, "*", "features", "*.parquet")
    )
    if feature_folders:
        existing_features = pd.concat(
            [
                pd.read_parquet(f, columns=["subject_id", "prediction_time_posix"])
                for f in feature_folders
            ],
            ignore_index=True,
        )
        subject_prediction_tuples = set(
            existing_features.apply(
                lambda row: f"{int(row['subject_id'])}-{int(row['prediction_time_posix'])}",
                axis=1,
            ).tolist()
        )
        processed_dataset = processed_dataset.filter(
            lambda _batch: [
                f"{int(subject)}-{int(time)}" not in subject_prediction_tuples
                for subject, time in zip(_batch["person_id"], _batch["index_date"])
            ],
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            batched=True,
        )
        LOG.info(
            "The datasets after filtering (train: %s, validation: %s, test: %s)",
            len(processed_dataset["train"]),
            len(processed_dataset["validation"]),
            len(processed_dataset["test"]),
        )

    return processed_dataset, cehrgpt_tokenizer, tokenizer_path


def shard_processed_dataset(
    processed_dataset: DatasetDict, num_shards: int, shard_root: str
) -> List[str]:
    """
    Split train+validation and test into `num_shards` disjoint, round-robin shards.

    Round-robin (not contiguous) keeps the sequence length distribution of every shard close
    to the full dataset, which keeps the sample packing batches comparable across workers.
    Each shard is saved as a DatasetDict with a "train" and a "test" split.
    """
    train_set = concatenate_datasets(
        [processed_dataset[s] for s in ("train", "validation") if s in processed_dataset]
    )
    test_set = processed_dataset["test"] if "test" in processed_dataset else train_set.select([])
    shard_paths = []
    for index in range(num_shards):
        shard_path = os.path.join(shard_root, f"shard_{index}")
        splits = {
            "train": train_set.shard(num_shards, index, contiguous=False),
            "test": test_set.shard(num_shards, index, contiguous=False),
        }
        # save_to_disk raises on a zero-example split
        DatasetDict({k: v for k, v in splits.items() if len(v) > 0}).save_to_disk(shard_path)
        shard_paths.append(shard_path)
    return shard_paths


def run_on_multiple_gpus(
    cehrgpt_args, data_args, model_args, training_args, cuda_devices: List[str]
) -> int:
    """
    Tokenize the dataset once, split it into disjoint shards, and extract each on its own worker.

    The dataset is prepared here, without loading the model, so the workers don't race on
    the datasets cache. Each worker re-runs this module on one shard with CUDA_VISIBLE_DEVICES
    pinned to one GPU. The feature files have unique names, so all workers write into the same
    features_with_label folders.
    """
    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    for additional_token in [LINEAR_PROB_TOKEN, RANDOM_TOKEN]:
        if additional_token not in cehrgpt_tokenizer.get_vocab():
            cehrgpt_tokenizer.add_tokens(additional_token)
    processed_dataset, _, tokenizer_path = prepare_processed_dataset(
        cehrgpt_args, data_args, model_args, training_args, cehrgpt_tokenizer
    )

    non_empty_sizes = [
        len(dataset)
        for dataset in (
            concatenate_datasets(
                [
                    processed_dataset[s]
                    for s in ("train", "validation")
                    if s in processed_dataset
                ]
            ),
            processed_dataset["test"] if "test" in processed_dataset else [],
        )
        if len(dataset) > 0
    ]
    num_workers = max(1, min([len(cuda_devices)] + non_empty_sizes))
    cuda_devices = cuda_devices[:num_workers]
    LOG.info("Extracting features with %s workers on GPUs %s", num_workers, cuda_devices)
    if training_args.dataloader_num_workers:
        LOG.info(
            "Each worker also starts %s DataLoader processes (%s in total)",
            training_args.dataloader_num_workers,
            training_args.dataloader_num_workers * num_workers,
        )

    shard_root = os.path.join(training_args.output_dir, "temp", "shards")
    # Shards left over from an interrupted run are stale
    shutil.rmtree(shard_root, ignore_errors=True)
    shard_paths = shard_processed_dataset(processed_dataset, num_workers, shard_root)

    worker_argv = sys.argv[1:]
    for option in ("--gpu_ids", "--workers_per_gpu", "--tokenizer_name_or_path"):
        worker_argv = strip_cli_option(worker_argv, option)
    # The dataset was tokenized with this tokenizer, so the workers must load the same one
    worker_argv += ["--tokenizer_name_or_path", tokenizer_path]
    exit_codes, log_paths = run_workers(
        module="cehrgpt.tools.linear_prob.compute_cehrgpt_features",
        worker_argv=worker_argv,
        shard_option="--shard_dataset_path",
        shard_paths=shard_paths,
        cuda_devices=cuda_devices,
        log_folder=os.path.join(training_args.output_dir, "worker_logs"),
    )
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


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    is_shard_worker = cehrgpt_args.shard_dataset_path is not None
    if (cehrgpt_args.gpu_ids or cehrgpt_args.workers_per_gpu > 1) and not is_shard_worker:
        cuda_devices = expand_gpu_workers(
            resolve_gpu_ids(cehrgpt_args.gpu_ids or "0"), cehrgpt_args.workers_per_gpu
        )
        if len(cuda_devices) > 1:
            exit_code = run_on_multiple_gpus(
                cehrgpt_args, data_args, model_args, training_args, cuda_devices
            )
            if exit_code:
                sys.exit(exit_code)
            return
        # CUDA has not been initialized yet, so this pins the single requested GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices[0]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path
    )
    torch_dtype = get_torch_dtype(model_args.torch_dtype)
    cehrgpt_model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=resolve_attn_implementation(
                backbone=read_backbone(model_args.model_name_or_path),
                requested=cehrgpt_args.force_attn_implementation,
            ),
            torch_dtype=torch_dtype,
        )
        .eval()
        .to(device)
    )

    for additional_token in [LINEAR_PROB_TOKEN, RANDOM_TOKEN]:
        if additional_token not in cehrgpt_tokenizer.get_vocab():
            cehrgpt_tokenizer.add_tokens(additional_token)
        if cehrgpt_tokenizer.vocab_size > cehrgpt_model.config.vocab_size:
            cehrgpt_model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)

    if is_shard_worker:
        processed_dataset = load_from_disk(cehrgpt_args.shard_dataset_path)
    else:
        processed_dataset, cehrgpt_tokenizer, _ = prepare_processed_dataset(
            cehrgpt_args, data_args, model_args, training_args, cehrgpt_tokenizer
        )

    if (
        cehrgpt_model.config.max_position_embeddings
        < model_args.max_position_embeddings
    ):
        LOG.info(
            "Increase model.config.max_position_embeddings to %s",
            model_args.max_position_embeddings,
        )
        cehrgpt_model.config.max_position_embeddings = (
            model_args.max_position_embeddings
        )
        cehrgpt_model.resize_position_embeddings(model_args.max_position_embeddings)

    train_set = concatenate_datasets(
        [processed_dataset[s] for s in ("train", "validation") if s in processed_dataset]
    )

    if cehrgpt_args.sample_packing:
        per_device_eval_batch_size = 1
        data_collator_fn = partial(
            SamplePackingCehrGptDataCollator,
            cehrgpt_args.max_tokens_per_batch,
            cehrgpt_model.config.max_position_embeddings,
        )
        train_batch_sampler = SamplePackingBatchSampler(
            lengths=train_set["num_of_concepts"],
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=cehrgpt_model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
        test_batch_sampler = SamplePackingBatchSampler(
            lengths=processed_dataset["test"]["num_of_concepts"],
            max_tokens_per_batch=cehrgpt_args.max_tokens_per_batch,
            max_position_embeddings=cehrgpt_model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
    else:
        data_collator_fn = CehrGptDataCollator
        train_batch_sampler = None
        test_batch_sampler = None
        per_device_eval_batch_size = training_args.per_device_eval_batch_size

    # We suppress the additional learning objectives in fine-tuning
    data_collator = data_collator_fn(
        tokenizer=cehrgpt_tokenizer,
        max_length=(
            cehrgpt_args.max_tokens_per_batch
            if cehrgpt_args.sample_packing
            else model_args.max_position_embeddings
        ),
        include_values=cehrgpt_model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
        add_linear_prob_token=cehrgpt_args.add_random_token,
    )

    # prefetch_factor is only valid with worker processes; with 0 workers batches are built
    # in the main process, which leaves the GPU idle while the collator runs.
    prefetch_factor = (
        training_args.dataloader_prefetch_factor
        if training_args.dataloader_num_workers > 0
        else None
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=train_batch_sampler,
        prefetch_factor=prefetch_factor,
    )

    test_dataloader = DataLoader(
        dataset=processed_dataset["test"],
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=test_batch_sampler,
        prefetch_factor=prefetch_factor,
    )

    # Loading demographics
    print("Loading demographics as a dictionary")
    if cehrgpt_args.tokenized_full_dataset_path is not None:
        # data_folder/test_data_folder may not exist as separate train/test directories in
        # this mode (the train/test split comes from the pre-tokenized dataset instead), so
        # read demographics from cohort_folder, the same source extract_cohort_sequences uses.
        demographic_dirs = [data_args.cohort_folder]
    else:
        demographic_dirs = [data_args.data_folder, data_args.test_data_folder]
    demographics_df = pd.concat(
        [pd.read_parquet(data_dir) for data_dir in demographic_dirs]
    )
    # Auto-detect MEDS format from the actual schema, regardless of the is_data_in_meds flag
    is_meds = data_args.is_data_in_meds or "subject_id" in demographics_df.columns
    id_col = "subject_id" if is_meds else "person_id"
    if {"gender_concept_id", "race_concept_id"}.issubset(demographics_df.columns):
        demographics_df = demographics_df[
            [id_col, "gender_concept_id", "race_concept_id"]
        ].rename(columns={id_col: "person_id"})
        demographics_lookup = DemographicsLookup(demographics_df)
    else:
        demographics_lookup = DemographicsLookup(None)

    data_loaders = [("train", train_loader), ("test", test_dataloader)]
    # Write the parquet files from background threads so the GPU loop does not wait on disk
    writer = AsyncParquetWriter()
    # What the training needs to know to take the global or the local features out of the vectors
    num_features = None
    global_is_last_hidden_state = False

    for split, data_loader in data_loaders:
        feature_output_folder = (
            Path(training_args.output_dir) / "features_with_label" / f"{split}_features"
        )
        feature_output_folder.mkdir(parents=True, exist_ok=True)

        LOG.info("Generating features for %s set at %s", split, feature_output_folder)

        with torch.no_grad():
            for index, batch in enumerate(
                tqdm(data_loader, desc="Generating features")
            ):
                prediction_time_ages = _ensure_1d(
                    batch.pop("age_at_index").numpy().astype(float).squeeze()
                )
                person_ids = _ensure_1d(
                    batch.pop("person_id").numpy().astype(int).squeeze()
                )
                prediction_time_posix = _ensure_1d(
                    batch.pop("index_date").numpy().squeeze()
                )
                prediction_time = [
                    datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)
                    for t in prediction_time_posix
                ]
                labels = _ensure_1d(
                    batch.pop("classifier_label")
                    .float()
                    .cpu()
                    .numpy()
                    .astype(bool)
                    .squeeze()
                )

                # Right now the model does not support this column, we need to pop it
                if "epoch_times" in batch:
                    batch.pop("epoch_times")

                if "ages" in batch:
                    batch.pop("ages")

                batch = {k: v.to(device) for k, v in batch.items()}
                cehrgpt_output = cehrgpt_model(
                    **batch, output_attentions=False, output_hidden_states=True
                )
                # When the model was trained without MOTOR (include_motor_time_to_event=False),
                # linear_prob_hidden_states is None.  Fall back to the last transformer
                # hidden state in that case.
                linear_prob_hs = cehrgpt_output.linear_prob_hidden_states
                if linear_prob_hs is None:
                    linear_prob_hs = cehrgpt_output.hidden_states[-1]
                    global_is_last_hidden_state = True

                if cehrgpt_args.sample_packing:
                    features = (
                        extract_features_from_packed_sequence(
                            linear_prob_hs,
                            batch["attention_mask"],
                        )
                        .cpu()
                        .float()
                        .detach()
                        .numpy()
                        .squeeze(axis=0)
                    )
                    if cehrgpt_args.combine_global_local_features:
                        last_features = (
                            extract_features_from_packed_sequence(
                                cehrgpt_output.hidden_states[-1],
                                batch["attention_mask"],
                            )
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                            .squeeze(axis=0)
                        )
                        features = np.concatenate([features, last_features], axis=-1)
                else:
                    features = (
                        linear_prob_hs[..., -1, :]
                        .cpu()
                        .float()
                        .detach()
                        .numpy()
                    )
                    if cehrgpt_args.combine_global_local_features:
                        last_features = (
                            cehrgpt_output.hidden_states[-1][..., -1, :]
                            .cpu()
                            .float()
                            .detach()
                            .numpy()
                        )
                        features = np.concatenate([features, last_features], axis=-1)

                num_features = int(features.shape[-1])
                features_list = [feature for feature in features]
                gender_concept_ids, race_concept_ids = demographics_lookup.lookup(
                    person_ids
                )

                features_pd = pd.DataFrame(
                    {
                        "subject_id": person_ids,
                        "prediction_time": prediction_time,
                        "prediction_time_posix": prediction_time_posix,
                        "boolean_value": labels,
                        "age_at_index": prediction_time_ages,
                    }
                )
                features_pd["features"] = features_list
                features_pd["race_concept_id"] = race_concept_ids
                features_pd["gender_concept_id"] = gender_concept_ids
                writer.write(features_pd, feature_output_folder / f"{uuid.uuid4()}.parquet")

        writer.flush()

    writer.close()
    if num_features is not None:
        info = write_feature_info(
            Path(training_args.output_dir),
            num_features,
            combined=bool(cehrgpt_args.combine_global_local_features),
            global_is_last_hidden_state=global_is_last_hidden_state,
        )
        LOG.info("The layout of the features: %s", info)


if __name__ == "__main__":
    main()
