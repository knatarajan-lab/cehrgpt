"""
Attention rollout of the linear probing features of a cohort, with the arguments of feature extraction.

A saved sample is not run again. See --attention_rollout_max_samples and --attention_rollout_subject_id.
"""

import glob
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN, RANDOM_TOKEN
from cehrgpt.runners.data_utils import get_torch_dtype
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.multi_gpu_util import resolve_gpu_ids
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer
from cehrgpt.tools.linear_prob.attention_rollout import (
    compute_token_importance,
    uniform_attention_baseline,
)
from cehrgpt.tools.linear_prob.compute_cehrgpt_features import (
    _load_or_create_prepared_dataset,
    rows_without_features,
    select_rows,
)
from cehrgpt.tools.linear_prob.feature_io import AsyncParquetWriter

LOG = logging.get_logger("transformers")


class ConceptImportanceAggregator:
    """
    Adds the token importance of the samples up per concept: the sum over its occurrences in a sample.

    lift, the importance over the uniform baseline, is the column to rank concepts on.
    """

    def __init__(self):
        self.num_samples = 0
        self.num_positive = 0
        # Per concept: the samples it is in, its importance, its baseline, its importance in positive samples
        self._totals = defaultdict(lambda: np.zeros(4))

    def add(
        self,
        concept_ids: Sequence[str],
        importance: Sequence[float],
        baseline_importance: Sequence[float],
        label: bool,
    ) -> None:
        per_concept = (
            pd.DataFrame(
                {
                    "concept_id": list(concept_ids),
                    "importance": np.asarray(importance, dtype=np.float64),
                    "baseline": np.asarray(baseline_importance, dtype=np.float64),
                }
            )
            .groupby("concept_id", sort=False)
            .sum()
        )
        self.num_samples += 1
        self.num_positive += int(label)
        for concept_id, row in per_concept.iterrows():
            self._totals[concept_id] += (1, row["importance"], row["baseline"], row["importance"] * label)

    def to_frame(self) -> pd.DataFrame:
        counts, importance, baseline, positive = np.array(list(self._totals.values())).reshape(-1, 4).T
        num_negative = self.num_samples - self.num_positive
        frame = pd.DataFrame(
            {
                "concept_id": list(self._totals),
                "num_samples": counts.astype(np.int64),
                "mean_importance": importance / self.num_samples,
                "mean_importance_when_present": importance / counts,
                "mean_baseline_when_present": baseline / counts,
                "lift": importance / baseline,
                "mean_importance_positive": positive / self.num_positive if self.num_positive else np.nan,
                "mean_importance_negative": (importance - positive) / num_negative if num_negative else np.nan,
            }
        )
        return frame.sort_values("mean_importance", ascending=False, ignore_index=True)


def select_samples(
    splits: DatasetDict, subject_ids: Optional[List[int]], max_samples: Optional[int], seed: int
) -> DatasetDict:
    """The samples of the subjects, then a random max_samples of every split, in the stored order."""
    selected = {}
    for split, dataset in splits.items():
        if subject_ids:
            dataset = dataset.select(np.flatnonzero(np.isin(dataset["person_id"], subject_ids)))
        if max_samples:
            # The start of a seeded shuffle: a later run with more samples extends the same set
            shuffled = np.random.default_rng(seed).permutation(len(dataset))
            dataset = dataset.select(np.sort(shuffled[:max_samples]))
        selected[split] = dataset
    return DatasetDict(selected)


def write_rollout(model, tokenizer, data_loader, target: str, num_probe_layers: int, folder: Path) -> None:
    """Writes the token importance of the samples of the loader into the folder, a file per batch."""
    with AsyncParquetWriter() as writer:
        for batch in tqdm(data_loader, desc=f"Attention rollout into {folder}"):
            person_ids = batch.pop("person_id").numpy().astype(int).reshape(-1)
            prediction_time_posix = batch.pop("index_date").numpy().reshape(-1)
            labels = batch.pop("classifier_label").numpy().astype(bool).reshape(-1)
            # The model does not take these
            for column in ("age_at_index", "epoch_times", "ages"):
                batch.pop(column, None)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            real = batch["attention_mask"].bool().cpu()
            importance = compute_token_importance(model, batch, target=target).cpu()
            baseline = uniform_attention_baseline(
                batch["attention_mask"], len(model.cehrgpt.h), num_probe_layers
            ).cpu()
            rollout_pd = pd.DataFrame(
                {
                    "subject_id": person_ids,
                    "prediction_time": [
                        datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)
                        for t in prediction_time_posix
                    ],
                    "prediction_time_posix": prediction_time_posix,
                    "boolean_value": labels,
                    # Aligned to the input tokens of the patient, padding left out
                    "concept_ids": [
                        tokenizer.convert_ids_to_tokens(ids[mask].tolist())
                        for ids, mask in zip(batch["input_ids"].cpu(), real)
                    ],
                    "importance": [row[mask].numpy() for row, mask in zip(importance, real)],
                    "baseline_importance": [row[mask].numpy() for row, mask in zip(baseline, real)],
                }
            )
            writer.write(rollout_pd, folder / f"{uuid.uuid4()}.parquet")


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    if cehrgpt_args.gpu_ids:
        # CUDA has not been initialized yet, so this pins the first of the requested GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = resolve_gpu_ids(cehrgpt_args.gpu_ids)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = cehrgpt_args.attention_rollout_target
    output_folder = Path(training_args.output_dir) / f"attention_rollout_{target}"
    info_file = output_folder / "attention_rollout_info.json"
    # A run adds to the samples that are saved, so they must come from the same model
    model_path = os.path.abspath(os.path.expanduser(model_args.model_name_or_path))
    if info_file.exists() and json.loads(info_file.read_text())["model_name_or_path"] != model_path:
        raise ValueError(f"{output_folder} holds the rollout of another model")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    # Only the eager attention computes the attention weights
    cehrgpt_model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="eager",
            torch_dtype=get_torch_dtype(model_args.torch_dtype),
        )
        .eval()
        .to(device)
    )
    for additional_token in [LINEAR_PROB_TOKEN, RANDOM_TOKEN]:
        if additional_token not in cehrgpt_tokenizer.get_vocab():
            cehrgpt_tokenizer.add_tokens(additional_token)
        if cehrgpt_tokenizer.vocab_size > cehrgpt_model.config.vocab_size:
            cehrgpt_model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)

    processed_dataset, cehrgpt_tokenizer, _, _ = _load_or_create_prepared_dataset(
        cehrgpt_args, data_args, model_args, training_args, cehrgpt_tokenizer
    )
    if cehrgpt_model.config.max_position_embeddings < model_args.max_position_embeddings:
        cehrgpt_model.config.max_position_embeddings = model_args.max_position_embeddings
        cehrgpt_model.resize_position_embeddings(model_args.max_position_embeddings)
    data_collator = CehrGptDataCollator(
        tokenizer=cehrgpt_tokenizer,
        # Where feature extraction cuts a patient: with sample packing that is the length of the model
        max_length=(
            cehrgpt_model.config.max_position_embeddings
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

    splits = DatasetDict(
        {
            "train": concatenate_datasets(
                [processed_dataset[s] for s in ("train", "validation") if s in processed_dataset]
            )
        }
    )
    if "test" in processed_dataset:
        splits["test"] = processed_dataset["test"]
    splits = select_samples(
        splits,
        cehrgpt_args.attention_rollout_subject_id,
        cehrgpt_args.attention_rollout_max_samples,
        training_args.seed,
    )
    if not sum(len(dataset) for dataset in splits.values()):
        raise ValueError("No sample of the cohort is selected")
    saved_files = os.path.join("*", "*.parquet")
    splits = select_rows(splits, rows_without_features(splits, str(output_folder), saved_files))

    num_probe_layers = (
        len(cehrgpt_model.linear_prob.linear_prob)
        if target == "global" and cehrgpt_model.linear_prob is not None
        else 0
    )
    num_samples = {}
    for split, dataset in splits.items():
        split_folder = output_folder / split
        if not len(dataset) and not split_folder.exists():
            continue
        split_folder.mkdir(parents=True, exist_ok=True)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=data_collator,
            pin_memory=training_args.dataloader_pin_memory,
        )
        write_rollout(cehrgpt_model, cehrgpt_tokenizer, data_loader, target, num_probe_layers, split_folder)
        # Over every saved sample of the split, whichever run wrote it
        aggregator = ConceptImportanceAggregator()
        for file in glob.glob(str(split_folder / "*.parquet")):
            for row in pd.read_parquet(file).itertuples():
                aggregator.add(row.concept_ids, row.importance, row.baseline_importance, row.boolean_value)
        aggregator.to_frame().to_csv(output_folder / f"{split}_concept_importance.csv", index=False)
        num_samples[split] = aggregator.num_samples

    info = {
        "target": target,
        "model_name_or_path": model_path,
        "num_decoder_layers": len(cehrgpt_model.cehrgpt.h),
        "num_probe_layers": num_probe_layers,
        "num_samples": num_samples,
    }
    info_file.write_text(json.dumps(info, indent=2))
    LOG.info("The attention rollout is in %s: %s", output_folder, info)


if __name__ == "__main__":
    main()
