"""
Attention rollout of the linear probing features of a cohort.

For every sample of a cohort this writes the importance of each input token for the patient
representation at the prediction time (see ``attention_rollout`` for the method), and a
summary per concept over the samples of a split.

It takes the arguments of ``compute_cehrgpt_features`` and prepares the cohort the same way
(the prepared dataset of a feature extraction is reused), with these differences:

* The model is loaded with eager attention, the only one that computes attention weights.
* Sample packing is off whatever ``--sample_packing`` says: the eager attention of a packed row
  of 16k tokens does not fit in memory. A batch is ``--per_device_eval_batch_size`` samples.
* One process on one GPU (or the CPU): the first of ``--gpu_ids`` when it is given.

Outputs, under ``<output_dir>/attention_rollout_<target>/``:

* ``<split>/*.parquet``: one row per sample with ``subject_id``, ``prediction_time``,
  ``prediction_time_posix``, ``boolean_value`` and three lists aligned to the input tokens of
  the patient, in order: ``concept_ids`` (the tokens), ``importance`` (sums to 1) and
  ``baseline_importance`` (what a model attending uniformly gives the same positions).
* ``<split>_concept_importance.csv``: one row per concept, see ``ConceptImportanceAggregator``.
* ``attention_rollout_info.json``: the settings of the run.

One sample instead of the cohort: ``--attention_rollout_subject_id`` with, optionally,
``--attention_rollout_prediction_time`` (a date or a date and time, UTC as in the features).
Only the samples of that subject are run, and each gets a table of its tokens in order,
``samples/<subject_id>_<prediction time>.csv``, with the columns ``position``,
``concept_id``, ``importance``, ``baseline_importance`` and ``lift``. The outputs of the
cohort are left as they are.

``scripts/run_linear_prob.sh --attention_rollout`` runs this module on every cohort once its
features are extracted.
"""

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.special_tokens import LINEAR_PROB_TOKEN, RANDOM_TOKEN
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.multi_gpu_util import resolve_gpu_ids
from cehrgpt.tools.linear_prob.attention_rollout import (
    TokenImportance,
    compute_token_importance,
    uniform_attention_baseline,
)
from cehrgpt.tools.linear_prob.feature_io import AsyncParquetWriter

LOG = logging.get_logger("transformers")

# What the collator adds for the tool and the model does not take
METADATA_COLUMNS = ("age_at_index", "person_id", "index_date", "classifier_label")
UNSUPPORTED_MODEL_INPUTS = ("epoch_times", "ages")


def decode_tokens(tokenizer, token_ids: Sequence[int]) -> List[str]:
    """The tokens of the ids, one per id. The ids themselves if the tokenizer joins or drops any."""
    tokens = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    if len(tokens) != len(token_ids):
        return [str(token_id) for token_id in token_ids]
    return list(tokens)


def per_sample_token_importance(
    importance: TokenImportance,
    baseline: TokenImportance,
    input_ids: torch.Tensor,
    tokenizer,
) -> List[Dict[str, list]]:
    """
    The tokens of every patient of a batch with their importance, padding left out.

    One dictionary per patient, in the order of the batch (and of the packed row), with the
    lists ``concept_ids``, ``importance`` and ``baseline_importance`` aligned to the input
    tokens of that patient.
    """
    samples = []
    input_ids = input_ids.cpu()
    for index in range(importance.scores.shape[0]):
        token_mask = importance.token_mask[index].cpu()
        token_ids = input_ids[importance.batch_indices[index].item()][token_mask].tolist()
        samples.append(
            {
                "concept_ids": decode_tokens(tokenizer, token_ids),
                "importance": importance.scores[index].cpu()[token_mask].numpy().astype(np.float32),
                "baseline_importance": baseline.scores[index]
                .cpu()[token_mask]
                .numpy()
                .astype(np.float32),
            }
        )
    return samples


class ConceptImportanceAggregator:
    """
    Adds the token importance of the samples up per concept, for a global view of a cohort.

    The importance of a concept in a sample is the sum over its occurrences, so a sample
    gives every concept a share of its representation and the shares of a sample sum to 1.

    The columns of ``to_frame()``:

    * ``num_samples``: the samples the concept is in.
    * ``mean_importance``: the average share over all the samples (0 where it is absent).
      The column sums to 1.
    * ``mean_importance_when_present``: the average share over the samples it is in.
    * ``mean_baseline_when_present``: the same for a model attending uniformly, i.e. what the
      positions of the concept get whatever the concept is.
    * ``lift``: importance over baseline. Above 1, the model attends to the concept more than
      to its positions on average. Rollout favours the first tokens of a history, so this is
      the column to rank concepts on.
    * ``mean_importance_positive`` / ``mean_importance_negative``: ``mean_importance`` over the
      samples with a positive / negative label.
    """

    def __init__(self):
        self.num_samples = 0
        self.num_positive = 0
        self._counts = defaultdict(int)
        self._importance = defaultdict(float)
        self._baseline = defaultdict(float)
        self._positive = defaultdict(float)

    def add(
        self,
        concept_ids: Sequence[str],
        importance: Sequence[float],
        baseline_importance: Sequence[float],
        label: bool,
    ) -> None:
        per_concept = pd.DataFrame(
            {
                "concept_id": list(concept_ids),
                "importance": np.asarray(importance, dtype=np.float64),
                "baseline": np.asarray(baseline_importance, dtype=np.float64),
            }
        ).groupby("concept_id", sort=False)[["importance", "baseline"]].sum()
        self.num_samples += 1
        self.num_positive += int(bool(label))
        for concept_id, row in per_concept.iterrows():
            self._counts[concept_id] += 1
            self._importance[concept_id] += row["importance"]
            self._baseline[concept_id] += row["baseline"]
            if label:
                self._positive[concept_id] += row["importance"]

    def to_frame(self) -> pd.DataFrame:
        concept_ids = list(self._counts)
        counts = np.asarray([self._counts[c] for c in concept_ids], dtype=np.float64)
        importance = np.asarray([self._importance[c] for c in concept_ids], dtype=np.float64)
        baseline = np.asarray([self._baseline[c] for c in concept_ids], dtype=np.float64)
        positive = np.asarray([self._positive[c] for c in concept_ids], dtype=np.float64)
        num_negative = self.num_samples - self.num_positive
        with np.errstate(divide="ignore", invalid="ignore"):
            frame = pd.DataFrame(
                {
                    "concept_id": concept_ids,
                    "num_samples": counts.astype(np.int64),
                    "mean_importance": importance / max(self.num_samples, 1),
                    "mean_importance_when_present": importance / counts,
                    "mean_baseline_when_present": baseline / counts,
                    "lift": np.where(baseline > 0, importance / baseline, np.nan),
                    "mean_importance_positive": (
                        positive / self.num_positive if self.num_positive else np.nan
                    ),
                    "mean_importance_negative": (
                        (importance - positive) / num_negative if num_negative else np.nan
                    ),
                }
            )
        return frame.sort_values("mean_importance", ascending=False, ignore_index=True)


def parse_prediction_time(prediction_time: str) -> Tuple[float, float, bool]:
    """
    The POSIX interval a prediction time given on the command line stands for.

    A date is the whole day and a date and time that second, both in UTC, which is how the
    prediction times of a cohort are stored. The last value tells whether a time was given.
    """
    parsed = datetime.fromisoformat(prediction_time.strip())
    start = parsed.replace(tzinfo=timezone.utc).timestamp()
    has_time = len(prediction_time.strip()) > 10
    return start, start + (1.0 if has_time else 86400.0), has_time


def select_sample_indices(
    person_ids: Sequence[int],
    index_dates: Sequence[float],
    subject_id: int,
    prediction_time: Optional[str] = None,
) -> List[int]:
    """
    The rows of a dataset that are samples of the subject, at the prediction time if one is given.

    A prediction time with a time of day that no sample has falls back to the samples of
    that day: cohorts are often stored with the date only.
    """
    person_ids = np.asarray(person_ids).reshape(-1)
    index_dates = np.asarray(index_dates, dtype=np.float64).reshape(-1)
    of_subject = np.flatnonzero(person_ids == subject_id)
    if prediction_time is None or len(of_subject) == 0:
        return of_subject.tolist()
    start, end, has_time = parse_prediction_time(prediction_time)
    times = index_dates[of_subject]
    selected = of_subject[(times >= start) & (times < end)]
    if len(selected) == 0 and has_time:
        day = start - start % 86400.0
        selected = of_subject[(times >= day) & (times < day + 86400.0)]
        if len(selected):
            LOG.warning(
                "No sample of subject %s at %s, using the %s of that day",
                subject_id,
                prediction_time,
                len(selected),
            )
    return selected.tolist()


def token_importance_frame(sample: Dict[str, list]) -> pd.DataFrame:
    """The tokens of a sample in order, one per row, with their importance, baseline and lift."""
    importance = np.asarray(sample["importance"], dtype=np.float64)
    baseline = np.asarray(sample["baseline_importance"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(baseline > 0, importance / baseline, np.nan)
    return pd.DataFrame(
        {
            "position": np.arange(len(importance)),
            "concept_id": list(sample["concept_ids"]),
            "importance": importance,
            "baseline_importance": baseline,
            "lift": lift,
        }
    )


def load_eager_model(model_name_or_path: str, torch_dtype, device) -> CEHRGPT2LMHeadModel:
    """The model with the attention that computes its weights, whatever it was trained with."""
    return (
        CEHRGPT2LMHeadModel.from_pretrained(
            model_name_or_path, attn_implementation="eager", torch_dtype=torch_dtype
        )
        .eval()
        .to(device)
    )


def main():
    # Imported here: it pulls in the training runners, which the functions above do not need
    from cehrgpt.tools.linear_prob.compute_cehrgpt_features import (
        _ensure_1d,
        get_torch_dtype,
        prepare_processed_dataset,
    )

    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()
    if cehrgpt_args.gpu_ids:
        # CUDA has not been initialized yet, so this pins the first of the requested GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = resolve_gpu_ids(cehrgpt_args.gpu_ids)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cehrgpt_args.sample_packing:
        LOG.warning(
            "Sample packing is not used for the attention rollout: the eager attention of a "
            "packed row does not fit in memory. Batches are %s samples.",
            training_args.per_device_eval_batch_size,
        )
    if cehrgpt_args.force_attn_implementation not in (None, "eager"):
        LOG.warning(
            "--force_attn_implementation %s is ignored: only the eager attention computes "
            "the attention weights",
            cehrgpt_args.force_attn_implementation,
        )

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    cehrgpt_model = load_eager_model(
        model_args.model_name_or_path, get_torch_dtype(model_args.torch_dtype), device
    )
    for additional_token in [LINEAR_PROB_TOKEN, RANDOM_TOKEN]:
        if additional_token not in cehrgpt_tokenizer.get_vocab():
            cehrgpt_tokenizer.add_tokens(additional_token)
        if cehrgpt_tokenizer.vocab_size > cehrgpt_model.config.vocab_size:
            cehrgpt_model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)

    processed_dataset, cehrgpt_tokenizer, _ = prepare_processed_dataset(
        cehrgpt_args, data_args, model_args, training_args, cehrgpt_tokenizer
    )
    if cehrgpt_model.config.max_position_embeddings < model_args.max_position_embeddings:
        LOG.info(
            "Increase model.config.max_position_embeddings to %s",
            model_args.max_position_embeddings,
        )
        cehrgpt_model.config.max_position_embeddings = model_args.max_position_embeddings
        cehrgpt_model.resize_position_embeddings(model_args.max_position_embeddings)

    data_collator = CehrGptDataCollator(
        tokenizer=cehrgpt_tokenizer,
        max_length=model_args.max_position_embeddings,
        include_values=cehrgpt_model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=cehrgpt_args.include_demographics,
        add_linear_prob_token=cehrgpt_args.add_random_token,
    )
    splits = {
        "train": concatenate_datasets(
            [processed_dataset[s] for s in ("train", "validation") if s in processed_dataset]
        )
    }
    if "test" in processed_dataset:
        splits["test"] = processed_dataset["test"]

    subject_id: Optional[int] = cehrgpt_args.attention_rollout_subject_id
    if subject_id is None and cehrgpt_args.attention_rollout_prediction_time is not None:
        raise ValueError(
            "--attention_rollout_prediction_time needs --attention_rollout_subject_id"
        )
    if subject_id is not None:
        for split in list(splits):
            dataset = splits[split]
            splits[split] = dataset.select(
                select_sample_indices(
                    dataset["person_id"],
                    dataset["index_date"],
                    subject_id,
                    cehrgpt_args.attention_rollout_prediction_time,
                )
            )
        if not any(len(dataset) for dataset in splits.values()):
            raise ValueError(
                f"No sample of subject {subject_id} in the cohort"
                + (
                    f" at {cehrgpt_args.attention_rollout_prediction_time}"
                    if cehrgpt_args.attention_rollout_prediction_time
                    else ""
                )
            )

    target = cehrgpt_args.attention_rollout_target
    max_samples: Optional[int] = (
        cehrgpt_args.attention_rollout_max_samples if subject_id is None else None
    )
    rollout_kwargs = dict(
        residual_weight=cehrgpt_args.attention_rollout_residual_weight,
        probe_layer_weighting=cehrgpt_args.attention_rollout_probe_layer_weighting,
    )
    num_layers = len(cehrgpt_model.cehrgpt.h)
    has_probe = getattr(cehrgpt_model, "linear_prob", None) is not None
    num_probe_layers = (
        len(cehrgpt_model.linear_prob.linear_prob) if target == "global" and has_probe else 0
    )
    if target == "global" and not has_probe:
        LOG.warning(
            "The model has no linear probing module (it was trained without MOTOR): its "
            "global features are the last decoder state, which is what is rolled out."
        )

    output_folder = Path(training_args.output_dir) / f"attention_rollout_{target}"
    output_folder.mkdir(parents=True, exist_ok=True)
    if subject_id is not None:
        samples_folder = output_folder / "samples"
        samples_folder.mkdir(parents=True, exist_ok=True)
    writer = AsyncParquetWriter()
    num_samples_per_split = {}
    for split, dataset in splits.items():
        if len(dataset) == 0:
            continue
        split_folder = output_folder / split
        if subject_id is None:
            split_folder.mkdir(parents=True, exist_ok=True)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=data_collator,
            pin_memory=training_args.dataloader_pin_memory,
        )
        aggregator = ConceptImportanceAggregator()
        for batch in tqdm(data_loader, desc=f"Attention rollout of the {split} set"):
            metadata = {column: batch.pop(column) for column in METADATA_COLUMNS}
            for column in UNSUPPORTED_MODEL_INPUTS:
                batch.pop(column, None)
            person_ids = _ensure_1d(metadata["person_id"].numpy().astype(int).squeeze())
            prediction_time_posix = _ensure_1d(metadata["index_date"].numpy().squeeze())
            labels = _ensure_1d(
                metadata["classifier_label"].float().numpy().astype(bool).squeeze()
            )

            batch = {k: v.to(device) for k, v in batch.items()}
            importance = compute_token_importance(
                cehrgpt_model, batch, target=target, **rollout_kwargs
            )
            baseline = uniform_attention_baseline(
                batch["attention_mask"], num_layers, num_probe_layers, **rollout_kwargs
            )
            samples = per_sample_token_importance(
                importance, baseline, batch["input_ids"], cehrgpt_tokenizer
            )
            # Without sample packing a row is a patient, in the order of the batch
            assert len(samples) == len(person_ids), (len(samples), len(person_ids))
            if subject_id is not None:
                for sample, person_id, posix, label in zip(
                    samples, person_ids, prediction_time_posix, labels
                ):
                    time = datetime.fromtimestamp(posix, tz=timezone.utc)
                    sample_file = samples_folder / f"{person_id}_{time:%Y%m%dT%H%M%S}.csv"
                    token_importance_frame(sample).to_csv(sample_file, index=False)
                    LOG.info(
                        "Subject %s at %s (%s set, label %s, %s tokens): %s",
                        person_id,
                        time.replace(tzinfo=None),
                        split,
                        bool(label),
                        len(sample["concept_ids"]),
                        sample_file,
                    )
                continue
            for sample, label in zip(samples, labels):
                aggregator.add(
                    sample["concept_ids"],
                    sample["importance"],
                    sample["baseline_importance"],
                    bool(label),
                )
            rollout_pd = pd.DataFrame(
                {
                    "subject_id": person_ids,
                    "prediction_time": [
                        datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None)
                        for t in prediction_time_posix
                    ],
                    "prediction_time_posix": prediction_time_posix,
                    "boolean_value": labels,
                }
            )
            for column in ("concept_ids", "importance", "baseline_importance"):
                rollout_pd[column] = [sample[column] for sample in samples]
            writer.write(rollout_pd, split_folder / f"{uuid.uuid4()}.parquet")
            if max_samples is not None and aggregator.num_samples >= max_samples:
                break
        if subject_id is not None:
            continue
        writer.flush()
        aggregator.to_frame().to_csv(
            output_folder / f"{split}_concept_importance.csv", index=False
        )
        num_samples_per_split[split] = aggregator.num_samples
    writer.close()
    if subject_id is not None:
        # The outputs of the cohort, attention_rollout_info.json included, are left as they are
        return

    info = {
        "target": target,
        "model_name_or_path": model_args.model_name_or_path,
        "num_decoder_layers": num_layers,
        "num_probe_layers": num_probe_layers,
        "head_fusion": "mean",
        "num_samples": num_samples_per_split,
        **rollout_kwargs,
    }
    with open(output_folder / "attention_rollout_info.json", "w") as f:
        json.dump(info, f, indent=2)
    LOG.info("The attention rollout is in %s: %s", output_folder, info)


if __name__ == "__main__":
    main()
