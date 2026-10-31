"""
Overlap (positivity) check for the drug swap.

A swapped future is only evidence about the comparator drug when the model finds
that drug plausible for this patient.  For each patient we read the model's
next-token distribution at the index position, given the history before the
index drug, and compare the probability mass of the two drug classes::

    preference_score = P(comparator class) / (P(source class) + P(comparator class))

A score near 0 or 1 means the patient would (almost) never start the other drug,
so the counterfactual would be an extrapolation.  Patients outside
``[trim, 1 - trim]`` are dropped.  The band is symmetric, so a patient passes in
one swap direction exactly when it passes in the other.

The source drug and its comparator share the same prefix, so a single forward
pass per patient gives both probabilities.
"""

from typing import Any, Dict, List, Set

import polars as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.utils import is_flash_attn_2_available

from cehrgpt.analysis.counterfactual.generate_counterfactual_sequences import (
    _SEQUENCE_ARRAY_COLS,
    _left_truncate_with_demographics,
)
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer

SCORE_COLUMNS = ["person_id", "p_source", "p_comparator", "preference_score", "retained"]


def class_token_ids(vocab: Dict[str, int], concepts: Set[str], label: str) -> List[int]:
    """Token ids of the concepts of a drug class that the tokenizer knows."""
    ids = sorted({int(vocab[c]) for c in concepts if c in vocab})
    if not ids:
        raise ValueError(
            f"None of the {len(concepts)} {label} concepts is in the tokenizer vocabulary, "
            f"so the overlap check cannot score patients."
        )
    return ids


def index_prefix_frame(
    df: pl.DataFrame, source_concepts: Set[str], context_length: int, tokenizer: Any
) -> pl.DataFrame:
    """
    Cut every context just before its index drug, then left-truncate it the way
    generation does.  The next token the model predicts after the cut is the
    drug.  Contexts without an index drug, or with nothing before it, are dropped.
    """
    columns = [c for c in _SEQUENCE_ARRAY_COLS if c in df.columns]
    rows = []
    for row in df.iter_rows(named=True):
        concept_ids = list(row["concept_ids"])
        position = next((i for i, c in enumerate(concept_ids) if c in source_concepts), None)
        if not position:
            continue
        prefix = dict(row)
        for column in columns:
            values = row[column]
            if values is not None and len(values) == len(concept_ids):
                prefix[column] = list(values[:position])
        rows.append(prefix)
    if not rows:
        return df.clear()
    return _left_truncate_with_demographics(
        pl.DataFrame(rows, schema=df.schema), max_length=context_length, tokenizer=tokenizer
    )


def load_scoring_model(model_path: str, tokenizer: CehrGptTokenizer, device: torch.device):
    model = (
        CEHRGPT2LMHeadModel.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager",
        )
        .eval()
        .to(device)
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.end_token_id
    model.generation_config.bos_token_id = tokenizer.end_token_id
    return model


def next_token_probabilities(
    model: CEHRGPT2LMHeadModel,
    tokenizer: CehrGptTokenizer,
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Softmax of the model's next-token logits for each prefix in *batch*, from the
    same ``generate`` path that produces the trajectories (padding, ages and
    values are handled identically).  The logits are the raw ones, before any
    generation-time processing.
    """
    input_ids = batch["input_ids"].to(device)
    config = GenerationConfig(
        max_length=input_ids.shape[1] + 1,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
        bos_token_id=tokenizer.end_token_id,
        eos_token_id=tokenizer.end_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    values = batch.get("values")
    value_indicators = batch.get("value_indicators")
    with torch.no_grad():
        output = model.generate(
            inputs=input_ids,
            ages=batch["ages"].to(device),
            values=values.to(device) if values is not None else None,
            value_indicators=(
                value_indicators.to(device) if value_indicators is not None else None
            ),
            generation_config=config,
            cehrgpt_tokenizer=tokenizer,
        )
    return torch.softmax(output.logits[0].float(), dim=-1)


def score_patients(
    df: pl.DataFrame,
    source_concepts: Set[str],
    comparator_concepts: Set[str],
    tokenizer: CehrGptTokenizer,
    model: CEHRGPT2LMHeadModel,
    device: torch.device,
    context_length: int,
    batch_size: int,
) -> pl.DataFrame:
    """
    ``person_id, p_source, p_comparator`` for each context in *df* that has an
    index drug with history before it.  The two probabilities are the model's
    mass over the tokens of each drug class at the index position.
    """
    vocab = tokenizer.get_vocab()
    source_ids = class_token_ids(vocab, source_concepts, "source")
    comparator_ids = class_token_ids(vocab, comparator_concepts, "comparator")

    prefixes = index_prefix_frame(df, source_concepts, context_length, tokenizer)
    if prefixes.is_empty():
        return pl.DataFrame(
            schema={"person_id": pl.Int64, "p_source": pl.Float64, "p_comparator": pl.Float64}
        )

    collator = CehrGptDataCollator(
        tokenizer=tokenizer,
        max_length=context_length,
        include_values=model.config.include_values,
        pretraining=False,
        include_ttv_prediction=False,
        use_sub_time_tokenization=False,
        include_demographics=False,
        add_linear_prob_token=False,
    )
    loader = DataLoader(
        Dataset.from_pandas(prefixes.to_pandas()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    person_ids: List[int] = []
    p_source: List[float] = []
    p_comparator: List[float] = []
    for batch in tqdm(loader, desc="overlap check"):
        probabilities = next_token_probabilities(model, tokenizer, batch, device)
        person_ids.extend(batch["person_id"].reshape(-1).tolist())
        p_source.extend(probabilities[:, source_ids].sum(dim=1).cpu().tolist())
        p_comparator.extend(probabilities[:, comparator_ids].sum(dim=1).cpu().tolist())
    return pl.DataFrame(
        {"person_id": person_ids, "p_source": p_source, "p_comparator": p_comparator},
        schema={"person_id": pl.Int64, "p_source": pl.Float64, "p_comparator": pl.Float64},
    )


def trim_by_preference_score(
    person_ids: List[int], probabilities: pl.DataFrame, trim: float
) -> pl.DataFrame:
    """
    One row per patient in *person_ids* with the preference score and whether it
    lies in ``[trim, 1 - trim]``.  A patient that could not be scored (no index
    drug with history, or both class probabilities zero) is not retained.
    """
    if not 0.0 < trim < 0.5:
        raise ValueError(f"preference score trim must be in (0, 0.5), got {trim}")
    total = pl.col("p_source") + pl.col("p_comparator")
    return (
        pl.DataFrame({"person_id": person_ids}, schema={"person_id": pl.Int64})
        .join(probabilities, on="person_id", how="left")
        .with_columns(
            pl.when(total > 0)
            .then(pl.col("p_comparator") / total)
            .otherwise(None)
            .alias("preference_score")
        )
        .with_columns(
            (pl.col("preference_score").is_between(trim, 1.0 - trim))
            .fill_null(False)
            .alias("retained")
        )
        .select(SCORE_COLUMNS)
    )


def print_overlap_summary(scores: pl.DataFrame, trim: float) -> None:
    n = len(scores)
    kept = int(scores["retained"].sum())
    unscored = int(scores["preference_score"].is_null().sum())
    print(f"Overlap check (preference score in [{trim:.3f}, {1.0 - trim:.3f}]):")
    print(f"  retained {kept:,} of {n:,} patients; dropped {n - kept:,} ({unscored:,} could not be scored)")
    series = scores["preference_score"].drop_nulls()
    if len(series):
        print(
            "  preference score quantiles: "
            + ", ".join(f"p{int(q * 100)}={series.quantile(q):.3f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95))
        )
