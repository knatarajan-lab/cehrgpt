#!/usr/bin/env python3
"""
Step 1: Extract patient partial histories up to first drug exposure.

For each patient in the patient_sequence dataset this script:
  1. Expands the user-supplied drug ingredient concept_ids to all OMOP
     descendants via the concept_ancestor table.
  2. Scans each patient's concept_ids list to find the first visit that
     contains any of these drug concepts.
  3. Produces two truncated sequence variants:

       non_treated_context  — tokens up to (but NOT including) the ATT
                              token that precedes the drug-initiation visit,
                              i.e. the full history ending with [VE] of the
                              visit immediately before the drug visit.
                              The model will generate a counterfactual future
                              *without* the drug.

       treated_context      — tokens up to and INCLUDING [VE] of the
                              drug-initiation visit.  The model will generate
                              the future *conditional on* having received the
                              drug.

  4. Writes chunked parquet files to sub-directories of <output_dir>:
       non_treated_context/
       treated_context/
       drug_info/           (person_id, drug_concept_id, drug_epoch_time, arm)
       observed_outcomes/   (person_id, drug_epoch_time, <one col per outcome>)

Usage
-----
python extract_drug_initiation_sequences.py \\
    --patient_sequence_path /path/to/patient_sequence \\
    --vocab_path            /path/to/omop_vocab_dir \\
    --source_concept_ids    1308216,1346654 \\
    --comparator_concept_ids 974166,978555 \\
    --output_dir            /path/to/output
"""

import argparse
import glob as glob_module
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

from cehrgpt.models.tokenization_hf_cehrgpt import NONE_BIN, UNKNOWN_BIN, CehrGptTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VISIT_START_TOKEN = "[VS]"
VISIT_END_TOKEN = "[VE]"
END_TOKEN = "[END]"

# All columns that are parallel arrays (length == len(concept_ids))
SEQUENCE_ARRAY_COLS = [
    "concept_ids",
    "input_ids",
    "value_indicators",
    "values",
    "visit_segments",
    "orders",
    "dates",
    "ages",
    "visit_concept_orders",
    "concept_value_masks",
    "number_as_values",
    "concept_as_values",
    "is_numeric_types",
    "concept_values",
    "mlm_skip_values",
    "priorities",
    "visit_concept_ids",
    "visit_rank_orders",
    "concept_orders",
    "record_ranks",
    "units",
    "event_group_ids",
    "epoch_times",
]

# Columns that contain nulls and must be dropped before writing parquet
_DROP_BEFORE_WRITE = {"number_as_values", "concept_as_values", "is_numeric_types"}

# Columns the model's data collator requires to be present
REQUIRED_COLS = {"concept_ids", "input_ids", "ages", "epoch_times", "value_indicators", "values"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_values(record: Dict[str, Any], tokenizer: CehrGptTokenizer) -> Dict[str, Any]:
    """
    Compute value_indicators and values from raw source columns, mirroring
    the logic in HFCehrGptTokenizationMapping.transform().

    Sets record["value_indicators"] and record["values"] in-place and removes
    the intermediate columns that contain nulls (not safe for parquet write).
    """
    concept_value_masks = _to_list(record.get("concept_value_masks", []))
    record["value_indicators"] = concept_value_masks

    n = len(concept_value_masks)
    number_as_values  = _to_list(record.get("number_as_values",  [None] * n))
    concept_as_values = _to_list(record.get("concept_as_values", [None] * n))
    is_numeric_types  = _to_list(record.get("is_numeric_types",  [0]    * n))
    concept_ids       = _to_list(record["concept_ids"])
    units             = _to_list(record.get("units", [None] * n))

    if any(m == 1 for m in concept_value_masks):
        values = []
        for concept_id, unit, mask, num_val, cat_val, is_num in zip(
            concept_ids, units, concept_value_masks,
            number_as_values, concept_as_values, is_numeric_types,
        ):
            if mask == 1:
                value = UNKNOWN_BIN
                if is_num == 1:
                    if concept_id in tokenizer.numeric_concept_ids:
                        value = tokenizer.normalize(concept_id, unit, num_val)
                elif isinstance(cat_val, str):
                    value = cat_val
                values.append(value)
            else:
                values.append(NONE_BIN)
    else:
        values = [NONE_BIN] * n

    record["values"] = tokenizer.encode_value(values)

    for col in _DROP_BEFORE_WRITE:
        record.pop(col, None)

    return record


def find_outcomes_after_drug(
    concept_ids: List[str],
    epoch_times: List[float],
    start_pos: int,
    outcome_concept_groups: Dict[str, Set[str]],
) -> Dict[str, Optional[float]]:
    """
    Scan *concept_ids[start_pos:]* for the first occurrence of each outcome
    concept group.

    Returns
    -------
    Dict mapping each outcome_label to the epoch_time of its first occurrence
    after *start_pos*, or None if not observed.
    """
    result: Dict[str, Optional[float]] = {k: None for k in outcome_concept_groups}
    remaining = set(outcome_concept_groups.keys())

    for i in range(start_pos, len(concept_ids)):
        if not remaining:
            break
        token = concept_ids[i]
        for label in list(remaining):
            if token in outcome_concept_groups[label]:
                result[label] = float(epoch_times[i])
                remaining.discard(label)
                break

    return result


def _is_att_token(token: str) -> bool:
    """Return True if *token* is an inter-visit time-delta (ATT) token."""
    if token in (VISIT_START_TOKEN, VISIT_END_TOKEN):
        return False
    # Standard ATT tokens: LT, D1, D7, D30, D154 …
    if token == "LT":
        return True
    if token.startswith("D") and token[1:].isdigit():
        return True
    # Inpatient-hour or inpatient-day tokens: i-H5, i-D3, VS-D1-VE …
    if token.startswith("i-") or token.startswith("VS-"):
        return True
    return False


def load_drug_descendant_concepts(vocab_path: str, drug_ingredient_ids: List[int]) -> Set[str]:
    """
    Return all OMOP concept_ids (as strings) that are descendants of
    any of the supplied *drug_ingredient_ids*, including the ingredients
    themselves.  Uses the concept_ancestor table with any separation level.
    """
    ancestor_glob = os.path.join(vocab_path, "concept_ancestor", "*.parquet")
    lf = pl.scan_parquet(ancestor_glob)

    descendant_ids: List[str] = (
        lf
        .filter(pl.col("ancestor_concept_id").is_in(drug_ingredient_ids))
        .select(pl.col("descendant_concept_id").cast(pl.String))
        .collect()
        ["descendant_concept_id"]
        .to_list()
    )

    result: Set[str] = set(descendant_ids)
    # Always include the ingredient concepts themselves
    result.update(str(d) for d in drug_ingredient_ids)
    return result


def find_first_drug_initiation(
    concept_ids: List[str],
    drug_concepts: Set[str],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """
    Scan *concept_ids* and find the first occurrence of any concept in
    *drug_concepts* that appears inside a visit block.

    Returns
    -------
    (drug_pos, vs_pos, ve_pos, drug_concept_id)
      drug_pos        : index of the drug token in concept_ids
      vs_pos          : index of the [VS] that opens the drug visit
      ve_pos          : index of the [VE] that closes the drug visit
                        (None if the sequence ends before [VE])
      drug_concept_id : the specific drug concept string found
    Returns (None, None, None, None) when no drug exposure is found.
    """
    current_vs_pos: Optional[int] = None

    for i, token in enumerate(concept_ids):
        if token == VISIT_START_TOKEN:
            current_vs_pos = i
        elif token == VISIT_END_TOKEN:
            current_vs_pos = None
        elif token in drug_concepts:
            # Find the closing [VE] for the current visit
            ve_pos: Optional[int] = None
            for j in range(i + 1, len(concept_ids)):
                if concept_ids[j] == VISIT_END_TOKEN:
                    ve_pos = j
                    break
                if concept_ids[j] == VISIT_START_TOKEN:
                    # Next visit opened before [VE] — guard against malformed seqs
                    ve_pos = j - 1
                    break
            return i, current_vs_pos, ve_pos, token

    return None, None, None, None


def compute_drug_era_end_time(
    concept_ids: List[str],
    epoch_times: List[float],
    target_concepts: Set[str],
    competing_concepts: Optional[Set[str]],
    start_drug_pos: int,
    era_gap_days: int = 30,
) -> Optional[float]:
    """
    Compute the end epoch_time of the drug era that starts at *start_drug_pos*.

    The era is defined as a continuous period of drug exposure where consecutive
    drug events occur within *era_gap_days* days.  Follow-up ends at the EARLIEST
    of three conditions:

    1. **Era gap**: a gap > *era_gap_days* days appears between consecutive target
       drug events.  Era end = epoch_time of the last target event + era_gap_seconds.
    2. **Treatment switch**: a concept from *competing_concepts* is encountered in a
       post-initiation visit.  Era end = epoch_time of the switching event.
    3. **Sequence end**: the ``[END]`` token or the end of the concept_ids list is
       reached without a gap or switch.  Returns ``None`` (right-censored; follow-up
       runs to *follow_up_days*).

    Parameters
    ----------
    concept_ids      : token sequence
    epoch_times      : parallel array of epoch timestamps in **seconds**
    target_concepts  : OMOP concept_ids (strings) of the patient's current drug
                       class (source for ACEi patients, comparator for thiazide)
    competing_concepts : OMOP concept_ids (strings) of any drug that marks a
                       treatment switch; typically the opposite arm plus all other
                       first-line antihypertensives from the exclusion list.
                       Pass ``None`` or an empty set to disable switch detection.
    start_drug_pos   : index in concept_ids of the first drug event (era start)
    era_gap_days     : persistence window in days (default: 30)

    Returns
    -------
    float  — epoch_time (seconds) at which the era ends, or
    None   — patient is right-censored (sequence ended without discontinuation)
    """
    era_gap_seconds = era_gap_days * 86_400
    if start_drug_pos >= len(epoch_times):
        return None

    last_target_time = float(epoch_times[start_drug_pos])
    competing: Set[str] = competing_concepts or set()

    # Skip through the rest of the sequence tracking visits
    current_vs_pos: Optional[int] = None
    in_start_visit = True   # we're still inside the visit that contains the era start

    for i in range(start_drug_pos + 1, len(concept_ids)):
        token = concept_ids[i]

        if token == END_TOKEN:
            # Sequence ended normally — right-censored
            return None

        if token == VISIT_START_TOKEN:
            current_vs_pos = i
            in_start_visit = False
            continue

        if token == VISIT_END_TOKEN:
            if in_start_visit:
                in_start_visit = False
            current_vs_pos = None
            continue

        # Only process clinical events inside a visit (not ATT tokens between visits)
        if in_start_visit or current_vs_pos is None:
            continue

        event_time = float(epoch_times[i]) if i < len(epoch_times) else None
        if event_time is None:
            continue

        if token in target_concepts:
            gap = event_time - last_target_time
            if gap > era_gap_seconds:
                # Gap exceeded — era ended before this event
                return last_target_time + era_gap_seconds
            # Still within the era — extend last known target event time
            last_target_time = event_time

        elif token in competing:
            # Treatment switch — follow-up ends at the switching event
            return event_time

    # Reached end of sequence without gap or switch — right-censored
    return None


def _to_list(val: Any) -> List:
    """Convert a numpy array or other sequence to a plain Python list."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    return list(val)


def _truncate_arrays(
    row: Dict[str, Any],
    cut_pos: int,
    available_cols: List[str],
) -> Dict[str, Any]:
    """Return a dict where every array column in *available_cols* is
    sliced to ``[:cut_pos]``.  Scalar columns are passed through as-is."""
    result: Dict[str, Any] = {}
    for col in available_cols:
        val = row.get(col)
        if val is None:
            result[col] = []
        elif isinstance(val, (str, bytes, int, float, bool)):
            result[col] = val
        else:
            result[col] = _to_list(val)[:cut_pos]
    return result


def build_non_treated_context(
    row: Dict[str, Any],
    vs_pos: int,
    available_cols: List[str],
) -> Dict[str, Any]:
    """
    Truncate the sequence so it ends just *before* the ATT token that
    precedes the drug-initiation visit (or just before [VS] if there is
    no ATT token, e.g. the drug visit is the patient's very first visit).

    The resulting context represents the patient's history *before* any
    exposure to the drug.
    """
    concept_ids = _to_list(row["concept_ids"])

    # Walk backwards from vs_pos to find the preceding ATT token
    cut_pos = vs_pos
    if cut_pos > 0 and _is_att_token(concept_ids[cut_pos - 1]):
        cut_pos -= 1  # exclude the ATT token as well

    return _truncate_arrays(row, cut_pos, available_cols)


def build_treated_context(
    row: Dict[str, Any],
    ve_pos: Optional[int],
    available_cols: List[str],
) -> Dict[str, Any]:
    """
    Truncate the sequence so it ends just *after* [VE] of the
    drug-initiation visit (inclusive).

    The resulting context represents the patient's history up to and
    including the drug administration.
    """
    concept_ids = _to_list(row["concept_ids"])
    if ve_pos is None:
        cut_pos = len(concept_ids)   # use the full available sequence
    else:
        cut_pos = ve_pos + 1         # inclusive of [VE]

    return _truncate_arrays(row, cut_pos, available_cols)


# ---------------------------------------------------------------------------
# Shard processor (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_shard_star(args: tuple) -> Tuple[int, int]:
    """Unpack a single tuple so imap_unordered can call _process_shard."""
    return _process_shard(*args)  # population_concepts passed as keyword via last positional slot


def _process_shard(
    shard_id: int,
    shard_files: List[str],
    needed_cols: List[str],
    source_concepts: Set[str],
    comparator_concepts: Set[str],
    drug_concepts: Set[str],
    available_array_cols: List[str],
    scalar_cols: List[str],
    outcome_concept_groups: Dict[str, Set[str]],
    non_treated_dir: str,
    treated_dir: str,
    drug_info_dir: str,
    observed_outcomes_dir: str,
    tokenizer_path: Optional[str],
    min_context_length: int,
    chunk_size: int,
    population_concepts: Optional[Set[str]] = None,
    exclusion_concepts: Optional[Set[str]] = None,
    era_gap_days: int = 30,
) -> Tuple[int, int]:
    """
    Process one shard of patient rows and write chunked parquet files.

    Each worker receives a list of parquet files assigned exclusively to it,
    so there is no I/O contention between workers.

    Chunk files are named ``shard_{shard_id:03d}_chunk_{chunk_idx:05d}.parquet``
    so that outputs from parallel shards never collide.

    Returns
    -------
    (n_found, n_skipped)
    """
    # Read only the files assigned to this shard — no overlap with other workers
    print(
        f"[shard {shard_id:03d}] reading {len(shard_files)} file(s) …",
        flush=True,
    )
    t0 = time.time()
    df_shard = pl.read_parquet(shard_files, columns=needed_cols)
    print(
        f"[shard {shard_id:03d}] loaded {len(df_shard):,} rows in {time.time() - t0:.1f}s",
        flush=True,
    )

    # Load tokenizer in the worker process (not picklable across processes)
    tokenizer: Optional[CehrGptTokenizer] = None
    if tokenizer_path:
        tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_path)

    non_treated_chunk: List[Dict[str, Any]] = []
    treated_chunk: List[Dict[str, Any]] = []
    drug_info_chunk: List[Dict[str, Any]] = []
    observed_outcomes_chunk: List[Dict[str, Any]] = []

    n_found = 0
    n_skipped = 0
    chunk_idx = 0

    def _write_atomic(df: pl.DataFrame, final_path: str) -> None:
        """Write parquet to a .tmp file then rename — avoids corrupt files on kill."""
        tmp_path = final_path + ".tmp"
        df.write_parquet(tmp_path)
        os.replace(tmp_path, final_path)  # atomic on Linux (same filesystem)

    def _flush_chunks() -> None:
        nonlocal chunk_idx
        prefix = f"shard_{shard_id:03d}_chunk_{chunk_idx:05d}"
        if non_treated_chunk:
            _write_atomic(
                pl.DataFrame(non_treated_chunk),
                os.path.join(non_treated_dir, f"{prefix}.parquet"),
            )
        if treated_chunk:
            _write_atomic(
                pl.DataFrame(treated_chunk),
                os.path.join(treated_dir, f"{prefix}.parquet"),
            )
        if drug_info_chunk:
            # Explicit schema: era_end_epoch_time may be None in every row of a
            # chunk (right-censored patients).  Without an explicit schema Polars
            # infers Null type, which breaks concatenation across chunks later.
            di_schema = {
                "person_id":          pl.Int64,
                "drug_concept_id":    pl.String,
                "drug_epoch_time":    pl.Float64,
                "era_end_epoch_time": pl.Float64,
                "arm":                pl.String,
            }
            _write_atomic(
                pl.DataFrame(drug_info_chunk, schema=di_schema),
                os.path.join(drug_info_dir, f"{prefix}.parquet"),
            )
        if observed_outcomes_chunk:
            # Explicit schema prevents Polars from inferring Null type for
            # outcome columns when the first rows all have None values.
            oc_schema = {
                "person_id": pl.Int64,
                "drug_epoch_time": pl.Float64,
                **{k: pl.Float64 for k in outcome_concept_groups},
            }
            _write_atomic(
                pl.DataFrame(observed_outcomes_chunk, schema=oc_schema),
                os.path.join(observed_outcomes_dir, f"{prefix}.parquet"),
            )
        non_treated_chunk.clear()
        treated_chunk.clear()
        drug_info_chunk.clear()
        observed_outcomes_chunk.clear()
        chunk_idx += 1

    # Precompute era-end competing-concept sets (constant across all patients).
    # Source patients' era ends if they encounter a comparator or exclusion drug.
    # Comparator patients' era ends if they encounter a source or exclusion drug.
    _excl: Set[str] = exclusion_concepts or set()
    competing_for_source: Set[str]     = comparator_concepts | _excl
    competing_for_comparator: Set[str] = source_concepts | _excl

    for row in tqdm(
        df_shard.iter_rows(named=True),
        total=len(df_shard),
        desc=f"[shard {shard_id:03d}]",
        position=shard_id + 1,
        leave=True,
    ):
        concept_ids = _to_list(row["concept_ids"])

        # ---- Find first drug initiation (source or comparator) ----
        drug_pos, vs_pos, ve_pos, drug_concept_id = find_first_drug_initiation(
            concept_ids, drug_concepts
        )

        if drug_pos is None or vs_pos is None:
            n_skipped += 1
            continue

        # ---- Population filter (e.g. require HTN diagnosis in pre-drug history) ----
        if population_concepts:
            pre_drug_set = set(concept_ids[:drug_pos])
            if not (pre_drug_set & population_concepts):
                n_skipped += 1
                continue

        # ---- Exclusion drug check (prior exposure to other antihypertensives) ----
        # Patients with any exclusion drug in their pre-index history are dropped
        # to enforce the "new users" / treatment-naive eligibility criterion.
        if exclusion_concepts:
            pre_drug_set = set(concept_ids[:drug_pos])
            if pre_drug_set & exclusion_concepts:
                n_skipped += 1
                continue

        # ---- Same-visit co-prescription check ----
        # Patients whose INDEX VISIT contains both source and comparator drugs
        # (concurrent prescription) cannot be cleanly assigned to one arm — exclude.
        # Post-initiation crossovers are NOT excluded here; compute_drug_era_end_time
        # detects them and stores the switch time as era_end_epoch_time, which is
        # used downstream to censor follow-up at the moment of the switch.
        # Note: pre-index contamination by the opposite arm is structurally impossible
        # here — find_first_drug_initiation returns the FIRST drug event, so if the
        # opposite arm had appeared earlier it would have been the index event instead.
        visit_end = ve_pos if ve_pos is not None else len(concept_ids)
        index_visit_tokens = set(concept_ids[vs_pos:visit_end + 1])
        if index_visit_tokens & source_concepts and index_visit_tokens & comparator_concepts:
            n_skipped += 1
            continue

        # ---- Build truncated context rows ----
        non_treated_row = build_non_treated_context(row, vs_pos, available_array_cols)
        treated_row     = build_treated_context(row, ve_pos, available_array_cols)

        # Skip if non-treated context is too short
        if len(_to_list(non_treated_row.get("concept_ids", []))) < min_context_length:
            n_skipped += 1
            continue

        # ---- Copy scalar metadata into context rows ----
        for col in scalar_cols:
            val = row.get(col)
            non_treated_row[col] = val
            treated_row[col] = val

        # ---- Encode values and input_ids via tokenizer ----
        epoch_times = _to_list(row.get("epoch_times", []))
        drug_epoch_time = float(epoch_times[drug_pos]) if drug_pos < len(epoch_times) else None

        if tokenizer is not None:
            non_treated_row = _encode_values(non_treated_row, tokenizer)
            treated_row     = _encode_values(treated_row, tokenizer)

            non_treated_row["input_ids"] = tokenizer.encode(
                _to_list(non_treated_row["concept_ids"])
            )
            treated_row["input_ids"] = tokenizer.encode(
                _to_list(treated_row["concept_ids"])
            )

        # ---- Attach drug metadata to context rows ----
        person_id = row.get("person_id")
        for ctx_row in (non_treated_row, treated_row):
            ctx_row["person_id"]       = person_id
            ctx_row["drug_epoch_time"] = drug_epoch_time
            ctx_row["drug_concept_id"] = str(drug_concept_id)

        non_treated_chunk.append(non_treated_row)
        treated_chunk.append(treated_row)

        # ---- Drug info row ----
        arm_label = "source" if drug_concept_id in source_concepts else "comparator"

        # Compute drug era end time.
        # Target concepts = the patient's own drug class.
        # Competing concepts = opposite arm + any exclusion drugs (treatment switch).
        if arm_label == "source":
            target_concepts_era = source_concepts
            competing_for_era   = competing_for_source
        else:
            target_concepts_era = comparator_concepts
            competing_for_era   = competing_for_comparator

        era_end_epoch_time = compute_drug_era_end_time(
            concept_ids=concept_ids,
            epoch_times=epoch_times,
            target_concepts=target_concepts_era,
            competing_concepts=competing_for_era,
            start_drug_pos=drug_pos,
            era_gap_days=era_gap_days,
        )

        drug_info_chunk.append({
            "person_id":          person_id,
            "drug_concept_id":    str(drug_concept_id),
            "drug_epoch_time":    drug_epoch_time,
            "era_end_epoch_time": era_end_epoch_time,
            "arm":                arm_label,
        })

        # ---- Observed outcomes (post-initiation) ----
        if outcome_concept_groups:
            oc_times = find_outcomes_after_drug(
                concept_ids, epoch_times, drug_pos + 1, outcome_concept_groups
            )
            observed_outcomes_chunk.append(
                {"person_id": person_id, "drug_epoch_time": drug_epoch_time, **oc_times}
            )

        n_found += 1

        # Flush accumulated records to disk to avoid OOM
        if n_found % chunk_size == 0:
            _flush_chunks()

    # Final flush of any remaining records
    if non_treated_chunk or treated_chunk or drug_info_chunk or observed_outcomes_chunk:
        _flush_chunks()

    return n_found, n_skipped


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(
    patient_sequence_path: str,
    vocab_path: str,
    source_ingredient_ids: List[int],
    comparator_ingredient_ids: List[int],
    output_dir: Path,
    tokenizer_path: Optional[str] = None,
    outcome_concept_ids: Optional[List[int]] = None,
    population_concept_ids: Optional[List[int]] = None,
    exclusion_concept_ids: Optional[List[int]] = None,
    era_gap_days: int = 30,
    min_context_length: int = 4,
    overwrite: bool = False,
    num_workers: int = 1,
) -> None:
    """
    Full pipeline: load sequences → find drug initiation → write outputs.

    Parameters
    ----------
    patient_sequence_path
        Path to a parquet file *or* a folder of parquet files containing
        pre-tokenised patient sequences (patient_sequence.parquet format).
    vocab_path
        Root directory of the OMOP vocabulary download (must contain
        ``concept_ancestor/`` sub-directory with parquet files).
    source_ingredient_ids
        OMOP concept_ids for the source drug *ingredients* (e.g. duloxetine).
        All descendants are automatically expanded.
    comparator_ingredient_ids
        OMOP concept_ids for the comparator drug *ingredients* (e.g. amitriptyline).
        All descendants are automatically expanded.
    output_dir
        Directory where chunked output parquets will be written.
    tokenizer_path
        Optional path to a CehrGptTokenizer.  When provided, ``input_ids``,
        ``value_indicators``, and ``values`` are computed and written.
    outcome_concept_ids
        Optional list of OMOP concept_ids for outcomes to scan for after
        drug initiation.  Each is expanded to descendants.
    population_concept_ids
        Optional list of OMOP concept_ids defining the eligible population
        (e.g. HTN diagnosis concepts).  Each is expanded to descendants.
        Only patients who have at least one of these concepts in their
        pre-drug history are kept.  If None, no population filter is applied.
    exclusion_concept_ids
        Optional list of OMOP ingredient concept_ids for drugs that disqualify
        a patient from the "new users" cohort.  Each is expanded to all
        descendants via concept_ancestor.  Any patient whose pre-drug history
        contains at least one of these concepts is excluded.  Typical use:
        list all other first-line antihypertensive classes (ARBs, CCBs,
        second-line agents) so that only treatment-naive patients are kept.
        If None, no exclusion filter is applied.
    era_gap_days
        Persistence window in days for drug era computation (default: 30).
        Consecutive drug events separated by more than this many days belong
        to different eras.  A treatment switch (concept from the opposing arm
        or exclusion list) also terminates the era.  The computed
        ``era_end_epoch_time`` is written to ``drug_info/`` and used by
        ``hazard_ratio_estimation.py`` as an additional censoring time.
    min_context_length
        Minimum number of tokens required in the non-treated context
        (patients with fewer tokens are skipped).  Default 4.
    overwrite
        If False (default), skip extraction when output dirs already exist.
    num_workers
        Number of parallel worker processes.  Default 1 (single-process).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 0. Skip if outputs already exist                                    #
    # ------------------------------------------------------------------ #
    non_treated_dir      = output_dir / "non_treated_context"
    treated_dir          = output_dir / "treated_context"
    drug_info_dir        = output_dir / "drug_info"
    observed_outcomes_dir = output_dir / "observed_outcomes"

    def _dir_has_parquets(d: Path) -> bool:
        return d.is_dir() and any(d.glob("*.parquet"))

    if (
        not overwrite
        and _dir_has_parquets(non_treated_dir)
        and _dir_has_parquets(treated_dir)
        and _dir_has_parquets(drug_info_dir)
    ):
        print("Outputs already exist — skipping extraction (use --overwrite to force).")
        print(f"  {non_treated_dir}/")
        print(f"  {treated_dir}/")
        print(f"  {drug_info_dir}/")
        return

    non_treated_dir.mkdir(exist_ok=True)
    treated_dir.mkdir(exist_ok=True)
    drug_info_dir.mkdir(exist_ok=True)
    observed_outcomes_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Expand source and comparator drug concepts to descendants        #
    # ------------------------------------------------------------------ #
    print(f"Source ingredient concept_ids:     {source_ingredient_ids}")
    print(f"Comparator ingredient concept_ids: {comparator_ingredient_ids}")
    print("Expanding to descendants via concept_ancestor …")
    source_concepts     = load_drug_descendant_concepts(vocab_path, source_ingredient_ids)
    comparator_concepts = load_drug_descendant_concepts(vocab_path, comparator_ingredient_ids)
    drug_concepts       = source_concepts | comparator_concepts
    print(
        f"  → {len(source_concepts):,} source concepts, "
        f"{len(comparator_concepts):,} comparator concepts, "
        f"{len(drug_concepts):,} combined"
    )

    # ------------------------------------------------------------------ #
    # 2. Expand outcome concept IDs to descendants                        #
    # ------------------------------------------------------------------ #
    outcome_concept_groups: Dict[str, Set[str]] = {}
    if outcome_concept_ids:
        print(f"Expanding {len(outcome_concept_ids)} outcome concept(s) to descendants …")
        for oc_id in outcome_concept_ids:
            descendants = load_drug_descendant_concepts(vocab_path, [oc_id])
            outcome_concept_groups[str(oc_id)] = descendants
            print(f"  → outcome {oc_id}: {len(descendants):,} descendants")

    # ------------------------------------------------------------------ #
    # 2b. Expand population concept IDs to descendants (optional filter)  #
    # ------------------------------------------------------------------ #
    population_concepts: Optional[Set[str]] = None
    if population_concept_ids:
        print(f"Expanding {len(population_concept_ids)} population concept(s) to descendants …")
        population_concepts = load_drug_descendant_concepts(vocab_path, population_concept_ids)
        print(f"  → {len(population_concepts):,} population concepts (patients must have ≥1 in pre-drug history)")

    # ------------------------------------------------------------------ #
    # 2c. Expand exclusion concept IDs to descendants (new-user filter)   #
    # ------------------------------------------------------------------ #
    exclusion_concepts: Optional[Set[str]] = None
    if exclusion_concept_ids:
        print(f"Expanding {len(exclusion_concept_ids)} exclusion concept(s) to descendants …")
        exclusion_concepts = load_drug_descendant_concepts(vocab_path, exclusion_concept_ids)
        print(f"  → {len(exclusion_concepts):,} exclusion concepts (patients with ≥1 in pre-drug history are dropped)")

    # ------------------------------------------------------------------ #
    # 3. Enumerate parquet files and inspect schema without loading data  #
    # ------------------------------------------------------------------ #
    if os.path.isdir(patient_sequence_path):
        all_files = sorted(glob_module.glob(os.path.join(patient_sequence_path, "*.parquet")))
    else:
        all_files = [patient_sequence_path]

    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {patient_sequence_path}")

    print(f"Scanning patient sequences ({len(all_files)} parquet file(s)) …")
    lf = pl.scan_parquet(all_files)
    schema = lf.collect_schema()
    all_cols = list(schema.keys())
    n_rows = lf.select(pl.len()).collect().item()
    print(f"  → {n_rows:,} patient sequences found")

    available_array_cols = [c for c in SEQUENCE_ARRAY_COLS if c in all_cols]
    scalar_cols = [c for c in all_cols if c not in SEQUENCE_ARRAY_COLS]

    missing_required = REQUIRED_COLS - {"input_ids"} - set(all_cols)
    if missing_required:
        print(f"  WARNING: required columns missing from source data: {sorted(missing_required)}")

    # ------------------------------------------------------------------ #
    # 4. Dispatch shards to worker processes                              #
    #    Files are distributed round-robin so each worker reads only     #
    #    its assigned files — no I/O contention, no data serialisation   #
    #    through the Pool queue.                                         #
    # ------------------------------------------------------------------ #
    CHUNK_SIZE = 25_000
    effective_workers = min(num_workers, len(all_files))

    # Round-robin file assignment: worker i gets files [i, i+N, i+2N, ...]
    file_groups = [all_files[i::effective_workers] for i in range(effective_workers)]

    print(
        f"Processing {n_rows:,} patients with {effective_workers} worker(s) "
        f"({len(all_files)} files distributed round-robin) …"
    )

    # Only read the columns we actually need — skip unrelated columns
    needed_cols = list(dict.fromkeys(available_array_cols + scalar_cols))

    shard_args = [
        (
            shard_id,
            file_groups[shard_id],   # list of parquet files for this worker
            needed_cols,
            source_concepts,
            comparator_concepts,
            drug_concepts,
            available_array_cols,
            scalar_cols,
            outcome_concept_groups,
            str(non_treated_dir),
            str(treated_dir),
            str(drug_info_dir),
            str(observed_outcomes_dir),
            tokenizer_path,
            min_context_length,
            CHUNK_SIZE,
            population_concepts,
            exclusion_concepts,
            era_gap_days,
        )
        for shard_id in range(effective_workers)
    ]

    if effective_workers == 1:
        total_found, total_skipped = _process_shard(*shard_args[0])
    else:
        # Use 'spawn' (not the default 'fork') to avoid deadlocks caused by
        # forking a process that already has background threads from Polars /
        # NumPy / HuggingFace.  With spawn each worker starts clean.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=effective_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_shard_star, shard_args),
                    total=effective_workers,
                    desc="[overall] shards done",
                    unit="shard",
                    position=0,
                    leave=True,
                )
            )
        total_found   = sum(r[0] for r in results)
        total_skipped = sum(r[1] for r in results)

    print(f"  → {total_found:,} patients kept  |  {total_skipped:,} skipped")
    print(f"Done.  Outputs written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract partial patient sequences up to first drug exposure"
    )
    parser.add_argument(
        "--patient_sequence_path",
        required=True,
        help="Path to patient_sequence parquet file or folder of parquet files",
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="Root directory of the OMOP vocabulary download "
             "(must contain concept_ancestor/ sub-folder with parquet files)",
    )
    parser.add_argument(
        "--source_concept_ids",
        required=True,
        help="Comma-separated OMOP ingredient concept_ids for the source drug class (e.g. ACEi)",
    )
    parser.add_argument(
        "--comparator_concept_ids",
        required=True,
        help="Comma-separated OMOP ingredient concept_ids for the comparator drug class (e.g. thiazide)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where output parquets will be written",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Path to a CehrGptTokenizer directory. When provided, input_ids are "
             "generated from concept_ids and written into the output parquets.",
    )
    parser.add_argument(
        "--outcome_concept_ids",
        default=None,
        help="Comma-separated OMOP concept_ids for outcomes to extract from the "
             "post-initiation sequence (e.g. 4329847,316139,4110192,376713). "
             "Each is expanded to descendants. Writes observed_outcomes/ chunks.",
    )
    parser.add_argument(
        "--population_concept_ids",
        default=None,
        help="Comma-separated OMOP concept_ids defining the eligible population "
             "(e.g. hypertension diagnosis concepts). Each is expanded to "
             "descendants via concept_ancestor. Only patients whose pre-drug history "
             "contains at least one matching concept are kept. Default: no filter.",
    )
    parser.add_argument(
        "--exclusion_concept_ids",
        default=None,
        help="Comma-separated OMOP ingredient concept_ids for drugs that disqualify "
             "a patient from the 'new users' cohort. Each is expanded to all descendants "
             "via concept_ancestor. Patients whose pre-drug history contains any of these "
             "concepts are excluded. Use to enforce treatment-naive eligibility (e.g. list "
             "all other first-line antihypertensive classes for LEGEND-HTN). Default: no filter.",
    )
    parser.add_argument(
        "--era_gap_days",
        type=int,
        default=30,
        help="Persistence window in days for drug era computation (default: 30). "
             "Consecutive drug events separated by more than this many days belong to "
             "different eras; a treatment switch also terminates the era. The computed "
             "era_end_epoch_time is written to drug_info/ for use in hazard ratio "
             "estimation as an additional censoring time.",
    )
    parser.add_argument(
        "--min_context_length",
        type=int,
        default=4,
        help="Minimum token count required in the non-treated context (default: 4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files even if they already exist",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for shard processing (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_ingredient_ids     = [int(x.strip()) for x in args.source_concept_ids.split(",")]
    comparator_ingredient_ids = [int(x.strip()) for x in args.comparator_concept_ids.split(",")]
    outcome_concept_ids = (
        [int(x.strip()) for x in args.outcome_concept_ids.split(",")]
        if args.outcome_concept_ids else None
    )
    population_concept_ids = (
        [int(x.strip()) for x in args.population_concept_ids.split(",")]
        if args.population_concept_ids else None
    )
    exclusion_concept_ids = (
        [int(x.strip()) for x in args.exclusion_concept_ids.split(",")]
        if args.exclusion_concept_ids else None
    )
    process(
        patient_sequence_path=args.patient_sequence_path,
        vocab_path=args.vocab_path,
        source_ingredient_ids=source_ingredient_ids,
        comparator_ingredient_ids=comparator_ingredient_ids,
        output_dir=Path(args.output_dir),
        tokenizer_path=args.tokenizer_path,
        outcome_concept_ids=outcome_concept_ids,
        population_concept_ids=population_concept_ids,
        exclusion_concept_ids=exclusion_concept_ids,
        era_gap_days=args.era_gap_days,
        min_context_length=args.min_context_length,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
