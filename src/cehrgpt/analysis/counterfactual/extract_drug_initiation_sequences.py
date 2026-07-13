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

  4. Writes three parquet files to <output_dir>:
       non_treated_context.parquet
       treated_context.parquet
       drug_info.parquet           (person_id, drug_concept_id, drug_epoch_time)

Usage
-----
python extract_drug_initiation_sequences.py \\
    --patient_sequence_path /path/to/patient_sequence.parquet \\
    --vocab_path            /path/to/omop_vocab_dir \\
    --drug_concept_ids      1308216,1367500 \\
    --output_dir            /path/to/output
"""

import argparse
import os
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

    Parameters
    ----------
    concept_ids
        Full token sequence for the patient.
    epoch_times
        Parallel epoch timestamps (seconds since epoch) for each token.
    start_pos
        Index to start scanning from (typically ve_pos + 1, i.e. after the
        drug-initiation visit closing token).
    outcome_concept_groups
        Mapping of {outcome_label: set_of_descendant_concept_ids}.

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
# Main processing
# ---------------------------------------------------------------------------

def process(
    patient_sequence_path: str,
    vocab_path: str,
    drug_ingredient_ids: List[int],
    output_dir: Path,
    tokenizer_path: Optional[str] = None,
    outcome_concept_ids: Optional[List[int]] = None,
    min_context_length: int = 4,
    overwrite: bool = False,
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
    drug_ingredient_ids
        OMOP concept_ids for the drug *ingredients* of interest.  All
        descendants are automatically expanded.
    output_dir
        Directory where the three output parquets will be written.
    min_context_length
        Minimum number of tokens required in the non-treated context
        (patients with fewer tokens are skipped).  The default of 4
        corresponds to the four demographic header tokens.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 0. Skip if outputs already exist                                    #
    # ------------------------------------------------------------------ #
    non_treated_dir = output_dir / "non_treated_context"
    treated_dir = output_dir / "treated_context"
    drug_info_dir = output_dir / "drug_info"
    observed_outcomes_dir = output_dir / "observed_outcomes"

    def _dir_has_parquets(d: Path) -> bool:
        return d.is_dir() and any(d.glob("*.parquet"))

    if not overwrite and _dir_has_parquets(non_treated_dir) and _dir_has_parquets(treated_dir) and _dir_has_parquets(drug_info_dir):
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
    # 1. Load tokenizer (optional)                                        #
    # ------------------------------------------------------------------ #
    tokenizer = None
    if tokenizer_path:
        print(f"Loading tokenizer from {tokenizer_path} …")
        tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_path)

    # ------------------------------------------------------------------ #
    # 2. Expand to all descendant drug concepts                           #
    # ------------------------------------------------------------------ #
    print(f"Drug ingredient concept_ids: {drug_ingredient_ids}")
    print("Expanding to descendants via concept_ancestor …")
    drug_concepts = load_drug_descendant_concepts(vocab_path, drug_ingredient_ids)
    print(f"  → {len(drug_concepts):,} drug concepts (ingredients + descendants)")

    # ------------------------------------------------------------------ #
    # 2b. Expand outcome concept IDs to descendants                       #
    # ------------------------------------------------------------------ #
    # outcome_concept_groups: {str(concept_id): set_of_descendants}
    outcome_concept_groups: Dict[str, Set[str]] = {}
    if outcome_concept_ids:
        print(f"Expanding {len(outcome_concept_ids)} outcome concept(s) to descendants …")
        for oc_id in outcome_concept_ids:
            descendants = load_drug_descendant_concepts(vocab_path, [oc_id])
            outcome_concept_groups[str(oc_id)] = descendants
            print(f"  → outcome {oc_id}: {len(descendants):,} descendants")

    # ------------------------------------------------------------------ #
    # 3. Load patient sequences                                           #
    # ------------------------------------------------------------------ #
    print("Loading patient sequences …")
    if os.path.isdir(patient_sequence_path):
        df = pl.read_parquet(os.path.join(patient_sequence_path, "*.parquet"))
    else:
        df = pl.read_parquet(patient_sequence_path)
    print(f"  → {len(df):,} patient sequences loaded")

    # Separate array columns (present in this dataset) from scalar ones
    all_cols = df.columns
    available_array_cols = [c for c in SEQUENCE_ARRAY_COLS if c in all_cols]
    scalar_cols = [c for c in all_cols if c not in SEQUENCE_ARRAY_COLS]

    missing_required = REQUIRED_COLS - {"input_ids"} - set(all_cols)  # input_ids generated later
    if missing_required:
        print(f"  WARNING: required columns missing from source data: {sorted(missing_required)}")

    # ------------------------------------------------------------------ #
    # 3. Process each patient — flush to disk every CHUNK_SIZE records   #
    #    to avoid accumulating all records in memory at once.            #
    # ------------------------------------------------------------------ #
    CHUNK_SIZE = 25_000

    non_treated_chunk: List[Dict[str, Any]] = []
    treated_chunk: List[Dict[str, Any]] = []
    drug_info_chunk: List[Dict[str, Any]] = []
    observed_outcomes_chunk: List[Dict[str, Any]] = []
    chunk_idx = 0
    n_found = 0
    n_skipped = 0

    def _flush_chunk() -> None:
        nonlocal chunk_idx
        pl.DataFrame(non_treated_chunk).write_parquet(non_treated_dir / f"chunk_{chunk_idx:05d}.parquet")
        pl.DataFrame(treated_chunk).write_parquet(treated_dir / f"chunk_{chunk_idx:05d}.parquet")
        pl.DataFrame(drug_info_chunk).write_parquet(drug_info_dir / f"chunk_{chunk_idx:05d}.parquet")
        if observed_outcomes_chunk:
            pl.DataFrame(observed_outcomes_chunk).write_parquet(observed_outcomes_dir / f"chunk_{chunk_idx:05d}.parquet")
        non_treated_chunk.clear()
        treated_chunk.clear()
        drug_info_chunk.clear()
        observed_outcomes_chunk.clear()
        chunk_idx += 1

    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Extracting drug initiations", unit="pt"):
        concept_ids = _to_list(row["concept_ids"])
        epoch_times = _to_list(row["epoch_times"])

        drug_pos, vs_pos, ve_pos, drug_concept_id = find_first_drug_initiation(
            concept_ids, drug_concepts
        )

        if drug_pos is None:
            n_skipped += 1
            continue

        drug_epoch_time = float(epoch_times[drug_pos])

        # ---- non-treated context ----------------------------------------
        nt = build_non_treated_context(row, vs_pos if vs_pos is not None else drug_pos, available_array_cols)

        if len(nt.get("concept_ids", [])) < min_context_length:
            n_skipped += 1
            continue

        for col in scalar_cols:
            nt[col] = row[col]
        nt["drug_concept_id"] = drug_concept_id
        nt["drug_epoch_time"] = drug_epoch_time
        nt["num_of_concepts"] = len(nt["concept_ids"])
        if tokenizer is not None:
            if "input_ids" not in nt:
                nt["input_ids"] = tokenizer.encode(nt["concept_ids"])
            nt = _encode_values(nt, tokenizer)
        non_treated_chunk.append(nt)

        # ---- treated context --------------------------------------------
        t = build_treated_context(row, ve_pos, available_array_cols)
        for col in scalar_cols:
            t[col] = row[col]
        t["drug_concept_id"] = drug_concept_id
        t["drug_epoch_time"] = drug_epoch_time
        t["num_of_concepts"] = len(t["concept_ids"])
        if tokenizer is not None:
            if "input_ids" not in t:
                t["input_ids"] = tokenizer.encode(t["concept_ids"])
            t = _encode_values(t, tokenizer)
        treated_chunk.append(t)

        # ---- drug info --------------------------------------------------
        drug_info_chunk.append(
            {
                "person_id": row["person_id"],
                "drug_concept_id": drug_concept_id,
                "drug_epoch_time": drug_epoch_time,
            }
        )

        # ---- observed outcomes (scan post-initiation sequence) ----------
        if outcome_concept_groups:
            scan_start = (ve_pos + 1) if ve_pos is not None else (drug_pos + 1)
            outcome_times = find_outcomes_after_drug(
                concept_ids, epoch_times, scan_start, outcome_concept_groups
            )
            for outcome_id, outcome_epoch_time in outcome_times.items():
                observed_outcomes_chunk.append(
                    {
                        "person_id": row["person_id"],
                        "drug_concept_id": drug_concept_id,
                        "drug_epoch_time": drug_epoch_time,
                        "outcome_concept_id": outcome_id,
                        "outcome_epoch_time": outcome_epoch_time,
                    }
                )

        n_found += 1

        if n_found % CHUNK_SIZE == 0:
            _flush_chunk()

    # Flush remaining records
    if non_treated_chunk:
        _flush_chunk()

    del df
    print(f"  → {n_found:,} patients with drug exposure kept  |  {n_skipped:,} skipped")
    print(f"Done.  {chunk_idx} chunk(s) written to {output_dir}")


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
        "--drug_concept_ids",
        required=True,
        help="Comma-separated OMOP concept_ids for drug ingredients, e.g. 1308216,1367500",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    drug_ingredient_ids = [int(x.strip()) for x in args.drug_concept_ids.split(",")]
    outcome_concept_ids = (
        [int(x.strip()) for x in args.outcome_concept_ids.split(",")]
        if args.outcome_concept_ids else None
    )
    process(
        patient_sequence_path=args.patient_sequence_path,
        vocab_path=args.vocab_path,
        drug_ingredient_ids=drug_ingredient_ids,
        output_dir=Path(args.output_dir),
        tokenizer_path=args.tokenizer_path,
        outcome_concept_ids=outcome_concept_ids,
        min_context_length=args.min_context_length,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
