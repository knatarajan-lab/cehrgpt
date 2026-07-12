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

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VISIT_START_TOKEN = "[VS]"
VISIT_END_TOKEN = "[VE]"

# All columns that are parallel arrays (length == len(concept_ids))
SEQUENCE_ARRAY_COLS = [
    "concept_ids",
    "input_ids",
    "visit_segments",
    "orders",
    "dates",
    "ages",
    "visit_concept_orders",
    "concept_value_masks",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    non_treated_path = output_dir / "non_treated_context.parquet"
    treated_path = output_dir / "treated_context.parquet"
    drug_info_path = output_dir / "drug_info.parquet"

    if not overwrite and non_treated_path.exists() and treated_path.exists() and drug_info_path.exists():
        print("Outputs already exist — skipping extraction (use --overwrite to force).")
        print(f"  {non_treated_path}")
        print(f"  {treated_path}")
        print(f"  {drug_info_path}")
        return

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
    # 2. Load patient sequences                                           #
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

    # ------------------------------------------------------------------ #
    # 3. Process each patient                                             #
    # ------------------------------------------------------------------ #
    non_treated_records: List[Dict[str, Any]] = []
    treated_records: List[Dict[str, Any]] = []
    drug_info_records: List[Dict[str, Any]] = []
    n_skipped = 0

    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Extracting drug initiations", unit="pt"):
        concept_ids = _to_list(row["concept_ids"])
        epoch_times = _to_list(row["epoch_times"])

        drug_pos, vs_pos, ve_pos, drug_concept_id = find_first_drug_initiation(
            concept_ids, drug_concepts
        )

        if drug_pos is None:
            # Patient never received any of the target drugs
            n_skipped += 1
            continue

        drug_epoch_time = float(epoch_times[drug_pos])

        # ---- non-treated context ----------------------------------------
        nt = build_non_treated_context(row, vs_pos if vs_pos is not None else drug_pos, available_array_cols)

        if len(nt.get("concept_ids", [])) < min_context_length:
            # Not enough history before the drug visit
            n_skipped += 1
            continue

        # Propagate scalar columns and add metadata
        for col in scalar_cols:
            nt[col] = row[col]
        nt["drug_concept_id"] = drug_concept_id
        nt["drug_epoch_time"] = drug_epoch_time
        nt["num_of_concepts"] = len(nt["concept_ids"])
        if tokenizer is not None and "input_ids" not in nt:
            nt["input_ids"] = tokenizer.encode(nt["concept_ids"])
        non_treated_records.append(nt)

        # ---- treated context --------------------------------------------
        t = build_treated_context(row, ve_pos, available_array_cols)
        for col in scalar_cols:
            t[col] = row[col]
        t["drug_concept_id"] = drug_concept_id
        t["drug_epoch_time"] = drug_epoch_time
        t["num_of_concepts"] = len(t["concept_ids"])
        if tokenizer is not None and "input_ids" not in t:
            t["input_ids"] = tokenizer.encode(t["concept_ids"])
        treated_records.append(t)

        # ---- drug info --------------------------------------------------
        drug_info_records.append(
            {
                "person_id": row["person_id"],
                "drug_concept_id": drug_concept_id,
                "drug_epoch_time": drug_epoch_time,
            }
        )

    n_found = len(non_treated_records)
    print(f"  → {n_found:,} patients with drug exposure kept  |  {n_skipped:,} skipped")

    # Free the input DataFrame before materialising outputs — the source data
    # can be several GB and would otherwise overlap in memory with the output
    # DataFrames, causing OOM during write_parquet.
    del df

    # ------------------------------------------------------------------ #
    # 4. Write outputs (one at a time to limit peak memory)               #
    # ------------------------------------------------------------------ #
    print("Writing non_treated_context.parquet …")
    pl.DataFrame(non_treated_records).write_parquet(non_treated_path)
    del non_treated_records

    print("Writing treated_context.parquet …")
    pl.DataFrame(treated_records).write_parquet(treated_path)
    del treated_records

    print("Writing drug_info.parquet …")
    pl.DataFrame(drug_info_records).write_parquet(drug_info_path)

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
    process(
        patient_sequence_path=args.patient_sequence_path,
        vocab_path=args.vocab_path,
        drug_ingredient_ids=drug_ingredient_ids,
        output_dir=Path(args.output_dir),
        tokenizer_path=args.tokenizer_path,
        min_context_length=args.min_context_length,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
