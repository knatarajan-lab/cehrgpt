#!/usr/bin/env python3
"""
LEGEND-HTN support: create an active-comparator context by swapping the drug
concept in a treated_context.parquet with a specific comparator drug concept.

Background
----------
Standard observational studies (e.g. LEGEND-HTN) compare two real patient
groups: Drug-A users vs Drug-B users, adjusting for confounders via propensity
scores.

The CEHR-GPT counterfactual approach replaces propensity score matching with
within-patient counterfactuals: for every patient we generate two synthetic
futures under both Drug-A and Drug-B.  This requires a "drug-swap" context
where the concept_id of the drug that was actually administered in the
initiation visit is replaced with a representative concept_id from the
comparator drug class.

The model has learned drug → outcome associations during pre-training, so
presenting it with a comparator concept in the context elicits a trajectory
that is conditionally plausible given that drug was given.

Pipeline integration
--------------------
                    extract_drug_initiation_sequences.py
                           |
              ┌────────────┴────────────┐
              ▼                         ▼
  treated_context.parquet        non_treated_context.parquet
         │
         │  create_drug_swap_context.py
         │  (for each pairwise comparison, e.g. ACEi ↔ Thiazide)
         ▼
  comparator_context.parquet
         │
         └──── generate_counterfactual_sequences.py ──► HR estimation

Usage
-----
# Build comparator context for ACEi patients: swap their ACEi drug to HCTZ
python create_drug_swap_context.py \\
    --treated_context_path  /data/counterfactual/treated_context.parquet \\
    --drug_info_path        /data/counterfactual/drug_info.parquet \\
    --vocab_path            /data/omop_vocab \\
    --source_class_id       21600381 \\   # ACE inhibitors ancestor
    --comparator_concept_id 974166 \\     # hydrochlorothiazide (HCTZ)
    --output_path           /data/counterfactual/acei_vs_thiazide/comparator_context.parquet

# Also copy the treated contexts for the same patients as the actual arm:
# (just filter treated_context.parquet to patients in source_class)
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Set

import polars as pl

# ---------------------------------------------------------------------------
# LEGEND-HTN reference concept_ids
# ---------------------------------------------------------------------------
# Drug class ATC/OMOP ancestor concept_ids (from htn_treatment_pathway.py)
DRUG_CLASS_ANCESTORS = {
    "acei":      21600381,   # ACE inhibitors
    "bb":        21601461,   # Beta-blockers
    "ccb":       21601560,   # Dihydropyridine calcium channel blockers
    "thiazide":  21601664,   # Thiazide and thiazide-like diuretics
    "arb":       21601744,   # Angiotensin II receptor blockers
    "other":     21601782,   # Other antihypertensives
}

# Representative "index" drugs per class (most commonly prescribed, used as
# swap target when the user does not specify --comparator_concept_id)
DEFAULT_COMPARATOR_CONCEPTS = {
    "acei":     1308216,   # lisinopril
    "bb":       1314002,   # atenolol
    "ccb":      1332418,   # amlodipine
    "thiazide":  974166,   # hydrochlorothiazide (HCTZ)
    "arb":      1367500,   # losartan
}

# LEGEND-HTN primary pairwise comparisons (comparator class, reference class)
LEGEND_HTN_COMPARISONS = [
    ("acei",    "thiazide"),   # ACEi vs Thiazide  (primary)
    ("arb",     "thiazide"),   # ARB  vs Thiazide
    ("ccb",     "thiazide"),   # CCB  vs Thiazide
    ("bb",      "thiazide"),   # BB   vs Thiazide
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_descendants(vocab_path: str, ancestor_concept_id: int) -> Set[str]:
    """Return all descendant concept_ids (as strings) for one ancestor."""
    ancestor_glob = os.path.join(vocab_path, "concept_ancestor", "*.parquet")
    descendants: List[str] = (
        pl.scan_parquet(ancestor_glob)
        .filter(pl.col("ancestor_concept_id") == ancestor_concept_id)
        .select(pl.col("descendant_concept_id").cast(pl.String))
        .collect()
        ["descendant_concept_id"]
        .to_list()
    )
    result = set(descendants)
    result.add(str(ancestor_concept_id))
    return result


# ---------------------------------------------------------------------------
# Drug swap logic
# ---------------------------------------------------------------------------

def swap_drug_in_concept_ids(
    concept_ids: List[str],
    source_concepts: Set[str],
    comparator_concept_id: str,
) -> Optional[List[str]]:
    """
    Replace the first occurrence of a source drug concept with
    *comparator_concept_id* in *concept_ids*.

    Returns the modified list, or None if no source drug was found.
    """
    swapped = list(concept_ids)
    for i, token in enumerate(swapped):
        if token in source_concepts:
            swapped[i] = comparator_concept_id
            return swapped
    return None


def create_comparator_context(
    treated_context_path: str,
    drug_info_path: str,
    vocab_path: str,
    source_class_id: int,
    comparator_concept_id: int,
    output_path: Path,
    also_write_treated_path: Optional[Path] = None,
) -> None:
    """
    Filter *treated_context_path* to patients whose initiated drug belongs to
    *source_class_id*, then swap the drug concept to *comparator_concept_id*.

    Parameters
    ----------
    treated_context_path
        Output of extract_drug_initiation_sequences.py (treated arm).
    drug_info_path
        drug_info.parquet from the same extraction run.
    vocab_path
        OMOP vocabulary root (for concept_ancestor lookup).
    source_class_id
        OMOP ancestor concept_id identifying the SOURCE drug class
        (e.g. 21600381 for ACEi).  Only patients who initiated a drug in this
        class are included.
    comparator_concept_id
        Specific OMOP concept_id to swap in as the comparator drug
        (e.g. 974166 for HCTZ).
    output_path
        Where to write the comparator_context parquet.
    also_write_treated_path
        If provided, the ACTUAL (source-drug) treated context for the same
        patients is also written here — ready to use as the "treated arm" in
        generate_counterfactual_sequences.py.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Get all descendants of the source class
    print(f"Loading descendants of source class {source_class_id} …")
    source_concepts = get_descendants(vocab_path, source_class_id)
    print(f"  → {len(source_concepts):,} source drug concepts")

    # 2. Load drug_info to identify which patients are in the source class
    drug_info = pl.read_parquet(drug_info_path)
    source_patients = set(
        drug_info
        .filter(pl.col("drug_concept_id").is_in(list(source_concepts)))
        ["person_id"]
        .to_list()
    )
    print(f"  → {len(source_patients):,} patients initiated a source-class drug")

    # 3. Load and filter treated contexts
    df = pl.read_parquet(treated_context_path)
    df_source = df.filter(pl.col("person_id").is_in(list(source_patients)))
    print(f"  → {len(df_source):,} treated contexts for source-class patients")

    # 4. Optionally write the actual treated contexts for these patients
    if also_write_treated_path is not None:
        Path(also_write_treated_path).parent.mkdir(parents=True, exist_ok=True)
        df_source.write_parquet(str(also_write_treated_path))
        print(f"  Actual treated contexts → {also_write_treated_path}")

    # 5. Swap the drug concept in concept_ids
    comparator_str = str(comparator_concept_id)
    records = df_source.to_pandas().to_dict(orient="records")

    swapped_records = []
    n_swapped = 0
    n_failed = 0

    for row in records:
        concept_ids = list(row["concept_ids"])
        new_ids = swap_drug_in_concept_ids(concept_ids, source_concepts, comparator_str)
        if new_ids is None:
            n_failed += 1
            continue
        row["concept_ids"] = new_ids
        # Update drug_concept_id to reflect the swap
        row["drug_concept_id"] = comparator_str
        # input_ids are stale after swap — mark as None so the collator re-tokenises
        if "input_ids" in row:
            row["input_ids"] = None
        swapped_records.append(row)
        n_swapped += 1

    print(f"  Swapped {n_swapped:,} contexts  |  {n_failed:,} failed (drug not found in context)")

    # 6. Write output
    pl.DataFrame(swapped_records).write_parquet(str(output_path))
    print(f"  Comparator context → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an active-comparator context by swapping the drug concept "
            "in treated_context.parquet (LEGEND-HTN active comparator design)."
        )
    )
    parser.add_argument(
        "--treated_context_path",
        required=True,
        help="treated_context.parquet from extract_drug_initiation_sequences.py",
    )
    parser.add_argument(
        "--drug_info_path",
        required=True,
        help="drug_info.parquet from extract_drug_initiation_sequences.py",
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="OMOP vocabulary root directory (must contain concept_ancestor/ parquets)",
    )
    parser.add_argument(
        "--source_class_id",
        type=int,
        required=True,
        help=(
            "OMOP ancestor concept_id for the SOURCE drug class. "
            "Only patients whose initiated drug is a descendant of this concept "
            "are included. E.g. 21600381 for ACE inhibitors, 21601664 for thiazides."
        ),
    )
    parser.add_argument(
        "--comparator_concept_id",
        type=int,
        required=True,
        help=(
            "Specific OMOP concept_id to swap in as the comparator drug. "
            "E.g. 974166 (HCTZ) to make ACEi patients look like thiazide recipients."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for the comparator context parquet file.",
    )
    parser.add_argument(
        "--also_write_treated_path",
        default=None,
        help=(
            "Optional: also write the actual (source-drug) treated contexts "
            "for the filtered patients to this path."
        ),
    )
    # Convenience: list available class names / defaults
    parser.add_argument(
        "--list_classes",
        action="store_true",
        help="Print available drug class names, their ancestor IDs, and default comparator concepts, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_classes:
        print("\nAvailable drug class shortnames:")
        print(f"  {'Name':<12} {'Ancestor ID':<14} {'Default comparator concept (drug name)'}")
        print(f"  {'-'*60}")
        names = {
            "acei":    "lisinopril",
            "bb":      "atenolol",
            "ccb":     "amlodipine",
            "thiazide":"hydrochlorothiazide (HCTZ)",
            "arb":     "losartan",
        }
        for name, anc in DRUG_CLASS_ANCESTORS.items():
            comp = DEFAULT_COMPARATOR_CONCEPTS.get(name, "N/A")
            comp_name = names.get(name, "")
            print(f"  {name:<12} {anc:<14} {comp}  ({comp_name})")
        print("\nLEGEND-HTN pairwise comparisons (comparator vs reference):")
        for comp, ref in LEGEND_HTN_COMPARISONS:
            print(f"  {comp} vs {ref}")
        return

    create_comparator_context(
        treated_context_path=args.treated_context_path,
        drug_info_path=args.drug_info_path,
        vocab_path=args.vocab_path,
        source_class_id=args.source_class_id,
        comparator_concept_id=args.comparator_concept_id,
        output_path=Path(args.output_path),
        also_write_treated_path=(
            Path(args.also_write_treated_path)
            if args.also_write_treated_path
            else None
        ),
    )


if __name__ == "__main__":
    main()
