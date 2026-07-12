#!/usr/bin/env python3
"""
LEGEND-HTN support: create an active-comparator context by swapping the drug
concept in a treated_context.parquet with a frequency-sampled comparator drug.

The comparator drug concept is NOT fixed — it is sampled proportional to the
empirical frequency of each comparator ingredient among patients who actually
received the comparator class.  This preserves the real-world distribution of
prescribing patterns in the counterfactual arm.

Example:
  If 65% of thiazide patients received HCTZ and 35% received chlorthalidone,
  then ACEi patients assigned to the thiazide counterfactual arm will receive
  HCTZ with p=0.65 and chlorthalidone with p=0.35 (independently per patient).

Pipeline integration
--------------------
                    extract_drug_initiation_sequences.py
                           |
              ┌────────────┴────────────┐
              ▼                         ▼
  treated_context.parquet        non_treated_context.parquet
         │
         │  create_drug_swap_context.py  (once per comparison direction)
         ▼
  comparator_context.parquet
         │
         └──── generate_counterfactual_sequences.py ──► HR estimation

Usage (ACEi vs Thiazide)
------------------------
# ACEi patients → frequency-sampled thiazide counterfactual
python create_drug_swap_context.py \\
    --treated_context_path  /data/contexts/treated_context.parquet \\
    --drug_info_path        /data/contexts/drug_info.parquet \\
    --vocab_path            /data/omop_vocab \\
    --source_concept_ids    1308216,1346654,1332418,135376,1395058,1310756,1363749,1328956,1373355,1307046 \\
    --comparator_concept_ids 1395058,974166,978555,907013 \\
    --output_path           /data/swap/acei_pts_thiazide_ctx.parquet \\
    --also_write_treated_path /data/swap/acei_pts_acei_ctx.parquet

# Thiazide patients → frequency-sampled ACEi counterfactual
python create_drug_swap_context.py \\
    --treated_context_path  /data/contexts/treated_context.parquet \\
    --drug_info_path        /data/contexts/drug_info.parquet \\
    --vocab_path            /data/omop_vocab \\
    --source_concept_ids    1395058,974166,978555,907013 \\
    --comparator_concept_ids 1308216,1346654,1332418,135376,1395058,1310756,1363749,1328956,1373355,1307046 \\
    --output_path           /data/swap/thiazide_pts_acei_ctx.parquet \\
    --also_write_treated_path /data/swap/thiazide_pts_thiazide_ctx.parquet
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import polars as pl

# ---------------------------------------------------------------------------
# LEGEND-HTN ACEi vs Thiazide ingredient concept_ids
# ---------------------------------------------------------------------------

# ACE inhibitor ingredient concept_ids (user-supplied)
# NOTE: 1395058 appears in both this list (Quinapril) and the thiazide list
#       (Chlorthalidone) — verify in your OMOP vocabulary before running.
ACEI_INGREDIENT_CONCEPT_IDS: List[int] = [
    1308216,   # Lisinopril
    1346654,   # Ramipril
    1332418,   # Enalapril
    135376,    # Benazepril   ← verify: typical OMOP IDs are 7 digits
    1395058,   # Quinapril    ← verify: same ID given for Chlorthalidone below
    1310756,   # Captopril
    1363749,   # Fosinopril
    1328956,   # Moexipril
    1373355,   # Perindopril
    1307046,   # Trandolapril
]

# Thiazide / thiazide-like diuretic ingredient concept_ids (user-supplied)
THIAZIDE_INGREDIENT_CONCEPT_IDS: List[int] = [
    1395058,   # Chlorthalidone  ← verify: same ID given for Quinapril above
    974166,    # Hydrochlorothiazide (HCTZ)
    978555,    # Indapamide
    907013,    # Metolazone
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_descendants_for_concepts(vocab_path: str, concept_ids: List[int]) -> Set[str]:
    """
    Return all descendant concept_ids (as strings) for a list of ingredient
    concept_ids, including the concepts themselves, via concept_ancestor.
    """
    ancestor_glob = os.path.join(vocab_path, "concept_ancestor", "*.parquet")
    descendants: List[str] = (
        pl.scan_parquet(ancestor_glob)
        .filter(pl.col("ancestor_concept_id").is_in(concept_ids))
        .select(pl.col("descendant_concept_id").cast(pl.String))
        .collect()
        ["descendant_concept_id"]
        .to_list()
    )
    result = set(descendants)
    result.update(str(c) for c in concept_ids)
    return result


def compute_ingredient_sampling_weights(
    drug_info: pl.DataFrame,
    comparator_patient_ids: Set[int],
    comparator_ingredient_ids: List[int],
    vocab_path: str,
) -> Dict[str, float]:
    """
    Compute the empirical frequency of each comparator ingredient among
    patients who actually received the comparator drug class.

    Each comparator patient's drug_concept_id is mapped back to the closest
    ingredient in *comparator_ingredient_ids* via concept_ancestor.  The
    resulting normalised weight dict {ingredient_str: probability} is used
    to sample a realistic comparator concept for each source patient.

    Falls back to a uniform distribution if frequencies cannot be estimated.
    """
    comp_drugs = (
        drug_info
        .filter(pl.col("person_id").is_in(list(comparator_patient_ids)))
        ["drug_concept_id"]
        .to_list()
    )

    if not comp_drugs:
        print("  Warning: no comparator patients found — using uniform ingredient weights")
        n = len(comparator_ingredient_ids)
        return {str(c): 1.0 / n for c in comparator_ingredient_ids}

    # Build per-ingredient descendants map
    print(f"  Building ingredient→descendants map for {len(comparator_ingredient_ids)} ingredients …")
    ingredient_descendants: Dict[str, Set[str]] = {
        str(ing): get_descendants_for_concepts(vocab_path, [ing])
        for ing in comparator_ingredient_ids
    }

    # Count patients per ingredient
    counts: Dict[str, int] = {str(c): 0 for c in comparator_ingredient_ids}
    unmatched = 0
    for drug_str in (str(d) for d in comp_drugs):
        matched = False
        for ing_str, descendants in ingredient_descendants.items():
            if drug_str in descendants:
                counts[ing_str] += 1
                matched = True
                break
        if not matched:
            unmatched += 1

    if unmatched:
        print(f"  Warning: {unmatched:,} comparator drug concepts did not map to any ingredient")

    total = sum(counts.values())
    if total == 0:
        print("  Warning: all comparator drugs unmatched — using uniform ingredient weights")
        n = len(comparator_ingredient_ids)
        return {str(c): 1.0 / n for c in comparator_ingredient_ids}

    weights = {ing: cnt / total for ing, cnt in counts.items() if cnt > 0}
    print("  Comparator ingredient sampling weights (empirical frequency):")
    for ing, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    concept {ing}: {w:.3%}")
    return weights


# ---------------------------------------------------------------------------
# Drug swap logic
# ---------------------------------------------------------------------------

def swap_drug_in_concept_ids(
    concept_ids: List[str],
    source_concepts: Set[str],
    comparator_concept_id: str,
) -> Optional[tuple]:
    """
    Replace the first occurrence of a source drug concept with
    *comparator_concept_id* in *concept_ids*.

    Returns (swap_pos, modified_list), or None if no source concept was found.
    """
    swapped = list(concept_ids)
    for i, token in enumerate(swapped):
        if token in source_concepts:
            swapped[i] = comparator_concept_id
            return i, swapped
    return None


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def create_comparator_context(
    treated_context_path: str,
    drug_info_path: str,
    vocab_path: str,
    source_concept_ids: List[int],
    comparator_concept_ids: List[int],
    output_path: Path,
    also_write_treated_path: Optional[Path] = None,
    max_source_patients: Optional[int] = None,
    seed: int = 42,
) -> None:
    """
    Filter *treated_context_path* to patients in the SOURCE drug class, then
    swap each patient's drug concept with a comparator concept sampled
    proportional to the empirical frequency of comparator ingredients.

    Parameters
    ----------
    treated_context_path
        treated_context.parquet from extract_drug_initiation_sequences.py.
    drug_info_path
        drug_info.parquet from the same extraction run.
    vocab_path
        OMOP vocabulary root (must contain concept_ancestor/ parquets).
    source_concept_ids
        OMOP ingredient concept_ids for the SOURCE drug class.  All clinical-
        drug descendants are expanded via concept_ancestor.
    comparator_concept_ids
        OMOP ingredient concept_ids for the COMPARATOR drug class.  Each
        source patient's drug is replaced by one of these, sampled
        proportional to the empirical frequency among patients who actually
        received the comparator class.
    output_path
        Destination parquet for the swapped (comparator) contexts.
    also_write_treated_path
        If provided, also write the unmodified (actual) treated contexts for
        the filtered patients — ready to use as the source arm in generation.
    max_source_patients
        If set, randomly sample this many source-class patients before
        generating contexts (useful for pilot runs on large cohorts).
    seed
        Random seed for reproducible frequency-based sampling and patient
        sub-sampling.
    """
    random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drug_info = pl.read_parquet(drug_info_path)

    # 1. Expand source ingredients to all descendants
    print(f"[source]     Expanding {len(source_concept_ids)} ingredient(s) via concept_ancestor …")
    source_concepts = get_descendants_for_concepts(vocab_path, source_concept_ids)
    print(f"             → {len(source_concepts):,} source drug concepts")

    source_patients: Set[int] = set(
        drug_info
        .filter(pl.col("drug_concept_id").is_in(list(source_concepts)))
        ["person_id"]
        .to_list()
    )
    print(f"             → {len(source_patients):,} source-class patients")

    # 2. Expand comparator ingredients and identify comparator patients
    print(f"[comparator] Expanding {len(comparator_concept_ids)} ingredient(s) via concept_ancestor …")
    comparator_concepts = get_descendants_for_concepts(vocab_path, comparator_concept_ids)
    comparator_patients: Set[int] = set(
        drug_info
        .filter(pl.col("drug_concept_id").is_in(list(comparator_concepts)))
        ["person_id"]
        .to_list()
    )
    print(f"             → {len(comparator_patients):,} comparator-class patients (for frequency estimation)")

    # 3. Compute empirical ingredient frequencies from comparator patients
    weights = compute_ingredient_sampling_weights(
        drug_info=drug_info,
        comparator_patient_ids=comparator_patients,
        comparator_ingredient_ids=comparator_concept_ids,
        vocab_path=vocab_path,
    )
    weight_concepts = list(weights.keys())
    weight_values   = [weights[c] for c in weight_concepts]

    # 4. Load and filter treated contexts for source patients
    df = pl.read_parquet(treated_context_path)
    df_source = df.filter(pl.col("person_id").is_in(list(source_patients)))
    print(f"\n{len(df_source):,} source-class treated contexts to swap")

    if max_source_patients is not None and len(df_source) > max_source_patients:
        df_source = df_source.sample(n=max_source_patients, seed=seed, shuffle=True)
        print(f"Randomly sampled {max_source_patients:,} source patients (seed={seed})")

    if also_write_treated_path is not None:
        Path(also_write_treated_path).parent.mkdir(parents=True, exist_ok=True)
        df_source.write_parquet(str(also_write_treated_path))
        print(f"Actual treated contexts → {also_write_treated_path}")

    # 5. Build concept_str → token_id mapping from the dataset so we can
    #    update input_ids at the swap position (concept_ids and input_ids
    #    are parallel arrays; the collator uses input_ids directly).
    vocab_map: Dict[str, int] = {}
    if "input_ids" in df_source.columns:
        for concepts, token_ids in zip(
            df_source["concept_ids"].to_list(),
            df_source["input_ids"].to_list(),
        ):
            for c, t in zip(concepts, token_ids):
                vocab_map.setdefault(str(c), int(t))

    # 6. Swap each patient's drug concept with a frequency-sampled comparator
    swapped_records = []
    n_swapped = 0
    n_failed = 0

    for row in df_source.iter_rows(named=True):
        # Sample comparator ingredient proportional to empirical frequency
        comparator_str = random.choices(weight_concepts, weights=weight_values, k=1)[0]
        result = swap_drug_in_concept_ids(
            list(row["concept_ids"]), source_concepts, comparator_str
        )
        if result is None:
            n_failed += 1
            continue
        swap_pos, new_concept_ids = result
        row = dict(row)
        row["concept_ids"] = new_concept_ids
        row["drug_concept_id"] = comparator_str
        # Keep input_ids in sync: update the token at the swap position so
        # the model is conditioned on the comparator drug, not the source drug.
        if "input_ids" in row and comparator_str in vocab_map:
            input_ids = list(row["input_ids"])
            input_ids[swap_pos] = vocab_map[comparator_str]
            row["input_ids"] = input_ids
        swapped_records.append(row)
        n_swapped += 1

    print(f"Swapped {n_swapped:,} | failed {n_failed:,} (source drug not found in context)")

    pl.DataFrame(swapped_records).write_parquet(str(output_path))
    print(f"Comparator context → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an active-comparator context by swapping each source "
            "patient's drug concept with a frequency-sampled comparator drug "
            "(LEGEND-HTN active-comparator design)."
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
        "--source_concept_ids",
        required=True,
        help=(
            "Comma-separated OMOP ingredient concept_ids for the SOURCE drug class. "
            "ACEi: 1308216,1346654,1332418,135376,1395058,1310756,1363749,1328956,1373355,1307046  "
            "Thiazide: 1395058,974166,978555,907013"
        ),
    )
    parser.add_argument(
        "--comparator_concept_ids",
        required=True,
        help=(
            "Comma-separated OMOP ingredient concept_ids for the COMPARATOR drug class. "
            "Each source patient's drug is replaced by one of these, sampled proportional "
            "to their empirical frequency among patients who actually received the comparator class."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for the comparator context parquet",
    )
    parser.add_argument(
        "--also_write_treated_path",
        default=None,
        help=(
            "Optional: also write the unmodified (actual) treated contexts "
            "for the filtered source patients to this path"
        ),
    )
    parser.add_argument(
        "--max_source_patients",
        type=int,
        default=None,
        help=(
            "Optional: randomly sample this many source-class patients before "
            "generating contexts (e.g. 10000 for a pilot run). "
            "Default: use all patients."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible frequency-based sampling (default: 42)",
    )
    parser.add_argument(
        "--list_concepts",
        action="store_true",
        help="Print the built-in ACEi and thiazide concept_id lists then exit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_concepts:
        print("ACEi ingredient concept_ids:")
        for cid in ACEI_INGREDIENT_CONCEPT_IDS:
            print(f"  {cid}")
        print("\nThiazide ingredient concept_ids:")
        for cid in THIAZIDE_INGREDIENT_CONCEPT_IDS:
            print(f"  {cid}")
        return

    source_concept_ids      = [int(x.strip()) for x in args.source_concept_ids.split(",")]
    comparator_concept_ids  = [int(x.strip()) for x in args.comparator_concept_ids.split(",")]

    create_comparator_context(
        treated_context_path=args.treated_context_path,
        drug_info_path=args.drug_info_path,
        vocab_path=args.vocab_path,
        source_concept_ids=source_concept_ids,
        comparator_concept_ids=comparator_concept_ids,
        output_path=Path(args.output_path),
        also_write_treated_path=(
            Path(args.also_write_treated_path) if args.also_write_treated_path else None
        ),
        max_source_patients=args.max_source_patients,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
