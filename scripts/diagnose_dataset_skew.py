"""
Diagnose why val split processes ~18x slower than train in extract_cohort_sequences.

Two hypotheses:
  A) Val patients have longer sequences (more tokens per record)
  B) Val patients have more prediction times in person_index_date_map

Usage:
    python scripts/diagnose_dataset_skew.py \
        --tokenized_path /path/to/tokenized_full_dataset \
        --cohort_path /path/to/cohort_folder
"""
import argparse
import os

import numpy as np
import polars as pl
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_path", required=True)
    parser.add_argument("--cohort_path", required=True)
    args = parser.parse_args()

    # ── Load cohort ──────────────────────────────────────────────────────────
    lf = pl.scan_parquet(os.path.join(args.cohort_path, "**", "*.parquet"))
    is_meds = "subject_id" in set(lf.schema.names())
    id_col = "subject_id" if is_meds else "person_id"
    time_col = "prediction_time" if is_meds else "index_date"
    cohort = lf.select([id_col, time_col]).collect()
    if is_meds:
        cohort = cohort.rename({"subject_id": "person_id", "prediction_time": "index_date"})

    pred_counts = (
        cohort.group_by("person_id")
        .agg(pl.len().alias("n_predictions"))
    )
    print("=== Cohort prediction times per patient ===")
    print(pred_counts.describe())

    # ── Load tokenized dataset ────────────────────────────────────────────────
    tokenized = load_from_disk(args.tokenized_path)
    all_person_ids = set(cohort["person_id"].to_list())
    pred_count_map = dict(
        zip(pred_counts["person_id"].to_list(), pred_counts["n_predictions"].to_list())
    )

    print("\n=== Per-split statistics ===")
    for split_name, split_ds in tokenized.items():
        # Filter to cohort patients only
        filtered = split_ds.filter(
            lambda batch: [pid in all_person_ids for pid in batch["person_id"]],
            batched=True,
            batch_size=5000,
        )
        if len(filtered) == 0:
            print(f"\n[{split_name}] No cohort patients — skipping")
            continue

        seq_lengths = np.array([len(ids) for ids in filtered["concept_ids"]])
        n_preds = np.array([
            pred_count_map.get(pid, 0) for pid in filtered["person_id"]
        ])
        work_per_record = seq_lengths * n_preds  # proxy for total work

        print(f"\n[{split_name}]  n={len(filtered):,}")
        print(f"  sequence length  — mean={seq_lengths.mean():.0f}  "
              f"median={np.median(seq_lengths):.0f}  "
              f"p95={np.percentile(seq_lengths, 95):.0f}  "
              f"max={seq_lengths.max():.0f}")
        print(f"  prediction times — mean={n_preds.mean():.2f}  "
              f"median={np.median(n_preds):.1f}  "
              f"p95={np.percentile(n_preds, 95):.0f}  "
              f"max={n_preds.max():.0f}")
        print(f"  work proxy       — mean={work_per_record.mean():.0f}  "
              f"total={work_per_record.sum():,}")


if __name__ == "__main__":
    main()
