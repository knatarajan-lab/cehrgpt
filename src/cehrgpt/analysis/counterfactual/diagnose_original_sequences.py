#!/usr/bin/env python3
"""
Diagnostics of the original patient sequences behind the prefixes of a counterfactual study.

For every patient who has a prefix (a context of Step 2, which ends right after the index drug)
this script reads the full original sequence and measures what the model would have had to
write, and what really happened, after the index drug:

* follow-up: days from the index drug to the last observation (the earliest of the last event,
  ``[END]`` and ``[DEATH]``), and the share with a full year;
* the token budget: how many tokens the first year after the index drug takes in the original
  record, and whether that is more than the generation can write
  (``--generation_max_new_tokens``, and the larger limit ``--generation_input_length`` +
  ``--generation_max_new_tokens`` minus the prefix); the days that the budget covers;
* persistence: repeat records of the index drug within the year, visits within the year, and the
  drug era (``drug_info``: the era ends 30 days after the last record of the drug class, so a
  patient with one prescription has an era of exactly 30 days);

all by the arm the patient actually received.  The same numbers on the generated trajectories
tell whether the model writes a year of follow-up with the drug like the original records do.

Usage
-----
python diagnose_original_sequences.py \\
    --patient_sequence_path  /path/to/patient_sequence/train \\
    --contexts               /path/to/swap_contexts/target_arm_ctx.parquet \\
    --drug_info              /path/to/target_contexts/drug_info \\
    --source_name thiazides --comparator_name ace_inhibitors \\
    --output_dir             /path/to/diagnostics

``--drug_info`` is the drug_info of ONE extraction (Step 1a); Step 1b writes the same patients
again with the arms swapped.  ``arm`` says which class the patient received: ``source`` is the
class that extraction used as its source (the target class for Step 1a).
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from cehrgpt.analysis.counterfactual.extract_drug_initiation_sequences import (
    DAY_SECONDS,
    VISIT_START_TOKEN,
    compute_last_observed_time,
)

SEQUENCE_COLUMNS = ["person_id", "concept_ids", "epoch_times"]


def patient_diagnostics(
    concept_ids: List[str],
    epoch_times: List[float],
    prefix_length: int,
    drug_concept_id: str,
    window_days: float = 365.0,
    budget_lower: int = 1024,
    budget_upper: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    What the original record holds after the index drug, the last token of the prefix.

    budget_lower / budget_upper: the fewest and the most tokens the generation can write for this
    patient (budget_upper defaults to budget_lower).  Returns None when the prefix does not fit
    the record.
    """
    index_position = prefix_length - 1
    if index_position < 0 or index_position >= len(epoch_times) or index_position >= len(concept_ids):
        return None
    budget_upper = budget_lower if budget_upper is None else budget_upper
    epochs = np.asarray(epoch_times[index_position:], dtype=float)
    tokens = np.asarray(concept_ids[index_position:], dtype=object)
    index_time = epochs[0]
    last_observed = compute_last_observed_time(concept_ids, epoch_times)
    follow_up_days = max(0.0, (last_observed - index_time) / DAY_SECONDS) if last_observed is not None else 0.0

    # tokens after the index drug that fall in the first window_days of the follow-up
    in_window = epochs[1:] <= index_time + min(follow_up_days, window_days) * DAY_SECONDS
    after = tokens[1:]

    def days_covered(budget: int) -> float:
        return float((epochs[min(budget, len(epochs) - 1)] - index_time) / DAY_SECONDS)

    tokens_in_window = int(in_window.sum())
    return {
        "prefix_tokens": prefix_length,
        "index_token_is_drug": bool(tokens[0] == drug_concept_id),
        "follow_up_days": follow_up_days,
        "tokens_after_index": int(len(tokens) - 1),
        "tokens_in_window": tokens_in_window,
        "visits_in_window": int(((after == VISIT_START_TOKEN) & in_window).sum()),
        "same_drug_repeats": int(((after == drug_concept_id) & in_window).sum()),
        "needs_more_than_lower_budget": tokens_in_window > budget_lower,
        "needs_more_than_upper_budget": tokens_in_window > budget_upper,
        "days_covered_lower_budget": days_covered(budget_lower),
        "days_covered_upper_budget": days_covered(budget_upper),
    }


def _glob_parquet(path: str) -> List[str]:
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    return sorted(glob.glob(path))


def read_prefix_lengths(context_paths: List[str]) -> Dict[int, int]:
    """person_id → number of tokens of the prefix (the first prefix of a patient; the arms agree)."""
    frames = []
    for path in context_paths:
        files = _glob_parquet(path)
        if not files:
            raise FileNotFoundError(f"No parquet files at {path}")
        frames += [pl.read_parquet(f, columns=["person_id", "concept_ids"]) for f in files]
    contexts = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.col("concept_ids").list.len().alias("prefix_tokens")
    )
    contexts = contexts.unique(subset="person_id", keep="first", maintain_order=True)
    return dict(zip(contexts["person_id"].to_list(), contexts["prefix_tokens"].to_list()))


def read_drug_info(path: str, era_gap_days: int, names: Dict[str, str]) -> pl.DataFrame:
    """drug_info with the arm named and the era in days (null when the era never ended)."""
    files = _glob_parquet(path)
    if not files:
        raise FileNotFoundError(f"No parquet files at {path}")
    info = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    info = info.unique(subset="person_id", keep="first", maintain_order=True)
    era_days = (pl.col("era_end_epoch_time") - pl.col("drug_epoch_time")) / DAY_SECONDS
    return info.with_columns(
        pl.col("arm").replace(names).alias("arm"),
        era_days.alias("era_days"),
        pl.col("era_end_epoch_time").is_not_null().alias("era_ended"),
        (era_days - era_gap_days).abs().lt(0.01).fill_null(False).alias("era_is_one_prescription"),
    )


def collect_diagnostics(
    sequence_files: List[str],
    prefix_lengths: Dict[int, int],
    drug_info: pl.DataFrame,
    window_days: float,
    generation_input_length: int,
    generation_max_new_tokens: int,
) -> pl.DataFrame:
    """One row per patient with a prefix, a drug_info row and an original record."""
    wanted = set(prefix_lengths) & set(drug_info["person_id"].to_list())
    drug_of = dict(zip(drug_info["person_id"].to_list(), drug_info["drug_concept_id"].to_list()))
    rows: List[Dict[str, Any]] = []
    for sequence_file in sequence_files:
        shard = pl.read_parquet(sequence_file, columns=SEQUENCE_COLUMNS)
        shard = shard.filter(pl.col("person_id").is_in(list(wanted)))
        for record in shard.iter_rows(named=True):
            person_id = record["person_id"]
            prefix_length = prefix_lengths[person_id]
            # the generation can write at most this much, when its prompt is not padded
            upper = generation_input_length + generation_max_new_tokens - min(prefix_length, generation_input_length)
            result = patient_diagnostics(
                list(record["concept_ids"]),
                list(record["epoch_times"]),
                prefix_length,
                str(drug_of[person_id]),
                window_days,
                generation_max_new_tokens,
                upper,
            )
            if result is not None:
                rows.append({"person_id": person_id, **result})
            wanted.discard(person_id)
    print(f"{len(rows):,} patients diagnosed; {len(wanted):,} had a prefix and a drug_info row but no usable record")
    return pl.DataFrame(rows).join(drug_info.select(
        "person_id", "arm", "drug_concept_id", "era_days", "era_ended", "era_is_one_prescription"
    ), on="person_id", how="left")


def describe(values: pl.Series) -> Dict[str, Optional[float]]:
    values = values.drop_nulls()
    if len(values) == 0:
        return {"n": 0, "mean": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    array = values.to_numpy().astype(float)
    p25, median, p75, p90 = np.percentile(array, [25, 50, 75, 90])
    return {"n": len(array), "mean": float(array.mean()), "p25": float(p25), "median": float(median),
            "p75": float(p75), "p90": float(p90), "max": float(array.max())}


def summarize(diagnostics: pl.DataFrame, window_days: float) -> Dict[str, Dict[str, Any]]:
    """The statistics per arm the patients actually received."""
    summary: Dict[str, Dict[str, Any]] = {}
    for (arm,), group in diagnostics.group_by("arm", maintain_order=True):
        full_year = group["follow_up_days"] >= window_days
        summary[str(arm)] = {
            "patients": len(group),
            "index_token_is_not_the_drug": int((~group["index_token_is_drug"]).sum()),
            "share_with_full_follow_up": float(full_year.mean()),
            "follow_up_days": describe(group["follow_up_days"]),
            "tokens_in_window": describe(group["tokens_in_window"]),
            "visits_in_window": describe(group["visits_in_window"]),
            "share_needing_more_than_lower_budget": float(group["needs_more_than_lower_budget"].mean()),
            "share_needing_more_than_upper_budget": float(group["needs_more_than_upper_budget"].mean()),
            "days_covered_lower_budget": describe(group["days_covered_lower_budget"]),
            "days_covered_upper_budget": describe(group["days_covered_upper_budget"]),
            "share_with_a_repeat_of_the_drug": float((group["same_drug_repeats"] > 0).mean()),
            "same_drug_repeats": describe(group["same_drug_repeats"]),
            "share_era_ended": float(group["era_ended"].mean()),
            "share_era_is_one_prescription": float(group["era_is_one_prescription"].mean()),
            "era_days_when_ended": describe(group["era_days"]),
        }
    return summary


def _line(label: str, stats: Dict[str, Optional[float]]) -> str:
    if not stats["n"]:
        return f"    {label:<34}: none"
    return (f"    {label:<34}: mean={stats['mean']:.1f}  median={stats['median']:.1f}  "
            f"IQR=[{stats['p25']:.1f}, {stats['p75']:.1f}]  p90={stats['p90']:.1f}  max={stats['max']:.1f}")


def format_report(summary: Dict[str, Dict[str, Any]], window_days: float, budgets: str) -> str:
    lines = ["=" * 60, f"Original sequences after the prefixes ({budgets})", "=" * 60]
    for arm, s in summary.items():
        lines += [
            f"\n  {arm}  (n={s['patients']:,})",
            f"    index token is not the drug     : {s['index_token_is_not_the_drug']:,}  (prefix and record disagree)",
            f"    follow-up of {window_days:g} days or more   : {s['share_with_full_follow_up']:.1%}",
            _line("follow-up (days)", s["follow_up_days"]),
            _line(f"tokens in the first {window_days:g} days", s["tokens_in_window"]),
            f"    need more tokens than the fewest: {s['share_needing_more_than_lower_budget']:.1%}",
            f"    need more tokens than the most  : {s['share_needing_more_than_upper_budget']:.1%}",
            _line("days the fewest tokens cover", s["days_covered_lower_budget"]),
            _line("days the most tokens cover", s["days_covered_upper_budget"]),
            _line(f"visits in the first {window_days:g} days", s["visits_in_window"]),
            f"    repeat record of the index drug : {s['share_with_a_repeat_of_the_drug']:.1%}",
            _line("repeats of the index drug", s["same_drug_repeats"]),
            f"    era ended (gap/death)           : {s['share_era_ended']:.1%}",
            f"    era = one prescription          : {s['share_era_is_one_prescription']:.1%}",
            _line("era days (ended)", s["era_days_when_ended"]),
        ]
    lines.append("=" * 60)
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient_sequence_path", required=True, help="The patient sequences (folder of parquet files) of Step 1")
    parser.add_argument("--contexts", nargs="+", required=True, help="Prefix parquet files or folders of Step 2 (the arm contexts)")
    parser.add_argument("--drug_info", required=True, help="drug_info folder of one extraction (Step 1a)")
    parser.add_argument("--source_name", default="source", help="Name of the arm 'source' of that extraction (the target class)")
    parser.add_argument("--comparator_name", default="comparator", help="Name of the arm 'comparator' of that extraction")
    parser.add_argument("--era_gap_days", type=int, default=30, help="ERA_GAP_DAYS of the extraction (default: 30)")
    parser.add_argument("--window_days", type=float, default=365.0, help="Follow-up window of the analysis (default: 365)")
    parser.add_argument("--generation_input_length", type=int, default=2048)
    parser.add_argument("--generation_max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", required=True, help="Where diagnostics.csv and diagnostics_summary.json are written")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sequence_files = _glob_parquet(args.patient_sequence_path)
    if not sequence_files:
        raise FileNotFoundError(f"No parquet files at {args.patient_sequence_path}")
    prefix_lengths = read_prefix_lengths(args.contexts)
    drug_info = read_drug_info(
        args.drug_info, args.era_gap_days, {"source": args.source_name, "comparator": args.comparator_name}
    )
    print(f"{len(prefix_lengths):,} prefixes, {len(drug_info):,} drug_info rows, {len(sequence_files)} sequence files")
    diagnostics = collect_diagnostics(
        sequence_files, prefix_lengths, drug_info, args.window_days,
        args.generation_input_length, args.generation_max_new_tokens,
    )
    summary = summarize(diagnostics, args.window_days)
    budgets = (f"fewest tokens written {args.generation_max_new_tokens}, "
               f"most {args.generation_input_length + args.generation_max_new_tokens} minus the prefix")
    print(format_report(summary, args.window_days, budgets))
    os.makedirs(args.output_dir, exist_ok=True)
    diagnostics.write_csv(os.path.join(args.output_dir, "diagnostics.csv"))
    with open(os.path.join(args.output_dir, "diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written to {args.output_dir}")


if __name__ == "__main__":
    main()
