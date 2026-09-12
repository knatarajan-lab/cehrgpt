"""
Validate the assumptions that CEHR-GPT's age/time reconstruction makes about a tokenized.

patient-sequence dataset (e.g. CoMET/ETHOS tokens).

CEHR-GPT feeds `ages` into the transformer as `position_ids` (see
CEHRGPT2LMHeadModel.forward and CehrGptForClassification.forward), and those ages come
from `construct_age_sequence` in cehrgpt.gpt_utils:

    1. if the dataset already has an `ages` column, it is used verbatim;
    2. else, if `concept_ids[1]` starts with "age", the integer after ":" is the base age
       and it is advanced by `extract_time_interval_in_days(...) // 365` at every ATT
       token;
    3. else it silently falls back to an all-zero vector (warning only).

Path 3 - and a base age that fails to parse in path 2 - produces positions that carry no
information at all, which matters if RoPE-over-age is enabled (`apply_rotary`). This
script reports which path a dataset actually takes, and whether the ATT tokens in the
data are recognized by `is_att_token` / `extract_time_interval_in_days` at all.

Usage:
    python -m cehrgpt.tools.validate_age_token_assumptions \
        --data_folder /path/to/ethos_tokens/patient_sequence/train \
        --num_rows 5000

Exits non-zero if a hard problem is found (fallback-to-zeros rows, unparseable base ages,
crash-inducing age tokens, or ages that never advance).
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pyarrow.parquet as pq

from cehrgpt.gpt_utils import (
    construct_age_sequence,
    construct_time_sequence,
    extract_time_interval_in_days,
    is_att_token,
)

# Columns construct_age_sequence / construct_time_sequence care about.
WANTED_COLUMNS = ["concept_ids", "ages", "epoch_times"]

# Tokens that *look* like a time interval (CEHR-GPT "D7"/"W2"/"M3"/"Y1"/"VS-D7-VE"/"i-D3"
# style, or ETHOS/CoMET minute-bucket style "5m-15m"/"3mt-6mt"/">=6mt"). Used only to
# surface tokens that resemble time tokens but are NOT recognized by is_att_token.
TEMPORAL_LOOKING = re.compile(
    r"^(?:i-|VS-)?(?:>=?)?\d*\s*(?:m|mt|h|d|w|y)(?:-(?:>=?)?\d+\s*(?:m|mt|h|d|w|y))?(?:-VE)?$",
    re.IGNORECASE,
)


def find_parquet_files(data_folders: List[str]) -> List[Path]:
    """Collect parquet files from one or more folders (recursively) or direct file paths."""
    files: List[Path] = []
    for folder in data_folders:
        path = Path(folder)
        if path.is_file():
            files.append(path)
        else:
            files.extend(sorted(path.rglob("*.parquet")))
    return files


def iter_rows(
    files: List[Path], num_rows: int, batch_size: int = 512
) -> Iterator[Dict[str, Any]]:
    """Yield row dicts, restricted to the columns in WANTED_COLUMNS that actually exist."""
    yielded = 0
    for file in files:
        parquet_file = pq.ParquetFile(file)
        available = [
            column
            for column in WANTED_COLUMNS
            if column in parquet_file.schema_arrow.names
        ]
        if "concept_ids" not in available:
            raise ValueError(
                f"{file} has no `concept_ids` column; columns are "
                f"{parquet_file.schema_arrow.names}"
            )
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=available
        ):
            for row in batch.to_pylist():
                yield row
                yielded += 1
                if num_rows and yielded >= num_rows:
                    return


def as_list(value: Any) -> Optional[List[Any]]:
    """Normalize a possibly-null, possibly-ndarray sequence column into a list."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return list(value)


def summarize(values: List[float]) -> str:
    if not values:
        return "n/a"
    array = np.asarray(values, dtype=float)
    return (
        f"min={array.min():.0f} p50={np.percentile(array, 50):.0f} "
        f"p99={np.percentile(array, 99):.0f} max={array.max():.0f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a tokenized patient-sequence dataset satisfies the "
            "assumptions of construct_age_sequence / construct_time_sequence."
        )
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        nargs="+",
        help="Folder(s) (searched recursively) or parquet file(s) of patient sequences.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1000,
        help="Number of rows to inspect; 0 means all rows. Default 1000.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of example sequences to print. Default 5.",
    )
    parser.add_argument(
        "--top_tokens",
        type=int,
        default=30,
        help="How many of the most frequent tokens to print. Default 30.",
    )
    args = parser.parse_args()

    files = find_parquet_files(args.data_folder)
    if not files:
        print(f"No parquet files found under {args.data_folder}")
        return 2

    # Schemas can differ between files, so check every footer rather than just the first.
    files_with_ages = [
        file for file in files if "ages" in pq.ParquetFile(file).schema_arrow.names
    ]
    files_with_epoch_times = [
        file
        for file in files
        if "epoch_times" in pq.ParquetFile(file).schema_arrow.names
    ]

    print("=" * 78)
    print("FILES AND SCHEMA")
    print("=" * 78)
    print(f"parquet files found      : {len(files)}")
    print(f"first file               : {files[0]}")
    print(f"columns (first file)     : {pq.ParquetFile(files[0]).schema_arrow.names}")
    print(f"files with `ages`        : {len(files_with_ages)} / {len(files)}")
    print(f"files with `epoch_times` : {len(files_with_epoch_times)} / {len(files)}")
    if files_with_ages and len(files_with_ages) != len(files):
        print("WARNING: `ages` is present in some files but not others.")

    # Counters for the stored `ages` column.
    n_rows = 0
    n_short_sequences = 0
    n_rows_with_ages_column = 0
    n_ages_null = 0
    n_ages_length_mismatch = 0
    n_ages_non_monotonic = 0
    stored_age_max: List[float] = []

    # Counters for the concept_ids[1] fallback path.
    n_second_token_is_age = 0
    n_second_token_no_colon = 0
    n_base_age_unparseable = 0
    n_would_fall_back_to_zeros = 0
    n_construct_raised = 0
    second_token_counter: Counter = Counter()
    base_age_values: List[float] = []

    # Counters for the concept_ids[0] year token (construct_time_sequence).
    n_rows_with_epoch_times_column = 0
    n_first_token_is_year = 0
    first_token_counter: Counter = Counter()

    # Token-level counters.
    token_counter: Counter = Counter()
    att_token_counter: Counter = Counter()
    unrecognized_temporal_counter: Counter = Counter()
    bad_att_tokens: Counter = Counter()

    # Derived age sequences (fallback path, i.e. ages=None).
    derived_age_spans: List[float] = []
    derived_age_max: List[float] = []
    derived_total_days: List[float] = []
    n_derived_all_zero = 0
    n_derived_flat = 0

    examples: List[Dict[str, Any]] = []

    for row in iter_rows(files, args.num_rows):
        concept_ids = as_list(row.get("concept_ids"))
        if not concept_ids:
            continue
        n_rows += 1
        concept_ids = [str(token) for token in concept_ids]
        token_counter.update(concept_ids)

        if len(concept_ids) < 2:
            # construct_age_sequence indexes concept_ids[1] unconditionally.
            n_short_sequences += 1
            continue

        # ---- stored `ages` column ------------------------------------------------
        # `iter_rows` only selects columns that exist, so key presence is per-file.
        row_has_ages = "ages" in row
        if "epoch_times" in row:
            n_rows_with_epoch_times_column += 1
        stored_ages = as_list(row.get("ages")) if row_has_ages else None
        if row_has_ages:
            n_rows_with_ages_column += 1
            if stored_ages is None or len(stored_ages) == 0:
                n_ages_null += 1
            else:
                if len(stored_ages) != len(concept_ids):
                    n_ages_length_mismatch += 1
                age_array = np.asarray(stored_ages, dtype=float)
                if np.any(np.diff(age_array) < 0):
                    n_ages_non_monotonic += 1
                stored_age_max.append(float(age_array.max()))

        # ---- concept_ids[0]: year token used by construct_time_sequence ----------
        first_token = concept_ids[0]
        first_token_counter.update([first_token])
        if first_token.lower().startswith("year"):
            n_first_token_is_year += 1

        # ---- concept_ids[1]: age token used by construct_age_sequence ------------
        second_token = concept_ids[1]
        second_token_counter.update([second_token])
        if second_token.lower().startswith("age"):
            n_second_token_is_age += 1
            if ":" not in second_token:
                # construct_age_sequence does concept_ids[1].split(":")[1] OUTSIDE its
                # try/except, so this raises IndexError rather than defaulting to 0.
                n_second_token_no_colon += 1
            else:
                age_str = second_token.split(":")[1]
                try:
                    base_age_values.append(float(max(int(age_str), 0)))
                except ValueError:
                    # Caught by construct_age_sequence, which warns and uses age 0.
                    n_base_age_unparseable += 1
        else:
            n_would_fall_back_to_zeros += 1

        # ---- token-level ATT recognition -----------------------------------------
        for token in set(concept_ids):
            if is_att_token(token):
                att_token_counter.update([token])
                try:
                    extract_time_interval_in_days(token)
                except ValueError:
                    bad_att_tokens.update([token])
            elif TEMPORAL_LOOKING.match(token):
                unrecognized_temporal_counter.update([token])

        # Total elapsed days the ATT tokens in this row encode. construct_age_sequence
        # advances the age by `total_days // 365`, so a row can accumulate a lot of days
        # and still show a flat age sequence.
        total_days = 0.0
        for token in concept_ids:
            if is_att_token(token):
                try:
                    total_days += extract_time_interval_in_days(token)
                except ValueError:
                    pass
        derived_total_days.append(total_days)

        # ---- what construct_age_sequence would actually produce -------------------
        # Pass ages=None to exercise the reconstruction path regardless of the column.
        try:
            derived = np.asarray(
                construct_age_sequence(concept_ids, None), dtype=float
            )
        except Exception as error:  # noqa: BLE001 - we want to report, not crash
            print(
                f"  construct_age_sequence raised {type(error).__name__}: {error}\n"
                f"  first 4 tokens: {concept_ids[:4]}"
            )
            n_construct_raised += 1
            continue

        derived_age_max.append(float(derived.max()))
        derived_age_spans.append(float(derived.max() - derived.min()))
        if np.all(derived == 0):
            n_derived_all_zero += 1
        elif derived.max() == derived.min():
            n_derived_flat += 1

        if len(examples) < args.num_examples:
            examples.append(
                {
                    "tokens": concept_ids[:12],
                    "stored_ages": (
                        stored_ages[:12] if stored_ages is not None else None
                    ),
                    "derived_ages": derived[:12].astype(int).tolist(),
                }
            )

    if n_rows == 0:
        print("\nNo rows with a non-empty `concept_ids` column were found.")
        return 2

    def pct(count: int) -> str:
        return f"{count:>7d} ({100.0 * count / n_rows:6.2f}%)"

    print("\n" + "=" * 78)
    print(f"STORED `ages` COLUMN  (rows inspected: {n_rows})")
    print("=" * 78)
    print(f"rows with an `ages` column: {pct(n_rows_with_ages_column)}")
    if n_rows_with_ages_column == 0:
        print("No `ages` column -> construct_age_sequence must RECONSTRUCT ages.")
        print("(Note: CehrGptDataProcessor.slice_out_input_sequence reads record['ages']")
        print(" directly, so the reconstruction has to happen in the tokenization")
        print(" mapping first, which uses record.get('ages', None).)")
    else:
        print(f"rows with null/empty ages: {pct(n_ages_null)}")
        print(f"length != len(concept_ids): {pct(n_ages_length_mismatch)}")
        print(f"non-monotonic ages        : {pct(n_ages_non_monotonic)}")
        print(f"per-row max age           : {summarize(stored_age_max)}")
        print(
            "Stored ages are returned verbatim by construct_age_sequence and become "
            "position_ids."
        )

    print("\n" + "=" * 78)
    print("RECONSTRUCTION PATH: concept_ids[1] AS AN AGE TOKEN")
    print("=" * 78)
    print(f"sequences shorter than 2 tokens : {pct(n_short_sequences)}")
    print(f"concept_ids[1] starts with 'age': {pct(n_second_token_is_age)}")
    print(f"  ...but has no ':' (IndexError): {pct(n_second_token_no_colon)}")
    print(f"  ...':' present, int() failed  : {pct(n_base_age_unparseable)}")
    print(f"NOT an age token -> zeros       : {pct(n_would_fall_back_to_zeros)}")
    print(f"construct_age_sequence raised   : {pct(n_construct_raised)}")
    print(f"parsed base age                 : {summarize(base_age_values)}")
    print(f"\nmost common concept_ids[1] values (top {args.top_tokens}):")
    for token, count in second_token_counter.most_common(args.top_tokens):
        print(f"  {count:>8d}  {token!r}")

    print("\n" + "=" * 78)
    print("RECONSTRUCTION PATH: concept_ids[0] AS A YEAR TOKEN (epoch_times)")
    print("=" * 78)
    print(f"rows with an `epoch_times` column: {pct(n_rows_with_epoch_times_column)}")
    print(f"concept_ids[0] starts with 'year': {pct(n_first_token_is_year)}")
    print("(when absent, construct_time_sequence silently anchors the patient at 1985)")
    print(f"\nmost common concept_ids[0] values (top {args.top_tokens}):")
    for token, count in first_token_counter.most_common(args.top_tokens):
        print(f"  {count:>8d}  {token!r}")

    print("\n" + "=" * 78)
    print("TIME (ATT) TOKEN RECOGNITION")
    print("=" * 78)
    print(f"distinct tokens seen              : {len(token_counter)}")
    print(f"distinct tokens seen as ATT tokens: {len(att_token_counter)}")
    if att_token_counter:
        print("\nrecognized ATT tokens -> days (top 40 by row frequency):")
        for token, count in att_token_counter.most_common(40):
            try:
                days = extract_time_interval_in_days(token)
                print(f"  {count:>8d} rows  {token!r:<18} -> {days:>5} days")
            except ValueError as error:
                print(f"  {count:>8d} rows  {token!r:<18} -> ERROR: {error}")
    else:
        print("\nNO ATT tokens recognized at all. Reconstructed ages cannot advance.")

    if bad_att_tokens:
        print("\nATT tokens that is_att_token accepts but the day extractor rejects:")
        for token, count in bad_att_tokens.most_common(40):
            print(f"  {count:>8d} rows  {token!r}")

    if unrecognized_temporal_counter:
        print("\nTokens that LOOK temporal but is_att_token does NOT recognize:")
        for token, count in unrecognized_temporal_counter.most_common(40):
            print(f"  {count:>8d} rows  {token!r}")
    else:
        print("\nNo unrecognized temporal-looking tokens found.")

    print(f"\nmost frequent tokens overall (top {args.top_tokens}, by occurrence):")
    for token, count in token_counter.most_common(args.top_tokens):
        marker = "ATT" if is_att_token(token) else "   "
        print(f"  {count:>10d}  {marker}  {token!r}")

    print("\n" + "=" * 78)
    print("RESULTING AGE SEQUENCES (reconstruction path, ages=None)")
    print("=" * 78)
    print(f"all-zero age vectors      : {pct(n_derived_all_zero)}")
    print(f"non-zero but never advance: {pct(n_derived_flat)}")
    print(f"per-row age span (max-min): {summarize(derived_age_spans)}")
    print(f"per-row max age           : {summarize(derived_age_max)}")
    print(f"per-row elapsed ATT days  : {summarize(derived_total_days)}")
    print(
        "(age advances by elapsed_days // 365, so rows can accumulate many days and "
        "still show a flat age sequence)"
    )

    if examples:
        print("\nexamples (first 12 positions):")
        for i, example in enumerate(examples):
            print(f"\n  [{i}] tokens      : {example['tokens']}")
            if example["stored_ages"] is not None:
                print(f"      stored_ages : {example['stored_ages']}")
            print(f"      derived_ages: {example['derived_ages']}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    problems: List[str] = []
    if n_short_sequences:
        problems.append(
            f"{n_short_sequences} sequence(s) have fewer than 2 tokens; "
            "construct_age_sequence indexes concept_ids[1] unconditionally."
        )
    if n_second_token_no_colon:
        problems.append(
            f"{n_second_token_no_colon} row(s) have an 'age'-prefixed second token with "
            "no ':' - construct_age_sequence would raise IndexError (the split is "
            "outside its try/except)."
        )
    if n_construct_raised:
        problems.append(
            f"construct_age_sequence raised an exception on {n_construct_raised} row(s)."
        )
    if n_base_age_unparseable:
        problems.append(
            f"{n_base_age_unparseable} row(s) have an age token whose value is not an "
            "int (e.g. an 'age:100-110' bucket); base age silently becomes 0."
        )
    if n_rows_with_ages_column < n_rows and n_would_fall_back_to_zeros:
        problems.append(
            f"{n_would_fall_back_to_zeros} row(s) have no usable age token, and not "
            "every row carries an `ages` column -> all-zero position_ids."
        )
    if not att_token_counter:
        problems.append(
            "No token in the sample is recognized by is_att_token, so reconstructed "
            "ages never advance across the sequence."
        )
    if bad_att_tokens:
        problems.append(
            f"{len(bad_att_tokens)} distinct token(s) pass is_att_token but fail "
            "extract_time_interval_in_days; construct_age_sequence would raise."
        )
    if unrecognized_temporal_counter:
        problems.append(
            f"{len(unrecognized_temporal_counter)} distinct temporal-looking token(s) "
            "are not recognized as ATT tokens; ages will not advance at them."
        )
    if derived_age_spans and float(np.median(derived_age_spans)) == 0.0:
        problems.append(
            "Median reconstructed age span is 0 - age-based positions would be constant "
            "within a patient."
        )

    if problems:
        print("PROBLEMS FOUND:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("No problems found: age and time reconstruction assumptions hold.")

    positions = stored_age_max if stored_age_max else derived_age_max
    if positions:
        print(
            "\nNote for the apply_rotary decision: the largest age used as a "
            f"position_id in this sample is {max(positions):.0f}. CEHR-GPT's rotary "
            "computes sin/cos from this value directly, so any magnitude is fine; a "
            "table-lookup RoPE (Qwen2/Llama style) indexes a cache sized by sequence "
            "length instead, and would need the position values to stay below it."
        )

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
