import json
import os
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


class AsyncParquetWriter:
    """
    Writes DataFrames to parquet on background threads so the GPU loop does not wait on disk.

    At most `max_pending` writes are in flight; `write` blocks once that many are queued, which
    bounds memory. A failed write is re-raised from the next `write` or from `flush`/`close`.
    """

    def __init__(self, max_workers: int = 2, max_pending: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._max_pending = max_pending
        self._pending: Set[Future] = set()

    def _reap(self, block: bool) -> None:
        """Collect finished writes (optionally waiting for one) and re-raise their errors."""
        if block and self._pending:
            wait(self._pending, return_when=FIRST_COMPLETED)
        done = {future for future in self._pending if future.done()}
        self._pending -= done
        for future in done:
            future.result()

    def write(self, df: pd.DataFrame, path: Path) -> None:
        self._reap(block=False)
        while len(self._pending) >= self._max_pending:
            self._reap(block=True)
        self._pending.add(self._pool.submit(df.to_parquet, path))

    def flush(self) -> None:
        while self._pending:
            self._reap(block=True)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._pool.shutdown(wait=True)

    def __enter__(self) -> "AsyncParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Do not mask the original error with a write error; just stop cleanly.
            self._pool.shutdown(wait=True)


class DemographicsLookup:
    """
    Maps person_id to gender/race concept ids.

    Lookups are vectorized: a batch is resolved with one index probe instead of a Python loop
    over a dictionary. Unknown persons resolve to 0. If a person appears more than once, the
    last row wins.
    """

    def __init__(self, demographics_df: Optional[pd.DataFrame]):
        self._index: Optional[pd.Index] = None
        self._gender = np.zeros(0, dtype=np.int64)
        self._race = np.zeros(0, dtype=np.int64)
        if demographics_df is None or demographics_df.empty:
            return
        frame = demographics_df[
            ["person_id", "gender_concept_id", "race_concept_id"]
        ].drop_duplicates("person_id", keep="last")
        self._index = pd.Index(frame["person_id"].astype("int64").to_numpy())
        self._gender = frame["gender_concept_id"].to_numpy()
        self._race = frame["race_concept_id"].to_numpy()

    def lookup(self, person_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
        person_ids = np.asarray(person_ids, dtype="int64")
        if self._index is None:
            zeros = [0] * len(person_ids)
            return zeros, list(zeros)
        positions = self._index.get_indexer(person_ids)
        found = positions >= 0
        safe = np.where(found, positions, 0)
        gender = np.where(found, self._gender[safe], 0)
        race = np.where(found, self._race[safe], 0)
        return gender.tolist(), race.tolist()


# What the extraction knows about the vectors it saved, for the training to slice them
FEATURE_INFO_FILE = "feature_info.json"
GLOBAL, LOCAL, CONCATENATED = "global", "local", "concatenated"
FEATURE_SETS = (CONCATENATED, GLOBAL, LOCAL)


def write_feature_info(
    features_dir: Path,
    num_features: int,
    combined: bool,
    global_is_last_hidden_state: bool,
) -> dict:
    """
    Record how the feature vectors saved in features_dir are laid out.

    With the global and local features combined a vector is the global features followed by the local features,
    so it has twice the hidden size: [global (the linear probe state) | local (the last token state)]. Without
    them it is the global features alone. A model without a MOTOR head has no linear probe state, and its
    "global" features are the last token state too (global_is_last_hidden_state), the same as the local ones.
    """
    parts = 2 if combined else 1
    if num_features % parts:
        raise ValueError(f"{num_features} features cannot be {parts} parts of the same size")
    info = {
        "layout": [GLOBAL, LOCAL] if combined else [GLOBAL],
        "hidden_size": num_features // parts,
        "num_features": num_features,
        "global_is_last_hidden_state": bool(global_is_last_hidden_state),
    }
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    # Workers that extract shards of the same cohort write the same file: replace it in one step
    temporary = features_dir / f"{FEATURE_INFO_FILE}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(info, indent=2))
    os.replace(temporary, features_dir / FEATURE_INFO_FILE)
    return info


def read_feature_info(features_dir: Path) -> Optional[dict]:
    path = Path(features_dir) / FEATURE_INFO_FILE
    return json.loads(path.read_text()) if path.exists() else None


def resolve_feature_sets(
    requested: Optional[Sequence[str]],
    num_features: int,
    info: Optional[dict] = None,
    hidden_size: Optional[int] = None,
) -> Dict[str, Optional[slice]]:
    """
    The feature sets to train, as the columns of the saved vectors that each one uses.

    requested is None or ["all"] for every set the saved features allow, or a list of "concatenated" (the vectors as
    they were saved), "global" and "local". The global and the local features can only be told apart in vectors that
    were saved combined (twice the hidden size), which feature_info.json tells, or hidden_size when the file is missing
    (an extraction from before it was written). The first set is the vectors as saved. A set that the vectors do not
    allow is an error when it was asked for by name.
    """
    if info and info.get("num_features") not in (None, num_features):
        raise ValueError(
            f"feature_info.json describes {info['num_features']} features but the vectors have {num_features}"
        )
    hidden_size = info["hidden_size"] if info else hidden_size
    if hidden_size and num_features not in (hidden_size, 2 * hidden_size):
        # a vector is the global features (the hidden size) or the global and the local features (twice)
        raise ValueError(
            f"The features are {num_features} wide, which is neither the hidden size {hidden_size} (global features "
            f"only) nor twice it (global and local features combined)"
        )
    combined = bool(hidden_size) and num_features == 2 * hidden_size
    if info and len(info["layout"]) == 2 and not combined:
        raise ValueError(f"The features are {num_features} wide, not twice the hidden size {hidden_size}")
    # The vectors as saved are the global features alone when they are known not to be combined; when nothing says
    # how they are laid out they are just the vectors
    only_global = bool(info and len(info["layout"]) == 1) or bool(hidden_size and not combined)
    available: Dict[str, Optional[slice]] = {GLOBAL if only_global else CONCATENATED: None}
    if combined:
        available[GLOBAL] = slice(0, hidden_size)
        available[LOCAL] = slice(hidden_size, num_features)

    if not requested or list(requested) == ["all"]:
        chosen = list(available)
        if combined and info and info.get("global_is_last_hidden_state"):
            # without a MOTOR head the two halves are the same features: train them once
            chosen.remove(LOCAL)
        return {name: available[name] for name in chosen}

    unknown = [name for name in requested if name not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature sets {unknown}; the choices are {list(FEATURE_SETS)} or all")
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"The feature sets {missing} need features that were saved combined (twice the hidden size), but the "
            f"vectors have {num_features} features"
            + (" and the hidden size is not known (pass --hidden_size)" if not hidden_size else "")
            + ". Extract them with --combine_global_local_features true."
        )
    return {name: available[name] for name in dict.fromkeys(requested)}
