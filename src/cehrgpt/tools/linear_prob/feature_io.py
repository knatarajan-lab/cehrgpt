from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

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
