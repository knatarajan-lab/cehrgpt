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


def _to_naive_day(values: pd.Series) -> pd.Series:
    values = pd.to_datetime(values)
    if values.dt.tz is not None:
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    return values.dt.normalize()


class DemographicsLookup:
    """
    Maps (person_id, calendar day of the prediction time) to gender/race concept ids.

    Lookups are vectorized: a batch is resolved with one index probe instead of a Python loop
    over a dictionary. Pairs that are missing resolve to 0. If several rows share a
    (person_id, day), the last one wins.
    """

    def __init__(self, demographics_df: Optional[pd.DataFrame]):
        self._gender = np.zeros(0, dtype=np.int64)
        self._race = np.zeros(0, dtype=np.int64)
        self._index: Optional[pd.MultiIndex] = None
        if demographics_df is None or demographics_df.empty:
            return
        frame = pd.DataFrame(
            {
                "person_id": demographics_df["person_id"].astype("int64").to_numpy(),
                "day": _to_naive_day(demographics_df["index_date"]).to_numpy(),
                "gender_concept_id": demographics_df["gender_concept_id"].to_numpy(),
                "race_concept_id": demographics_df["race_concept_id"].to_numpy(),
            }
        ).drop_duplicates(["person_id", "day"], keep="last")
        self._index = pd.MultiIndex.from_arrays([frame["person_id"], frame["day"]])
        self._gender = frame["gender_concept_id"].to_numpy()
        self._race = frame["race_concept_id"].to_numpy()

    def lookup(
        self, person_ids: Sequence[int], prediction_times: Sequence
    ) -> Tuple[List[int], List[int]]:
        person_ids = np.asarray(person_ids, dtype="int64")
        if self._index is None:
            zeros = [0] * len(person_ids)
            return zeros, list(zeros)
        days = pd.DatetimeIndex(prediction_times)
        if days.tz is not None:
            days = days.tz_convert("UTC").tz_localize(None)
        positions = self._index.get_indexer(
            pd.MultiIndex.from_arrays([person_ids, days.normalize()])
        )
        found = positions >= 0
        safe = np.where(found, positions, 0)
        gender = np.where(found, self._gender[safe], 0)
        race = np.where(found, self._race[safe], 0)
        return gender.tolist(), race.tolist()
