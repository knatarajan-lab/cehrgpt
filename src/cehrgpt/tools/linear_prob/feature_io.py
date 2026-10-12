from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Set

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
