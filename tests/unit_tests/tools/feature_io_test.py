import threading
import time

import pandas as pd
import pytest

from cehrgpt.tools.linear_prob.feature_io import AsyncParquetWriter


def make_frame(i):
    return pd.DataFrame({"subject_id": [i, i + 1], "features": [[0.1 * i, 0.2], [0.3, 0.4 * i]]})


def test_writes_every_frame_with_identical_content(tmp_path):
    frames = {tmp_path / f"{i}.parquet": make_frame(i) for i in range(20)}
    with AsyncParquetWriter(max_workers=3, max_pending=4) as writer:
        for path, frame in frames.items():
            writer.write(frame, path)
    for path, frame in frames.items():
        pd.testing.assert_frame_equal(
            pd.read_parquet(path).applymap(lambda v: list(v) if hasattr(v, "tolist") else v),
            frame,
        )


def test_flush_waits_for_pending_writes(tmp_path):
    writer = AsyncParquetWriter(max_workers=1, max_pending=8)
    for i in range(10):
        writer.write(make_frame(i), tmp_path / f"{i}.parquet")
    writer.flush()
    assert len(list(tmp_path.glob("*.parquet"))) == 10
    writer.close()


def test_pending_writes_are_bounded(tmp_path, monkeypatch):
    in_flight, peak, lock = 0, 0, threading.Lock()
    release = threading.Event()
    real = pd.DataFrame.to_parquet

    def slow(self, path, *a, **k):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        release.wait(timeout=5)
        real(self, path, *a, **k)
        with lock:
            in_flight -= 1

    monkeypatch.setattr(pd.DataFrame, "to_parquet", slow)
    writer = AsyncParquetWriter(max_workers=2, max_pending=2)
    submitted = []

    def producer():
        for i in range(6):
            writer.write(make_frame(i), tmp_path / f"{i}.parquet")
            submitted.append(i)

    thread = threading.Thread(target=producer)
    thread.start()
    time.sleep(0.5)
    # The producer must be blocked after filling the queue, not have enqueued everything.
    assert len(submitted) <= 2
    release.set()
    thread.join(timeout=10)
    writer.close()
    assert len(list(tmp_path.glob("*.parquet"))) == 6
    assert peak <= 2


def test_write_error_is_raised_on_flush(tmp_path):
    writer = AsyncParquetWriter()
    writer.write(make_frame(0), tmp_path / "missing_dir" / "x.parquet")
    with pytest.raises(OSError):
        writer.flush()
    writer._pool.shutdown(wait=True)


def test_write_error_is_raised_by_a_later_write(tmp_path):
    writer = AsyncParquetWriter(max_workers=1, max_pending=1)
    writer.write(make_frame(0), tmp_path / "missing_dir" / "x.parquet")
    with pytest.raises(OSError):
        writer.write(make_frame(1), tmp_path / "ok.parquet")
    writer._pool.shutdown(wait=True)


def test_context_manager_does_not_mask_the_original_error(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with AsyncParquetWriter() as writer:
            writer.write(make_frame(0), tmp_path / "missing_dir" / "x.parquet")
            raise RuntimeError("boom")
