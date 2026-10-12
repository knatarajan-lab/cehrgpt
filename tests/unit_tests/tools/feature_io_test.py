import threading
import time

import numpy as np
import pandas as pd
import pytest

from cehrgpt.tools.linear_prob.feature_io import AsyncParquetWriter, DemographicsLookup


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


def make_demographics():
    return pd.DataFrame(
        {
            "person_id": [1, 2, 2, 3],
            "gender_concept_id": [8507, 8532, 8507, 8532],
            "race_concept_id": [8527, 8516, 8527, 8515],
        }
    )


def test_demographics_lookup_by_person_id():
    lookup = DemographicsLookup(make_demographics())
    gender, race = lookup.lookup([1, 2, 3, 99])
    # person 2 appears twice, the last row wins; an unknown person falls back to 0
    assert gender == [8507, 8507, 8532, 0]
    assert race == [8527, 8527, 8515, 0]


def test_demographics_lookup_ignores_extra_columns_and_accepts_numpy_ids():
    df = make_demographics()
    df["index_date"] = pd.Timestamp("2020-01-01")
    gender, race = DemographicsLookup(df).lookup(np.array([3, 1]))
    assert gender == [8532, 8507] and race == [8515, 8527]


def test_demographics_lookup_without_data_returns_zeros():
    for lookup in (DemographicsLookup(None), DemographicsLookup(make_demographics().iloc[0:0])):
        gender, race = lookup.lookup([1, 2])
        assert gender == [0, 0] and race == [0, 0]


def test_demographics_lookup_matches_a_reference_dict_on_random_data():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame(
        {
            "person_id": rng.integers(0, 3000, n),
            "gender_concept_id": rng.choice([8507, 8532], n),
            "race_concept_id": rng.choice([8527, 8516, 8515], n),
        }
    )
    reference = {
        row.person_id: (row.gender_concept_id, row.race_concept_id)
        for row in df.itertuples()
    }
    query = rng.integers(0, 3500, 2000)
    gender, race = DemographicsLookup(df).lookup(query)
    assert list(zip(gender, race)) == [reference.get(int(i), (0, 0)) for i in query]
