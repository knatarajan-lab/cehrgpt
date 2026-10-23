"""
Test whether HuggingFace Dataset.map() parallelism is limited by physical shard count.

If shard count limits workers: a 1-shard dataset mapped with num_proc=16
will show throughput ≈ 1-shard throughput (num_proc=1).

If sharding is row-based: throughput scales with num_proc regardless of shards.

Run: python scripts/benchmark_shard_count.py
"""
import time
import tempfile
import os

import datasets
import numpy as np

datasets.disable_progress_bar()


def slow_fn(batch):
    """Simulate realistic work: iterate over a string array of ~1000 items."""
    results = []
    for seq in batch["seq"]:
        count = sum(1 for tok in seq if tok.startswith("D"))
        results.append(count)
    batch["count"] = results
    return batch


def make_dataset_n_shards(n_rows: int, n_shards: int, tmpdir: str) -> datasets.Dataset:
    rng = np.random.default_rng(42)
    tokens = ["D" + str(i) for i in range(50)] + ["cond" + str(i) for i in range(50)]
    data = {
        "seq": [
            [tokens[rng.integers(0, 100)] for _ in range(1000)]
            for _ in range(n_rows)
        ]
    }
    ds = datasets.Dataset.from_dict(data)
    path = os.path.join(tmpdir, f"ds_{n_rows}_{n_shards}")
    ds.save_to_disk(path, num_shards=n_shards)
    return datasets.load_from_disk(path)


N_ROWS = 2000
NUM_PROCS = [1, 4, 16]

with tempfile.TemporaryDirectory() as tmpdir:
    for n_shards in [1, 4, 16]:
        ds = make_dataset_n_shards(N_ROWS, n_shards, tmpdir)
        print(f"\n--- {n_shards} physical shard(s), {N_ROWS} rows ---")
        for num_proc in NUM_PROCS:
            t0 = time.perf_counter()
            ds.map(slow_fn, batched=True, batch_size=200, num_proc=num_proc)
            elapsed = time.perf_counter() - t0
            print(f"  num_proc={num_proc:>2}  time={elapsed:.2f}s  "
                  f"throughput={N_ROWS / elapsed:,.0f} ex/s")
