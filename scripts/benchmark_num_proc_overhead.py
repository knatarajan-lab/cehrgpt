"""
Benchmark: does HuggingFace Dataset.map() process-startup overhead dominate
for small splits when num_proc is large?

Hypothesis: wall-clock time is roughly constant across dataset sizes for a
given num_proc, because forking N processes costs a fixed amount regardless
of how many records each worker processes.

Run:
    python scripts/benchmark_num_proc_overhead.py
"""
import time

import datasets
import numpy as np
import pandas as pd

datasets.disable_progress_bar()
datasets.disable_caching()


def noop(batch):
    """Trivial map function — no real computation so we measure only overhead."""
    return batch


def make_dataset(n: int) -> datasets.Dataset:
    rng = np.random.default_rng(42)
    return datasets.Dataset.from_dict(
        {
            "x": rng.integers(0, 1000, size=n).tolist(),
            "y": rng.random(n).tolist(),
        }
    )


SIZES = [500, 2_000, 10_000, 50_000, 200_000]
NUM_PROCS = [1, 2, 4, 8, 16]
BATCH_SIZE = 1000

rows = []
for n in SIZES:
    ds = make_dataset(n)
    for num_proc in NUM_PROCS:
        if num_proc > n:
            continue
        t0 = time.perf_counter()
        ds.map(noop, batched=True, batch_size=BATCH_SIZE, num_proc=num_proc)
        elapsed = time.perf_counter() - t0
        throughput = n / elapsed
        rows.append(
            {
                "dataset_size": n,
                "num_proc": num_proc,
                "wall_sec": round(elapsed, 2),
                "examples_per_sec": round(throughput, 0),
            }
        )
        print(
            f"  size={n:>7,}  num_proc={num_proc:>2}  "
            f"time={elapsed:5.2f}s  throughput={throughput:>10,.0f} ex/s"
        )

print()
df = pd.DataFrame(rows)
pivot = df.pivot(index="dataset_size", columns="num_proc", values="wall_sec")
pivot.index.name = "size \\ num_proc"
print("Wall-clock seconds (rows=dataset size, cols=num_proc):")
print(pivot.to_string())

print()
pivot2 = df.pivot(index="dataset_size", columns="num_proc", values="examples_per_sec")
pivot2.index.name = "size \\ num_proc"
print("Throughput examples/s (rows=dataset size, cols=num_proc):")
print(pivot2.to_string())
