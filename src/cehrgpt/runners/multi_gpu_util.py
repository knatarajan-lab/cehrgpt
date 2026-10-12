import os
import signal
import subprocess
import sys
from typing import List, Sequence, Tuple

import torch
from transformers.utils import logging

LOG = logging.get_logger("transformers")


def resolve_gpu_ids(gpu_ids: str) -> List[str]:
    """
    Resolve --gpu_ids ("all" or comma separated indices) into CUDA_VISIBLE_DEVICES values.

    The indices are relative to the GPUs visible to this process, so the option composes
    with an existing CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=1,3 with --gpu_ids all
    runs on physical GPUs 1 and 3.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = (
        [_.strip() for _ in visible.split(",") if _.strip()] if visible else None
    )
    num_visible = (
        len(visible_ids) if visible_ids is not None else torch.cuda.device_count()
    )
    if num_visible == 0:
        raise RuntimeError("--gpu_ids was provided but no CUDA device is visible.")
    if gpu_ids.strip().lower() == "all":
        indices = list(range(num_visible))
    else:
        indices = [int(_) for _ in gpu_ids.split(",") if _.strip()]
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"--gpu_ids contains duplicates: {gpu_ids}; use --workers_per_gpu to run "
            f"several workers on one GPU."
        )
    out_of_range = [_ for _ in indices if not 0 <= _ < num_visible]
    if out_of_range:
        raise ValueError(
            f"--gpu_ids {out_of_range} out of range, {num_visible} GPU(s) are visible."
        )
    return [visible_ids[_] if visible_ids is not None else str(_) for _ in indices]


def expand_gpu_workers(gpu_ids: Sequence[str], workers_per_gpu: int) -> List[str]:
    """
    One CUDA_VISIBLE_DEVICES value per worker, interleaved across GPUs.

    ["0", "1"] with 2 workers per GPU gives ["0", "1", "0", "1"], so that capping the number
    of workers (when there are fewer samples than workers) still spreads them over all GPUs.
    """
    if workers_per_gpu < 1:
        raise ValueError(f"--workers_per_gpu must be >= 1, got {workers_per_gpu}")
    return [gpu for _ in range(workers_per_gpu) for gpu in gpu_ids]


def strip_cli_option(argv: List[str], option: str) -> List[str]:
    """Drop `option` and its value (both `--opt value` and `--opt=value`) from argv."""
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
        elif arg == option:
            skip_next = True
        elif not arg.startswith(option + "="):
            stripped.append(arg)
    return stripped


def run_workers(
    module: str,
    worker_argv: List[str],
    shard_option: str,
    shard_paths: Sequence[str],
    cuda_devices: Sequence[str],
    log_folder: str,
) -> Tuple[List[int], List[str]]:
    """
    Run `python -m <module> <worker_argv> <shard_option> <shard>` once per shard, in parallel.

    Worker i has CUDA_VISIBLE_DEVICES pinned to cuda_devices[i] and logs to
    <log_folder>/worker_<i>_gpu_<device>.log. Blocks until every worker exits and returns
    (exit codes, log paths). Workers are terminated if this process is interrupted.
    """
    os.makedirs(log_folder, exist_ok=True)
    processes: List[subprocess.Popen] = []
    log_paths: List[str] = []
    # Make sure the workers don't outlive us when we are asked to stop
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for worker_index, shard_path in enumerate(shard_paths):
            log_path = os.path.join(
                log_folder, f"worker_{worker_index}_gpu_{cuda_devices[worker_index]}.log"
            )
            log_paths.append(log_path)
            with open(log_path, "w") as log_file:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            module,
                            *worker_argv,
                            shard_option,
                            shard_path,
                        ],
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": cuda_devices[worker_index],
                        },
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                )
            LOG.info(
                "Started worker %s on GPU %s, log: %s",
                worker_index,
                cuda_devices[worker_index],
                log_path,
            )
        exit_codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        signal.signal(signal.SIGTERM, previous_handler)
    return exit_codes, log_paths
