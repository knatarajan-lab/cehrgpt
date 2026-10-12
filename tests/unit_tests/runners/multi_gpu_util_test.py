import json
import os
import sys
import textwrap
from unittest import mock

import pytest

from cehrgpt.runners import multi_gpu_util
from cehrgpt.runners.multi_gpu_util import (
    expand_gpu_workers,
    resolve_gpu_ids,
    run_workers,
    strip_cli_option,
)


def test_resolve_gpu_ids_composes_with_cuda_visible_devices():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3,5"}):
        assert resolve_gpu_ids("0,2") == ["1", "5"]
        assert resolve_gpu_ids("all") == ["1", "3", "5"]


def test_resolve_gpu_ids_without_cuda_visible_devices():
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        multi_gpu_util.torch.cuda, "device_count", return_value=4
    ):
        assert resolve_gpu_ids("all") == ["0", "1", "2", "3"]
        assert resolve_gpu_ids("3,1") == ["3", "1"]


def test_resolve_gpu_ids_rejects_bad_input():
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}):
        with pytest.raises(ValueError, match="out of range"):
            resolve_gpu_ids("0,2")
        with pytest.raises(ValueError, match="workers_per_gpu"):
            resolve_gpu_ids("0,0")
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), mock.patch.object(
        multi_gpu_util.torch.cuda, "device_count", return_value=0
    ):
        with pytest.raises(RuntimeError, match="no CUDA device"):
            resolve_gpu_ids("all")


def test_expand_gpu_workers_interleaves_gpus():
    assert expand_gpu_workers(["0", "1"], 1) == ["0", "1"]
    assert expand_gpu_workers(["0", "1"], 2) == ["0", "1", "0", "1"]
    assert expand_gpu_workers(["5"], 3) == ["5", "5", "5"]
    # Keeping only the first workers (when there are few samples) still covers every GPU
    assert expand_gpu_workers(["0", "1", "2"], 2)[:3] == ["0", "1", "2"]
    with pytest.raises(ValueError):
        expand_gpu_workers(["0"], 0)


def test_strip_cli_option_handles_both_forms():
    argv = ["--a", "1", "--gpu_ids", "0,1", "--b=2", "--gpu_ids=0,1", "--c"]
    assert strip_cli_option(argv, "--gpu_ids") == ["--a", "1", "--b=2", "--c"]
    assert strip_cli_option(["--gpu_ids_other", "x"], "--gpu_ids") == [
        "--gpu_ids_other",
        "x",
    ]


@pytest.fixture
def recorder_module(tmp_path, monkeypatch):
    """A module that records how it was launched, importable with `python -m`."""
    out = tmp_path / "records"
    out.mkdir()
    (tmp_path / "fake_worker.py").write_text(
        textwrap.dedent(
            f"""
            import json, os, sys
            shard = sys.argv[sys.argv.index("--shard") + 1]
            record = {{
                "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "argv": sys.argv[1:],
            }}
            json.dump(record, open(os.path.join({str(out)!r}, os.path.basename(shard) + ".json"), "w"))
            print("hello from", shard)
            sys.exit(7 if shard.endswith("fail") else 0)
            """
        )
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return "fake_worker", out


def test_run_workers_pins_gpus_passes_shards_and_logs(tmp_path, recorder_module):
    module, out = recorder_module
    codes, logs = run_workers(
        module=module,
        worker_argv=["--x", "1"],
        shard_option="--shard",
        shard_paths=["s0", "s1", "s2"],
        cuda_devices=["4", "4", "6"],
        log_folder=str(tmp_path / "logs"),
    )
    assert codes == [0, 0, 0]
    for i, (shard, cuda) in enumerate([("s0", "4"), ("s1", "4"), ("s2", "6")]):
        record = json.loads((out / f"{shard}.json").read_text())
        assert record["cuda"] == cuda
        assert record["argv"] == ["--x", "1", "--shard", shard]
        assert logs[i].endswith(f"worker_{i}_gpu_{cuda}.log")
        assert f"hello from {shard}" in open(logs[i]).read()


def test_run_workers_reports_failures_without_raising(tmp_path, recorder_module):
    module, _ = recorder_module
    codes, logs = run_workers(
        module=module,
        worker_argv=[],
        shard_option="--shard",
        shard_paths=["ok", "fail"],
        cuda_devices=["0", "1"],
        log_folder=str(tmp_path / "logs"),
    )
    assert codes == [0, 7]
    assert "hello from fail" in open(logs[1]).read()


def test_run_workers_restores_the_sigterm_handler(tmp_path, recorder_module):
    import signal

    module, _ = recorder_module
    before = signal.getsignal(signal.SIGTERM)
    run_workers(module, [], "--shard", ["ok"], ["0"], str(tmp_path / "logs"))
    assert signal.getsignal(signal.SIGTERM) == before
