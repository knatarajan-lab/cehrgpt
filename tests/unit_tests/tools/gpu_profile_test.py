import os
from types import SimpleNamespace

import torch

from cehrgpt.tools import gpu_profile
from cehrgpt.tools.gpu_profile import GpuProfileCallback


def run_steps(callback, output_dir, total_steps):
    args = SimpleNamespace(output_dir=str(output_dir))
    state = SimpleNamespace(global_step=0)
    weight = torch.randn(64, 64, requires_grad=True)
    for _ in range(total_steps):
        callback.on_step_begin(args, state, None)
        with torch.profiler.record_function("motor_loss"):
            (torch.randn(32, 64) @ weight).relu().sum().backward()
        state.global_step += 1
        callback.on_step_end(args, state, None)


def test_profiles_the_requested_window_and_logs_a_table(tmp_path):
    callback = GpuProfileCallback(num_steps=3, start_step=2)
    with __import__("unittest").TestCase().assertLogs(gpu_profile.LOG.name, level="INFO") as logs:
        run_steps(callback, tmp_path, total_steps=8)
    text = "\n".join(logs.output)
    assert callback.done
    assert "Profiling 3 training steps starting at step 2" in text
    assert "seconds per step on average" in text
    assert "motor_loss" in text  # the named range shows up in the table
    assert os.path.exists(tmp_path / "gpu_profile_trace.json")


def test_profiles_only_once(tmp_path):
    callback = GpuProfileCallback(num_steps=1, start_step=1)
    with __import__("unittest").TestCase().assertLogs(gpu_profile.LOG.name, level="INFO") as logs:
        run_steps(callback, tmp_path, total_steps=10)
    assert sum("Top operations" in line for line in logs.output) == 1


def test_does_nothing_before_the_start_step(tmp_path):
    callback = GpuProfileCallback(num_steps=3, start_step=50)
    run_steps(callback, tmp_path, total_steps=5)
    assert not callback.done and callback._profiler is None
    assert not os.path.exists(tmp_path / "gpu_profile_trace.json")
