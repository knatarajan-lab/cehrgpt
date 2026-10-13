"""
Trainer callback that profiles a few training steps and logs where the GPU time goes.

Enable it with --profile_gpu_steps N. After --profile_gpu_start_step steps of warm-up (the
first steps include compilation and allocator warm-up, which would drown everything else),
it profiles the next N steps, logs the average seconds per step and a table of the most
expensive operations, and writes a Chrome trace next to the checkpoints.

How to read the table:
  * the `motor_loss` row is the forward of the MOTOR loss (head matmuls and loss);
  * kernels named `triton_...` come from torch.compile; if --motor_fused_loss is working you
    see fused `triton_*` reductions and not a long list of aten elementwise kernels with
    (rows, bins, vocab)-sized outputs;
  * the dense matmuls (`aten::mm`, `aten::bmm`, `flash_attn`) are the transformer itself.
"""
import os
import time
from typing import Optional

import torch
from transformers import TrainerCallback
from transformers.utils import logging

LOG = logging.get_logger("transformers")


class GpuProfileCallback(TrainerCallback):
    def __init__(self, num_steps: int, start_step: int = 20, row_limit: int = 30):
        self.num_steps = num_steps
        self.start_step = start_step
        self.row_limit = row_limit
        self._profiler: Optional[torch.profiler.profile] = None
        self._started_at = 0.0
        self.done = False

    @staticmethod
    def _synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.done or self._profiler is not None or state.global_step != self.start_step:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        LOG.info(
            "Profiling %s training steps starting at step %s", self.num_steps, state.global_step
        )
        self._synchronize()
        self._profiler = torch.profiler.profile(activities=activities)
        self._profiler.__enter__()
        self._started_at = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._profiler is None or state.global_step < self.start_step + self.num_steps:
            return
        self._synchronize()
        elapsed = time.perf_counter() - self._started_at
        self._profiler.__exit__(None, None, None)
        profiler, self._profiler, self.done = self._profiler, None, True
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        LOG.info(
            "Profiled %s steps: %.3f seconds per step on average",
            self.num_steps,
            elapsed / self.num_steps,
        )
        LOG.info(
            "Top operations by %s over the profiled steps:\n%s",
            sort_key,
            profiler.key_averages().table(sort_by=sort_key, row_limit=self.row_limit),
        )
        trace_path = os.path.join(args.output_dir, "gpu_profile_trace.json")
        try:
            profiler.export_chrome_trace(trace_path)
            LOG.info("Wrote the profile trace to %s", trace_path)
        except Exception as error:  # noqa: BLE001 - the table above is the main output
            LOG.warning("Could not write the profile trace: %s", error)
