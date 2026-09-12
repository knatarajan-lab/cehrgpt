"""
Trainer callback that localises training instability instead of reporting it after the
fact.

On a fixed step interval it logs the gradient norm, the largest absolute logit, the
token-embedding norm, the final-norm weight norm, and the per-layer gradient norms.
Optionally stops the run the first time a non-finite gradient appears, so the offending
step is the last thing in the log rather than being buried under thousands of NaN steps.

The per-layer breakdown is the point: gradient growth spread evenly across layers points
at the learning rate or the regularisation, whereas growth concentrated in the last layers
points at the logit/embedding path.

Two ways of reading gradients are implemented, because the callback event that would give
pre-clip values does not exist in every transformers version:

  * `on_pre_optimizer_step`, when the installed transformers defines it. Verified absent
    from both 4.40.0 and the pinned 4.44.1, so in practice this repo takes the path below.
  * an optimizer step pre-hook, installed on the unwrapped optimizer. In 4.44.1 the
    trainer clips immediately before `optimizer.step()`, so these values are POST-clip.

Post-clip is fine for the question this is meant to answer. Clipping rescales every
gradient by one shared factor, so the per-layer *distribution* is exact; only the absolute
scale is capped at `max_grad_norm`. The trainer already logs the pre-clip total as
`grad_norm`, so a pre-clip per-layer value can be recovered when clipping was active:

    per_layer_preclip = per_layer_logged * (trainer_grad_norm / max_grad_norm)

The log line states which mode produced the numbers.

Usage - in a runner, or any script that builds a Trainer:

    from cehrgpt.tools.instability_probe import InstabilityProbe

    trainer.add_callback(InstabilityProbe(every=25, abort_on_nonfinite=True))
"""

from typing import Optional

import torch
from transformers import TrainerCallback
from transformers import __version__ as transformers_version
from transformers.utils import logging

LOG = logging.get_logger("transformers")

# `on_pre_optimizer_step` was added in transformers 4.42.
SUPPORTS_PRE_OPTIMIZER_STEP = hasattr(TrainerCallback, "on_pre_optimizer_step")


class InstabilityProbe(TrainerCallback):
    def __init__(
        self,
        every: int = 25,
        abort_on_nonfinite: bool = True,
        per_layer: bool = True,
        residual_scale: bool = True,
    ):
        self.every = every
        self.abort_on_nonfinite = abort_on_nonfinite
        self.per_layer = per_layer
        self.residual_scale = residual_scale
        self._model = None
        self._hook_handle = None
        self._block_hooks = []
        self._block_rms = {}
        self._qk_rms = {}
        self._capture_rms = False
        self._step = 0
        self._should_stop = False
        self._clipped = not SUPPORTS_PRE_OPTIMIZER_STEP

    def _install_block_hooks(self, model):
        """
        Record the RMS of the hidden states entering each decoder block.

        RMSNorm's Jacobian scales as 1/rms, so a residual stream that grows over training
        progressively shrinks the gradient reaching each block's weights. Block 0 is
        immune - its input is the embedding, whose scale is pinned - which produces a
        profile where L0 holds steady while deeper layers starve. Measuring the per-block
        input scale alongside the per-block gradient tells you whether that is what is
        happening, rather than leaving it as an inference from the gradient shape alone.
        """
        for index, block in enumerate(self._decoder_blocks(model)):

            def hook(module, args, kwargs=None, _index=index):
                if not self._capture_rms:
                    return
                hidden = args[0] if args else None
                if isinstance(hidden, torch.Tensor):
                    with torch.no_grad():
                        self._block_rms[_index] = (
                            hidden.detach().float().pow(2).mean().sqrt().item()
                        )

            self._block_hooks.append(block.register_forward_pre_hook(hook))

            # Scale of the query/key vectors entering the attention logits. This is the
            # quantity QK-norm is meant to bound: without it nothing stops the q/k
            # projections from growing, and the logits grow with them. With QK-norm on,
            # these should stay pinned near the learned gain regardless of how the
            # projection weights drift.
            attention = getattr(block, "attn", None)
            for label, projection in (
                ("q", getattr(attention, "q_proj", None)),
                ("k", getattr(attention, "k_proj", None)),
            ):
                if projection is None:
                    continue

                def qk_hook(module, args, output, _index=index, _label=label):
                    if not self._capture_rms:
                        return
                    if isinstance(output, torch.Tensor):
                        with torch.no_grad():
                            self._qk_rms[(_index, _label)] = (
                                output.detach().float().pow(2).mean().sqrt().item()
                            )

                self._block_hooks.append(projection.register_forward_hook(qk_hook))

    # ---- lifecycle -------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self._model = model
        if self.residual_scale and model is not None:
            self._install_block_hooks(model)
        if SUPPORTS_PRE_OPTIMIZER_STEP:
            LOG.info(
                "InstabilityProbe active (pre-clip gradients, every %d steps)", self.every
            )
            return
        # Accelerate wraps the optimizer in AcceleratedOptimizer, which inherits
        # register_step_pre_hook but never initialises the dict backing it, so calling it
        # on the wrapper raises. Unwrap to the real optimizer first.
        target = optimizer
        for _ in range(3):
            if target is None or hasattr(target, "_optimizer_step_pre_hooks"):
                break
            target = getattr(target, "optimizer", None)

        if target is not None and hasattr(target, "register_step_pre_hook"):
            try:
                self._hook_handle = target.register_step_pre_hook(self._optimizer_hook)
            except Exception as error:  # noqa: BLE001 - must never break training
                LOG.error(
                    "InstabilityProbe could not attach a step pre-hook (%s: %s). No "
                    "gradient diagnostics will be produced for this run.",
                    type(error).__name__,
                    error,
                )
                return
            LOG.info(
                "InstabilityProbe attached via an optimizer step pre-hook on %s "
                "(transformers %s has no on_pre_optimizer_step event). Gradient values "
                "are POST-clip: per-layer ratios are exact, absolute magnitudes are "
                "capped at max_grad_norm=%s. Recover pre-clip per-layer values as "
                "logged_value * (trainer grad_norm / max_grad_norm).",
                type(target).__name__,
                transformers_version,
                getattr(args, "max_grad_norm", None),
            )
        else:
            LOG.error(
                "InstabilityProbe could not attach: no on_pre_optimizer_step hook and no "
                "usable optimizer step pre-hook. No gradient diagnostics will be "
                "produced for this run."
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        for handle in self._block_hooks:
            handle.remove()
        self._block_hooks = []

    def on_step_begin(self, args, state, control, **kwargs):
        self._step = state.global_step
        # Only measure residual scale on steps that will be reported, so the hooks cost
        # nothing on the other 24 out of 25 steps.
        self._capture_rms = self.residual_scale and (
            self.every > 0 and state.global_step % self.every == 0
        )

    def on_step_end(self, args, state, control, **kwargs):
        # The fallback path cannot touch `control`, so honour the abort here.
        if self._should_stop:
            control.should_training_stop = True

    # ---- gradient reading ------------------------------------------------------

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self._safe_report(state.global_step, model or self._model, control)

    def _optimizer_hook(self, optimizer, hook_args, hook_kwargs):
        self._safe_report(self._step, self._model, None)

    def _safe_report(self, step, model, control):
        """A diagnostic must never be the thing that kills a long training run."""
        try:
            self._report(step, model, control)
        except Exception as error:  # noqa: BLE001
            LOG.error(
                "InstabilityProbe failed at step %s (%s: %s); continuing training.",
                step,
                type(error).__name__,
                error,
            )

    # ---- reporting -------------------------------------------------------------

    @staticmethod
    def _decoder_blocks(model):
        backbone = getattr(model, "cehrgpt", None)
        if backbone is None:
            return []
        return list(getattr(backbone, "h", []))

    def _report(self, step, model, control):
        if model is None:
            return

        total_sq = 0.0
        nonfinite = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                nonfinite.append(name)
            else:
                total_sq += grad.float().pow(2).sum().item()

        if nonfinite:
            LOG.error(
                "step %d: %d parameter(s) have non-finite gradients; first: %s",
                step,
                len(nonfinite),
                nonfinite[:5],
            )
            if self.abort_on_nonfinite:
                self._should_stop = True
                if control is not None:
                    control.should_training_stop = True
            return

        if self.every <= 0 or step % self.every != 0:
            return

        scope = "post-clip" if self._clipped else "pre-clip"
        backbone = getattr(model, "cehrgpt", None)
        blocks = self._decoder_blocks(model)

        def grad_norm(parameters):
            total = 0.0
            for parameter in parameters:
                if parameter.grad is not None:
                    total += parameter.grad.detach().float().pow(2).sum().item()
            return total**0.5

        embedding = model.get_input_embeddings()
        # The tied embedding / lm_head sits OUTSIDE every decoder block, so a per-layer
        # breakdown alone cannot say whether growth lives there. Report its gradient, not
        # just its weight norm.
        embedding_grad = (
            grad_norm([embedding.weight]) if embedding is not None else float("nan")
        )
        final_norm = getattr(backbone, "ln_f", None) if backbone else None
        final_norm_grad = (
            grad_norm([final_norm.weight]) if final_norm is not None else float("nan")
        )
        block_total = grad_norm(
            [p for block in blocks for p in block.parameters()]
        )

        with torch.no_grad():
            embedding_norm = (
                embedding.weight.norm().item() if embedding is not None else float("nan")
            )
            final_norm_value = (
                final_norm.weight.norm().item()
                if final_norm is not None
                else float("nan")
            )

        message = (
            f"InstabilityProbe step {step}: grad_norm[{scope}]={total_sq ** 0.5:.4f} "
            f"blocks={block_total:.4f} embed_grad={embedding_grad:.4f} "
            f"ln_f_grad={final_norm_grad:.4f} embedding_norm={embedding_norm:.3f} "
            f"final_norm_w={final_norm_value:.3f}"
        )

        if self.per_layer and blocks:
            per_layer = [
                f"L{index}={grad_norm(list(block.parameters())):.4f}"
                for index, block in enumerate(blocks)
            ]
            message += " | grad " + " ".join(per_layer)
        if self.residual_scale and self._block_rms:
            per_block_rms = [
                f"L{index}={self._block_rms[index]:.3f}"
                for index in sorted(self._block_rms)
            ]
            message += " | in_rms " + " ".join(per_block_rms)
            # Gradient starvation from a growing residual stream shows up as the ratio
            # between the last and first block's input scale drifting upward over
            # training while the deep-layer gradients fall.
            first = self._block_rms.get(0)
            last = self._block_rms.get(max(self._block_rms))
            if first and last:
                message += f" (last/first={last / first:.2f})"
        if self.residual_scale and self._qk_rms:
            indices = sorted({index for index, _ in self._qk_rms})
            q_values = " ".join(
                f"L{index}={self._qk_rms[(index, 'q')]:.3f}"
                for index in indices
                if (index, "q") in self._qk_rms
            )
            k_values = " ".join(
                f"L{index}={self._qk_rms[(index, 'k')]:.3f}"
                for index in indices
                if (index, "k") in self._qk_rms
            )
            message += f" | q_rms {q_values} | k_rms {k_values}"
        LOG.info(message)


def assert_no_fully_masked_rows(attention_mask: torch.Tensor) -> Optional[str]:
    """
    Check a 4D additive attention mask for rows that attend to nothing.

    Returns a description of the offending rows, or None when the mask is sound. SDPA's
    fused kernels return undefined values for fully-masked rows, so this is worth
    asserting directly when diagnosing a suspect run.
    """
    if attention_mask is None or attention_mask.dim() != 4:
        return None
    floor = torch.finfo(attention_mask.dtype).min
    attends = (attention_mask > floor).any(dim=-1)
    if attends.all():
        return None
    offending = (~attends).nonzero()
    return (
        f"{offending.shape[0]} fully masked row(s); SDPA fused kernels return undefined "
        f"values for these. First few: {offending[:5].tolist()}"
    )
