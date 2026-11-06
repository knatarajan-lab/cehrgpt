"""
PPO-clip trainer for CEHR-GPT.

Extends CehrGptGRPOTrainer by replacing the REINFORCE policy-gradient term
with a PPO clipped surrogate objective and a learned value network for
advantage estimation.

Algorithm per patient i
-----------------------
1–2. Same as GRPO (rollouts, rewards).
3.   Value network: V(prefix_i) = value_head(last_hidden(prefix_i))
         A_i = R_i − V(prefix_i).detach()
4.   Old log-probs π_old: current model weights evaluated without gradient.
5.   PPO clipped surrogate (per token, averaged over rollouts and patients):
         r_t    = π_θ(a_t) / π_old(a_t)  = exp(log π_θ − log π_old)
         L_PPO  = −mean_t min(r_t · A_i, clip(r_t, 1−ε, 1+ε) · A_i)
6.   KL regularisation against π_ref (DeepSeek GRPO estimator, always ≥ 0):
         D̂_KL(t) = exp(log π_ref(t) − log π_θ(t)) + (log π_θ(t) − log π_ref(t)) − 1
7.   Value loss: L_V = MSE(V(prefix_i), R_i)
8.   Total loss: L = L_PPO + β · L_KL + λ_V · L_V
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cehrgpt.rl.grpo_trainer import CehrGptGRPOTrainer


class CehrGptPPOTrainer(CehrGptGRPOTrainer):
    """
    PPO-clip + value network trainer for CEHR-GPT.

    Inherits all infrastructure from ``CehrGptGRPOTrainer``.  Two methods
    are overridden:

    * ``_compute_advantages`` — uses a learned ``value_head`` attached to the
      policy model to estimate V(prefix) and compute A = R − V.
    * ``_compute_pg_loss`` — uses PPO clipped surrogate instead of REINFORCE.

    The value head (``nn.Linear(hidden_size, 1)``) is attached to
    ``self.model`` in ``__init__`` so that:
      - it is automatically moved to the correct device alongside the model;
      - its parameters are included in the Trainer's optimizer;
      - it is saved and loaded with model checkpoints.

    Hyperparameters (from ``RLArguments``):
      - ``ppo_clip_epsilon``  (ε, default 0.2)
      - ``value_loss_coef``   (λ_V, default 0.5)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.model.config.hidden_size
        self.model.value_head = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Advantage estimation via learned value network
    # ------------------------------------------------------------------

    def _compute_advantages(
        self,
        rewards_t: torch.Tensor,
        raw_model,
        prefix_input_ids: torch.Tensor,
        prefix_ages: torch.Tensor,
        prefix_epoch_times: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_values: Optional[torch.Tensor],
        prefix_value_indicators: Optional[torch.Tensor],
        prefix_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate V(prefix_i) with a linear value head on the last hidden state,
        compute A_i = R_i − V(prefix_i), and return the MSE value loss.

        The prefix is left-padded, so the last real token is always at
        position −1 in the sequence dimension.

        Returns (advantages, value_loss).
        """
        values = self._compute_values(
            raw_model, prefix_input_ids, prefix_ages, prefix_epoch_times,
            prefix_attention_mask, prefix_values, prefix_value_indicators,
        )                                                    # (B,)
        # rewards_t is (B, K); value target is the mean over K rollouts
        value_targets = rewards_t.mean(dim=1)                # (B,)
        advantages = rewards_t - values.detach().unsqueeze(1)  # (B, K)
        value_loss = F.mse_loss(values, value_targets)
        return advantages, value_loss

    def _compute_values(
        self,
        raw_model,
        prefix_input_ids: torch.Tensor,
        prefix_ages: torch.Tensor,
        prefix_epoch_times: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_values: Optional[torch.Tensor],
        prefix_value_indicators: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the policy model to obtain V(prefix_i) for each
        patient in the batch.

        Uses ``output_hidden_states=True`` to retrieve the last transformer
        layer's hidden states without an extra forward pass.  Since the prefix
        is left-padded, position −1 is always the last real token.

        Returns a (B,) tensor of value estimates (gradients enabled).
        """
        fwd_kwargs: Dict[str, Any] = {}
        if prefix_values is not None:
            fwd_kwargs["values"]           = prefix_values
            fwd_kwargs["value_indicators"] = prefix_value_indicators

        output = raw_model(
            input_ids=prefix_input_ids,
            ages=prefix_ages,
            epoch_times=prefix_epoch_times,
            attention_mask=prefix_attention_mask,
            output_hidden_states=True,
            **fwd_kwargs,
        )
        # hidden_states: tuple of (num_layers + 1) tensors, each (B, L, hidden_size)
        last_hidden = output.hidden_states[-1][:, -1, :]   # (B, hidden_size)
        return raw_model.value_head(last_hidden).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    # PPO clipped surrogate loss
    # ------------------------------------------------------------------

    def _compute_pg_loss(
        self,
        model,
        prefix_ids: torch.Tensor,
        prefix_ages: torch.Tensor,
        prefix_times: torch.Tensor,
        prefix_values: Optional[torch.Tensor],
        prefix_value_indicators: Optional[torch.Tensor],
        prefix_lengths: torch.Tensor,
        rollout_token_strs: List[List[str]],
        rollout_seq_vals: Optional[torch.Tensor],
        rollout_seq_val_masks: Optional[torch.Tensor],
        K: int,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PPO clipped surrogate + GRPO KL estimator.

        Three forward passes are performed per call:
          1. ``model`` with no_grad  → old log-probs  (π_old)
          2. ``model`` with grad     → current log-probs (π_θ)
          3. ``self.ref_model`` with no_grad → reference log-probs (π_ref)

        Returns (ppo_loss, kl_loss) where total_loss = ppo_loss + β·kl_loss + λ_V·value_loss.
        """
        B   = prefix_ids.shape[0]
        dev = prefix_ids.device
        eps = self.rl_args.ppo_clip_epsilon

        entries = self._build_entries(
            B, K, prefix_ids, prefix_ages, prefix_times,
            prefix_values, prefix_value_indicators, prefix_lengths,
            rollout_token_strs, rollout_seq_vals, rollout_seq_val_masks, dev,
        )
        if not entries:
            zero = prefix_ids.new_zeros(1, dtype=torch.float32).squeeze().requires_grad_(True)
            return zero, zero

        batch_ids, batch_ages, batch_times, batch_vals, batch_vmask, batch_attn, has_values = \
            self._left_pad_entries(entries, dev)

        fwd_kwargs: Dict[str, Any] = {}
        if has_values:
            fwd_kwargs["values"]           = batch_vals
            fwd_kwargs["value_indicators"] = batch_vmask

        # Pass 1: old policy — current weights, no gradient (π_old)
        old_logits  = self._forward(model, batch_ids, batch_ages, batch_times, batch_attn, fwd_kwargs, no_grad=True)
        # Pass 2: current policy — gradients flow (π_θ)
        curr_logits = self._forward(model, batch_ids, batch_ages, batch_times, batch_attn, fwd_kwargs, no_grad=False)
        # Pass 3: reference model — no gradient (π_ref)
        ref_logits  = self._forward(self.ref_model, batch_ids, batch_ages, batch_times, batch_attn, fwd_kwargs, no_grad=True)

        pg_per_patient: Dict[int, List[torch.Tensor]] = {}
        kl_per_patient: Dict[int, List[torch.Tensor]] = {}

        for n, (i, k, _, _, _, _, _, rollout_len, new_ids_t) in enumerate(entries):
            sl  = slice(-(rollout_len + 1), -1)
            idx = new_ids_t.unsqueeze(1)

            curr_log_probs = F.log_softmax(curr_logits[n, sl, :], dim=-1)          # (T, vocab)
            ref_log_probs  = F.log_softmax(ref_logits[n,  sl, :], dim=-1).detach() # (T, vocab)

            # Per-token log-probs of sampled tokens under current policy
            token_lp = curr_log_probs.gather(1, idx).squeeze(1)

            # Old log-probs (no gradient)
            old_token_lp = F.log_softmax(old_logits[n, sl, :], dim=-1).gather(1, idx).squeeze(1).detach()

            # PPO clipped surrogate using per-rollout advantage
            log_ratio_old = token_lp - old_token_lp
            ratio         = torch.exp(log_ratio_old)
            A_ik          = advantages[i, k]
            surr1         = ratio * A_ik
            surr2         = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * A_ik
            ppo_term      = -torch.min(surr1, surr2).mean()

            # Exact per-position KL(π_θ || π_ref) = Σ_v π_θ(v)·(log π_θ(v) − log π_ref(v))
            kl_approx = (curr_log_probs.exp() * (curr_log_probs - ref_log_probs)).sum(dim=-1).mean()

            pg_per_patient.setdefault(i, []).append(ppo_term)
            kl_per_patient.setdefault(i, []).append(kl_approx)

        pg_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        for i in sorted(pg_per_patient):
            pg_terms.append(torch.stack(pg_per_patient[i]).mean())
            kl_terms.append(torch.stack(kl_per_patient[i]).mean())

        if not pg_terms:
            zero = prefix_ids.new_zeros(1, dtype=torch.float32).squeeze().requires_grad_(True)
            return zero, zero

        return torch.stack(pg_terms).mean(), torch.stack(kl_terms).mean()
