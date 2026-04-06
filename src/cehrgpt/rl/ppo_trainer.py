"""
PPO-clip trainer for CEHR-GPT.

Extends CehrGptGRPOTrainer by replacing the REINFORCE policy-gradient term
with a PPO clipped surrogate objective.  All other components — rollout
generation, reward computation, KL regularisation, logging, and eval
subsampling — are inherited unchanged from CehrGptGRPOTrainer.

Algorithm per patient i
-----------------------
1–3. Same as GRPO (rollouts, rewards, baseline-adjusted advantage A_i).
4.   Old log-probs π_old: current model weights evaluated without gradient.
5.   PPO clipped surrogate (per token, averaged over rollouts and patients):
         r_t    = π_θ(a_t) / π_old(a_t)  = exp(log π_θ - log π_old)
         L_PPO  = -mean_t min(r_t · A_i, clip(r_t, 1-ε, 1+ε) · A_i)
6.   KL regularisation against π_ref (DeepSeek GRPO estimator, always ≥ 0):
         D̂_KL(t) = exp(log π_ref(t) - log π_θ(t)) + (log π_θ(t) - log π_ref(t)) - 1
7.   Total loss: L = L_PPO + β · L_KL
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cehrgpt.rl.grpo_trainer import CehrGptGRPOTrainer


class CehrGptPPOTrainer(CehrGptGRPOTrainer):
    """
    PPO-clip + KL trainer for CEHR-GPT.

    Inherits all infrastructure from ``CehrGptGRPOTrainer``; only
    ``_compute_pg_loss`` is overridden to use a clipped surrogate objective
    instead of plain REINFORCE.

    The clip epsilon ``ε`` is read from ``rl_args.ppo_clip_epsilon``.
    """

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

        Returns (ppo_loss, kl_loss) where total_loss = ppo_loss + β · kl_loss.
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

        for n, (i, _, _, _, _, _, rollout_len, new_ids_t) in enumerate(entries):
            token_lp, ref_token_lp = self._extract_token_lp(
                curr_logits, ref_logits, n, rollout_len, new_ids_t
            )

            # Old log-probs (no gradient)
            idx          = new_ids_t.unsqueeze(1)
            sl           = slice(-(rollout_len + 1), -1)
            old_token_lp = F.log_softmax(old_logits[n, sl, :], dim=-1).gather(1, idx).squeeze(1).detach()

            # PPO clipped surrogate
            log_ratio_old = token_lp - old_token_lp
            ratio         = torch.exp(log_ratio_old)
            A_i           = advantages[i]
            surr1         = ratio * A_i
            surr2         = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * A_i
            ppo_term      = -torch.min(surr1, surr2).mean()

            # GRPO KL estimator against π_ref (always ≥ 0)
            log_ratio_ref = token_lp - ref_token_lp
            kl_approx     = (torch.exp(-log_ratio_ref) + log_ratio_ref - 1).mean()

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
