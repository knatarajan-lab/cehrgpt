"""
GRPO (Group Relative Policy Optimization / REINFORCE + KL) trainer for CEHR-GPT.

Algorithm per patient i
-----------------------
1. Sample K rollout trajectories τ_1…τ_K ~ π_θ(· | prefix_i)
2. Compute reward R_i using the weighted condition-recovery objective
3. Compute advantage  A_i = R_i - b  (b = moving-average baseline)
4. Policy-gradient loss:
       L_PG = - A_i · (1/K) Σ_k Σ_t log π_θ(τ_k[t] | prefix_i + τ_k[:t])
5. KL regularisation (token-level log-ratio approximation):
       L_KL = (1/K) Σ_k mean_t [log π_θ(τ_k[t]|·) - log π_ref(τ_k[t]|·)]
6. Total loss: L = L_PG + β · L_KL
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.utils import logging

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token
from cehrgpt.rl.reward import compute_patient_reward, extract_conditions_from_rollout
from cehrgpt.runners.hf_gpt_rl_runner_argument_dataclass import RLArguments

LOG = logging.get_logger("transformers")

_SECONDS_PER_DAY = 86400.0


class CehrGptGRPOTrainer(Trainer):
    """
    REINFORCE + KL trainer for CEHR-GPT.

    Args:
        ref_model: Frozen reference model π_ref (same architecture as the policy).
        rl_args: ``RLArguments`` hyperparameter container.
        prevalence_stats: Dict mapping (concept_id, window_days) → π_{c,w}.
        target_concept_ids: Set of condition concept ID strings used for rewards.
        cehrgpt_tokenizer: ``CehrGptTokenizer`` instance (stored separately from the
            HF ``tokenizer`` slot which may not be set for GPT-style models).
        All remaining kwargs are forwarded to ``CehrGptTrainer``.
    """

    def __init__(
        self,
        ref_model,
        rl_args: RLArguments,
        prevalence_stats: Dict[Tuple[str, int], float],
        target_concept_ids: Set[str],
        cehrgpt_tokenizer,
        eval_sample_size: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._eval_sample_size = eval_sample_size
        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)


        self.rl_args = rl_args
        self.prevalence_stats = prevalence_stats
        self.target_concept_ids = target_concept_ids
        self.cehrgpt_tokenizer = cehrgpt_tokenizer
        self._baseline: float = 0.0
        # Accumulators for RL metrics — flushed and averaged in log()
        self._rl_metric_sums: Dict[str, float] = {}
        self._rl_metric_counts: Dict[str, int] = {}
        # _signature_columns is normally set by _remove_unused_columns, which is
        # skipped when remove_unused_columns=False.  Initialise it here so that
        # the empty-batch error message in _prepare_inputs doesn't crash.
        if self._signature_columns is None:
            self._signature_columns = []

    # ------------------------------------------------------------------
    # Logging — flush accumulated RL metrics at each logging step
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, Any], *args, **kwargs):
        if self._rl_metric_sums:
            for name, total in self._rl_metric_sums.items():
                count = self._rl_metric_counts.get(name, 1)
                logs[name] = round(total / count, 6)
            self._rl_metric_sums = {}
            self._rl_metric_counts = {}
        super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # Evaluation — subsample eval set each time to keep eval fast
    # ------------------------------------------------------------------

    def get_eval_dataloader(self, eval_dataset=None):
        import random
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is not None and len(dataset) > self._eval_sample_size:
            indices = random.sample(range(len(dataset)), self._eval_sample_size)
            dataset = dataset.select(indices)
        return super().get_eval_dataloader(dataset)

    # ------------------------------------------------------------------
    # Evaluation step — route through compute_loss instead of model(**inputs)
    # ------------------------------------------------------------------

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    # ------------------------------------------------------------------
    # Training step — skip empty batches from the collator
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, **kwargs):
        # The RL collator returns {} when no example passes min_prefix_visits.
        # _prepare_inputs raises ValueError on empty dict before compute_loss
        # is reached, so intercept here.  Still flow a zero loss through all
        # parameters so DDP's gradient reducer marks every bucket ready.
        if not inputs:
            raw = self.accelerator.unwrap_model(model)
            loss = sum(p.sum() * 0.0 for p in raw.parameters() if p.requires_grad)
            self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps
        return super().training_step(model, inputs, **kwargs)

    # ------------------------------------------------------------------
    # Main loss entry point
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Empty batch (all examples filtered by collator) or non-RL batch.
        # With DDP, we must still flow through all model parameters so the
        # gradient reducer sees every bucket and does not hang waiting for them.
        if not inputs or "prefix_input_ids" not in inputs:
            raw = self.accelerator.unwrap_model(model)
            return sum(p.sum() * 0.0 for p in raw.parameters() if p.requires_grad)

        prefix_input_ids: torch.Tensor = inputs["prefix_input_ids"]           # (B, L)
        prefix_ages: torch.Tensor = inputs["prefix_ages"]                   # (B, L)
        prefix_epoch_times: torch.Tensor = inputs["prefix_epoch_times"]     # (B, L)
        prefix_attention_mask: torch.Tensor = inputs["prefix_attention_mask"]  # (B, L)
        prefix_values: Optional[torch.Tensor] = inputs.get("prefix_values")  # (B, L)
        prefix_value_indicators: Optional[torch.Tensor] = inputs.get("prefix_value_indicators")  # (B, L)
        future_conditions: List[List[Tuple[str, float]]] = inputs["future_conditions"]
        prefix_lengths: torch.Tensor = inputs["prefix_lengths"]              # (B,)

        B = prefix_input_ids.shape[0]
        K = self.rl_args.num_rollouts

        # ---------------------------------------------------------------
        # 1. Sample K rollouts per patient
        # ---------------------------------------------------------------
        # Expand prefix K-fold: (B*K, L)
        rep_ids = prefix_input_ids.repeat_interleave(K, dim=0)
        rep_ages = prefix_ages.repeat_interleave(K, dim=0)
        rep_mask = prefix_attention_mask.repeat_interleave(K, dim=0)
        rep_values = prefix_values.repeat_interleave(K, dim=0) if prefix_values is not None else None
        rep_val_masks = prefix_value_indicators.repeat_interleave(K, dim=0) if prefix_value_indicators is not None else None

        # Unwrap DataParallel / DistributedDataParallel so we can call .generate()
        raw_model = self.accelerator.unwrap_model(model)
        raw_model.eval()
        with torch.no_grad():
            gen_output = self._generate_rollouts(raw_model, rep_ids, rep_ages, rep_mask, rep_values, rep_val_masks)
        raw_model.train()

        rollout_token_strs: List[List[str]] = gen_output["sequences"]           # B*K items
        rollout_seq_vals: Optional[torch.Tensor] = gen_output.get("sequence_vals")     # (B*K, full_len) or None
        rollout_seq_val_masks: Optional[torch.Tensor] = gen_output.get("sequence_val_masks")  # (B*K, full_len) or None

        # Free the generation KV-cache and any intermediate tensors before the
        # PG/KL forward pass, which materialises large (N, full_len, vocab) logits.
        del gen_output, rep_ids, rep_ages, rep_mask, rep_values, rep_val_masks
        torch.cuda.empty_cache()

        # ---------------------------------------------------------------
        # 2. Compute rewards
        # ---------------------------------------------------------------
        rewards: List[float] = []
        for i in range(B):
            prefix_len_i = int(prefix_lengths[i].item())
            rollout_conds_i = [
                extract_conditions_from_rollout(
                    rollout_token_strs[i * K + k],
                    prefix_len_i,
                    self.target_concept_ids,
                    self.rl_args.prediction_windows,
                )
                for k in range(K)
            ]
            R_i = compute_patient_reward(
                future_conditions[i],
                rollout_conds_i,
                self.prevalence_stats,
                self.rl_args.prediction_windows,
                self.rl_args.rarity_gamma,
                self.rl_args.alpha_max,
                self.rl_args.window_eta,
                self.rl_args.window_ref_days,
                self.rl_args.prevalence_epsilon,
                self.rl_args.false_positive_lambda,
            )
            rewards.append(R_i)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=prefix_input_ids.device)

        # ---------------------------------------------------------------
        # 3. Baseline-adjusted advantage
        # ---------------------------------------------------------------
        advantages = rewards_t - self._baseline
        m = self.rl_args.baseline_momentum
        self._baseline = m * self._baseline + (1.0 - m) * rewards_t.mean().item()

        # Baseline EMA is maintained per-rank; slight divergence across DDP
        # ranks is acceptable for an approximation baseline.

        # ---------------------------------------------------------------
        # 4 & 5. PG loss + KL loss
        # ---------------------------------------------------------------
        pg_loss, kl_loss = self._compute_pg_and_kl_loss(
            raw_model,
            prefix_input_ids,
            prefix_ages,
            prefix_epoch_times,
            prefix_values,
            prefix_value_indicators,
            prefix_lengths,
            rollout_token_strs,
            rollout_seq_vals,
            rollout_seq_val_masks,
            K,
            advantages,
        )

        total_loss = pg_loss + self.rl_args.kl_beta * kl_loss

        # Accumulate RL metrics; they are averaged and injected into the log
        # dict by the log() override, which the Trainer calls at the right step.
        for name, value in (
            ("rl_pg_loss",     pg_loss.item() if isinstance(pg_loss, torch.Tensor) else float(pg_loss)),
            ("rl_kl_loss",     kl_loss.item() if isinstance(kl_loss, torch.Tensor) else float(kl_loss)),
            ("rl_reward_mean", rewards_t.mean().item()),
            ("rl_baseline",    self._baseline),
        ):
            self._rl_metric_sums[name] = self._rl_metric_sums.get(name, 0.0) + value
            self._rl_metric_counts[name] = self._rl_metric_counts.get(name, 0) + 1

        return total_loss

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------

    def _generate_rollouts(
        self,
        model,
        rep_ids: torch.Tensor,
        rep_ages: torch.Tensor,
        rep_attention_mask: torch.Tensor,
        rep_values: Optional[torch.Tensor] = None,
        rep_val_masks: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Sample rollout trajectories with an explicit attention mask to avoid
        the 'bool has no .view()' error that occurs when pad_token_id == eos_token_id.

        Returns a dict with:
          - "sequences": List[List[str]] — decoded concept token strings (prefix + generated)
          - "sequence_vals": (B*K, full_len) LongTensor of value bin IDs, or None
          - "sequence_val_masks": (B*K, full_len) BoolTensor, or None
        """
        kwargs: Dict[str, Any] = {}
        if rep_values is not None:
            kwargs["values"] = rep_values
        if rep_val_masks is not None:
            kwargs["value_indicators"] = rep_val_masks

        outputs = model.generate(
            input_ids=rep_ids,
            attention_mask=rep_attention_mask,
            ages=rep_ages,
            max_new_tokens=self.rl_args.max_new_tokens,
            do_sample=True,
            top_p=self.rl_args.rollout_top_p,
            temperature=self.rl_args.rollout_temperature,
            pad_token_id=self.cehrgpt_tokenizer.pad_token_id,
            return_dict_in_generate=True,
            cehrgpt_tokenizer=self.cehrgpt_tokenizer,
            **kwargs,
        )
        sequences = [
            self.cehrgpt_tokenizer.convert_ids_to_tokens(seq.cpu().tolist())
            for seq in outputs.sequences
        ]
        return {
            "sequences": sequences,
            # Custom fields present in CehrGptGenerateDecoderOnlyOutput; None for standard output
            "sequence_vals": getattr(outputs, "sequence_vals", None),
            "sequence_val_masks": getattr(outputs, "sequence_val_masks", None),
        }

    # ------------------------------------------------------------------
    # PG + KL loss computation
    # ------------------------------------------------------------------

    def _compute_pg_and_kl_loss(
        self,
        model,
        prefix_ids: torch.Tensor,                       # (B, L)
        prefix_ages: torch.Tensor,                      # (B, L)
        prefix_times: torch.Tensor,                     # (B, L)
        prefix_values: Optional[torch.Tensor],          # (B, L) or None
        prefix_value_indicators: Optional[torch.Tensor],  # (B, L) or None
        prefix_lengths: torch.Tensor,                   # (B,)
        rollout_token_strs: List[List[str]],
        rollout_seq_vals: Optional[torch.Tensor],       # (B*K, full_len) or None
        rollout_seq_val_masks: Optional[torch.Tensor],  # (B*K, full_len) or None
        K: int,
        advantages: torch.Tensor,                       # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched PG + KL computation.

        All valid (i, k) full sequences (prefix + rollout) are left-padded to the
        same length and forwarded through the current model and the reference model
        in two single batched calls.  With left-padding the generated token logits
        are always the last ``rollout_len`` positions:
            logits[n, -(rollout_len+1):-1, :]
        """
        B = prefix_ids.shape[0]
        dev = prefix_ids.device

        # ------------------------------------------------------------------
        # Step 1: Build per-(i, k) full sequences and metadata
        # ------------------------------------------------------------------
        # Each entry: (patient_idx, full_ids_1d, full_ages_1d, full_times_1d,
        #              full_vals_1d_or_None, full_valmask_1d_or_None,
        #              rollout_len, new_ids_t)
        entries: List[Tuple] = []

        for i in range(B):
            prefix_len_i = int(prefix_lengths[i].item())
            prefix_ids_i   = prefix_ids[i, -prefix_len_i:]
            prefix_ages_i  = prefix_ages[i, -prefix_len_i:]
            prefix_times_i = prefix_times[i, -prefix_len_i:]
            prefix_vals_i    = prefix_values[i, -prefix_len_i:]         if prefix_values is not None else None
            prefix_valmask_i = prefix_value_indicators[i, -prefix_len_i:] if prefix_value_indicators is not None else None

            prefix_end_age  = int(prefix_ages_i[-1].item())
            prefix_end_time = float(prefix_times_i[-1].item())

            for k in range(K):
                bk_idx = i * K + k
                rollout_tokens = rollout_token_strs[bk_idx]
                new_tokens = rollout_tokens[prefix_len_i:]

                if not new_tokens:
                    continue

                new_ids = self.cehrgpt_tokenizer.convert_tokens_to_ids(new_tokens)
                if not new_ids:
                    continue

                rollout_len = len(new_ids)
                new_ids_t = torch.tensor(new_ids, dtype=torch.long, device=dev)

                rollout_ages, rollout_times = self._reconstruct_rollout_context(
                    new_tokens, prefix_end_age, prefix_end_time
                )
                rollout_ages_t  = torch.tensor(rollout_ages,  dtype=torch.long,    device=dev)
                rollout_times_t = torch.tensor(rollout_times, dtype=torch.float32,  device=dev)

                if rollout_seq_vals is not None:
                    gen_vals_t    = rollout_seq_vals[bk_idx, prefix_len_i: prefix_len_i + rollout_len].to(dev)
                    gen_valmask_t = rollout_seq_val_masks[bk_idx, prefix_len_i: prefix_len_i + rollout_len].to(dev)
                else:
                    gen_vals_t    = torch.zeros(rollout_len, dtype=torch.long, device=dev)
                    gen_valmask_t = torch.zeros(rollout_len, dtype=torch.bool,  device=dev)

                full_ids_1d   = torch.cat([prefix_ids_i,   new_ids_t])
                full_ages_1d  = torch.cat([prefix_ages_i,  rollout_ages_t])
                full_times_1d = torch.cat([prefix_times_i, rollout_times_t])

                if prefix_vals_i is not None:
                    full_vals_1d    = torch.cat([prefix_vals_i,    gen_vals_t])
                    full_valmask_1d = torch.cat([prefix_valmask_i, gen_valmask_t])
                else:
                    full_vals_1d    = None
                    full_valmask_1d = None

                entries.append((
                    i, full_ids_1d, full_ages_1d, full_times_1d,
                    full_vals_1d, full_valmask_1d, rollout_len, new_ids_t,
                ))

        if not entries:
            zero = prefix_ids.new_zeros(1, dtype=torch.float32).squeeze().requires_grad_(True)
            return zero, zero

        # ------------------------------------------------------------------
        # Step 2: Left-pad all sequences to max_full_len
        # ------------------------------------------------------------------
        N = len(entries)
        max_full_len = max(e[1].shape[0] for e in entries)
        pad_id = self.cehrgpt_tokenizer.pad_token_id
        has_values = entries[0][4] is not None  # consistent across all entries

        batch_ids    = torch.full((N, max_full_len), pad_id, dtype=torch.long,    device=dev)
        batch_ages   = torch.zeros((N, max_full_len),          dtype=torch.long,    device=dev)
        batch_times  = torch.zeros((N, max_full_len),          dtype=torch.float32, device=dev)
        batch_vals   = torch.zeros((N, max_full_len),          dtype=torch.long,    device=dev)
        batch_vmask  = torch.zeros((N, max_full_len),          dtype=torch.bool,    device=dev)
        batch_attn   = torch.zeros((N, max_full_len),          dtype=torch.long,    device=dev)

        for n, (_, fids, fages, ftimes, fvals, fvmask, _, _) in enumerate(entries):
            slen = fids.shape[0]
            batch_ids[n,   -slen:] = fids
            batch_ages[n,  -slen:] = fages
            batch_times[n, -slen:] = ftimes
            if fvals is not None:
                batch_vals[n,  -slen:] = fvals
                batch_vmask[n, -slen:] = fvmask
            batch_attn[n, -slen:] = 1

        # ------------------------------------------------------------------
        # Step 3: Two batched forward passes (current model + ref model)
        # ------------------------------------------------------------------
        fwd_kwargs: Dict[str, Any] = {}
        if has_values:
            fwd_kwargs["values"]           = batch_vals
            fwd_kwargs["value_indicators"] = batch_vmask

        # Current model — gradients flow
        curr_logits = model(
            input_ids=batch_ids,
            ages=batch_ages,
            epoch_times=batch_times,
            attention_mask=batch_attn,
            **fwd_kwargs,
        ).logits  # (N, max_full_len, vocab)

        # Reference model — no gradients
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=batch_ids,
                ages=batch_ages,
                epoch_times=batch_times,
                attention_mask=batch_attn,
                **fwd_kwargs,
            ).logits  # (N, max_full_len, vocab)

        # ------------------------------------------------------------------
        # Step 4: Extract per-token log-probs and aggregate per patient
        # ------------------------------------------------------------------
        # With left-padding, rollout tokens occupy the last `rollout_len`
        # positions; the preceding logit (autoregressive shift) is at
        #   logits[n, -(rollout_len+1):-1, :]
        pg_per_patient: Dict[int, List[torch.Tensor]] = {}
        kl_per_patient: Dict[int, List[torch.Tensor]] = {}

        for n, (i, _, _, _, _, _, rollout_len, new_ids_t) in enumerate(entries):
            gen_logits     = curr_logits[n, -(rollout_len + 1):-1, :]  # (rollout_len, vocab)
            curr_lp        = F.log_softmax(gen_logits, dim=-1)
            token_lp       = curr_lp.gather(1, new_ids_t.unsqueeze(1)).squeeze(1)
            seq_lp         = token_lp.mean()

            ref_gen_logits = ref_logits[n, -(rollout_len + 1):-1, :]
            ref_lp         = F.log_softmax(ref_gen_logits, dim=-1)
            ref_token_lp   = ref_lp.gather(1, new_ids_t.unsqueeze(1)).squeeze(1)
            kl_approx      = (token_lp - ref_token_lp).mean()

            pg_per_patient.setdefault(i, []).append(seq_lp)
            kl_per_patient.setdefault(i, []).append(kl_approx)

        pg_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        for i in sorted(pg_per_patient):
            mean_seq_lp = torch.stack(pg_per_patient[i]).mean()
            pg_terms.append(-advantages[i] * mean_seq_lp)
            kl_terms.append(torch.stack(kl_per_patient[i]).mean())

        if not pg_terms:
            zero = prefix_ids.new_zeros(1, dtype=torch.float32).squeeze().requires_grad_(True)
            return zero, zero

        return torch.stack(pg_terms).mean(), torch.stack(kl_terms).mean()

    # ------------------------------------------------------------------
    # Context reconstruction for generated tokens
    # ------------------------------------------------------------------

    def _reconstruct_rollout_context(
        self,
        new_tokens: List[str],
        prefix_end_age: int,
        prefix_end_time_sec: float,
    ) -> Tuple[List[int], List[float]]:
        """
        Approximate ages and epoch_times for each generated token by tracking
        elapsed days via ATT tokens in the rollout.
        """
        ages: List[int] = []
        times: List[float] = []
        cumulative_days = 0.0

        for token in new_tokens:
            if is_att_token(token):
                try:
                    cumulative_days += extract_time_interval_in_days(token)
                except ValueError:
                    pass
            current_time = prefix_end_time_sec + cumulative_days * _SECONDS_PER_DAY
            current_age = int(prefix_end_age + cumulative_days / 365.25)
            ages.append(current_age)
            times.append(current_time)

        return ages, times
