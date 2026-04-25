"""
GRPO (Group Relative Policy Optimization / REINFORCE + KL) trainer for CEHR-GPT.

Algorithm per patient i  (DeepSeek GRPO formulation)
-----------------------------------------------------
1. Sample K rollout trajectories τ_1…τ_K ~ π_θ(· | prefix_i)
2. Compute reward R_i using the weighted condition-recovery objective
3. Compute advantage  A_i = R_i - b  (b = moving-average baseline)
4. Policy-gradient loss:
       L_PG = - (1/B) Σ_i A_i · (1/K) Σ_k (1/T_k) Σ_t log π_θ(τ_k[t]|·)
5. Per-rollout advantage normalised within the patient's K rollouts:
       A_{i,k} = (R_{i,k} − mean_k R_{i,k}) / (std_k R_{i,k} + ε)
6. Exact per-position KL divergence (always ≥ 0):
       KL(π_θ || π_ref)(t) = Σ_v π_θ(v|·) · (log π_θ(v|·) − log π_ref(v|·))
7. Total loss:
       L = L_PG + β · (1/B) Σ_i (1/K) Σ_k mean_t KL(π_θ || π_ref)(t)
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.utils import logging

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token, is_visit_start
from cehrgpt.rl.reward import compute_patient_reward, compute_visit_embedding_reward, extract_conditions_from_rollout
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
        # Accumulators for train RL metrics — flushed and averaged in log()
        self._rl_metric_sums: Dict[str, float] = {}
        self._rl_metric_counts: Dict[str, int] = {}
        # Accumulators for eval metrics — flushed in log() when eval_loss is present
        self._eval_metric_sums: Dict[str, float] = {}
        self._eval_metric_counts: Dict[str, int] = {}
        # _signature_columns is normally set by _remove_unused_columns, which is
        # skipped when remove_unused_columns=False.  Initialise it here so that
        # the empty-batch error message in _prepare_inputs doesn't crash.
        if self._signature_columns is None:
            self._signature_columns = []

        # PyTorch >=2.6 changed the default of torch.load to weights_only=True.
        # Checkpoint RNG state files contain numpy arrays, which are not in the
        # default safe-globals allowlist.  Register _reconstruct so that
        # Trainer._load_rng_state can load them without arbitrary-code-execution risk.
        try:
            import numpy.core.multiarray as _npcma
            torch.serialization.add_safe_globals([_npcma._reconstruct])
        except (AttributeError, ImportError):
            pass

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
        # Flush eval metrics when the Trainer logs eval results.
        if "eval_loss" in logs and self._eval_metric_sums:
            for name, total in self._eval_metric_sums.items():
                count = self._eval_metric_counts.get(name, 1)
                logs[name] = round(total / count, 6)
            self._eval_metric_sums = {}
            self._eval_metric_counts = {}
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
    # Evaluation step — compute reward directly (not PG loss)
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
        future_input_ids: Optional[torch.Tensor] = inputs.get("future_input_ids")   # (B, F)
        future_ages: Optional[torch.Tensor] = inputs.get("future_ages")             # (B, F)
        future_epoch_times: Optional[torch.Tensor] = inputs.get("future_epoch_times")  # (B, F)
        future_lengths: Optional[torch.Tensor] = inputs.get("future_lengths")       # (B,)

        B = prefix_input_ids.shape[0]
        is_training = model.training
        K = self.rl_args.num_rollouts if is_training else self.rl_args.eval_num_rollouts

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
        torch.cuda.empty_cache()
        with torch.no_grad():
            gen_output = self._generate_rollouts(raw_model, rep_ids, rep_ages, rep_mask, rep_values, rep_val_masks)
        if is_training:
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
        # outputs.sequences includes the full left-padded prefix, so generated tokens
        # start at index max_prefix_len (the padded width), not prefix_lengths[i].
        max_prefix_len = prefix_input_ids.shape[1]

        # ---------------------------------------------------------------
        # Per-rollout rewards: rewards_t[i, k] = R_{i,k}
        #
        # Embedding-based reward: for each patient, compare visit-level hidden-state
        # representations from the ref model between the true future sequence and each
        # generated rollout.  R_{i,k} = Σ_v exp(-||e_orig_v − e_gen_v||_2).
        if (
            future_input_ids is not None
            and future_ages is not None
            and future_epoch_times is not None
            and future_lengths is not None
        ):
            rewards_t = self._compute_embedding_rewards(
                prefix_input_ids, prefix_ages, prefix_epoch_times,
                prefix_lengths,
                future_input_ids, future_ages, future_epoch_times, future_lengths,
                rollout_token_strs, K,
                prefix_input_ids.device,
            )  # (B, K)
        else:
            # Fallback: condition-prediction reward (no future token IDs in batch)
            rollout_rewards: List[List[float]] = []
            for i in range(B):
                patient_rewards: List[float] = []
                for k in range(K):
                    conds_k = extract_conditions_from_rollout(
                        rollout_token_strs[i * K + k],
                        max_prefix_len,
                        self.target_concept_ids,
                        self.rl_args.prediction_windows,
                    )
                    R_k = compute_patient_reward(
                        future_conditions[i],
                        [conds_k],
                        self.prevalence_stats,
                        self.rl_args.prediction_windows,
                        self.rl_args.rarity_gamma,
                        self.rl_args.alpha_max,
                        self.rl_args.window_eta,
                        self.rl_args.window_ref_days,
                        self.rl_args.prevalence_epsilon,
                        self.rl_args.false_positive_lambda,
                    )
                    patient_rewards.append(R_k)
                rollout_rewards.append(patient_rewards)
            rewards_t = torch.tensor(rollout_rewards, dtype=torch.float32, device=prefix_input_ids.device)  # (B, K)


        # ---------------------------------------------------------------
        # 3. Advantage estimation (baseline subtraction or value network)
        # ---------------------------------------------------------------
        advantages, value_loss = self._compute_advantages(
            rewards_t, raw_model,
            prefix_input_ids, prefix_ages, prefix_epoch_times,
            prefix_attention_mask, prefix_values, prefix_value_indicators,
            prefix_lengths,
        )

        # ---------------------------------------------------------------
        # 4. PG loss + KL regularisation (GRPO formulation)
        # ---------------------------------------------------------------
        pg_loss, kl_loss = self._compute_pg_loss(
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
        if value_loss is not None:
            total_loss = total_loss + self.rl_args.value_loss_coef * value_loss

        # Build the shared metric set logged for both train and eval.
        # Train metrics are flushed by log() at each logging step (no prefix).
        # Eval metrics are flushed by log() when eval_loss is present (eval_ prefix).
        base_metrics = [
            ("rl_pg_loss",     pg_loss.item() if isinstance(pg_loss, torch.Tensor) else float(pg_loss)),
            ("rl_kl_loss",     kl_loss.item() if isinstance(kl_loss, torch.Tensor) else float(kl_loss)),
            ("rl_reward_mean", rewards_t.mean().item()),
            ("rl_reward_std",  rewards_t.std(dim=1).mean().item()),
        ]
        if value_loss is not None:
            base_metrics.append(("rl_value_loss", value_loss.item()))

        if is_training:
            for name, value in base_metrics:
                self._rl_metric_sums[name] = self._rl_metric_sums.get(name, 0.0) + value
                self._rl_metric_counts[name] = self._rl_metric_counts.get(name, 0) + 1
        else:
            for name, value in base_metrics:
                # Prefix with eval_; rename rl_reward_mean → eval_reward for model selection.
                eval_name = "eval_reward" if name == "rl_reward_mean" else f"eval_{name}"
                self._eval_metric_sums[eval_name] = self._eval_metric_sums.get(eval_name, 0.0) + value
                self._eval_metric_counts[eval_name] = self._eval_metric_counts.get(eval_name, 0) + 1

        return total_loss

    # ------------------------------------------------------------------
    # Embedding-based reward computation
    # ------------------------------------------------------------------

    def _get_vs_token_ids(self) -> Set[int]:
        """Return the set of token IDs that represent a visit-start ([VS] / VS)."""
        vs_ids: Set[int] = set()
        unk_id = self.cehrgpt_tokenizer.unk_token_id
        for tok in ["VS", "[VS]"]:
            tid = self.cehrgpt_tokenizer.convert_tokens_to_ids([tok])[0]
            if tid != unk_id:
                vs_ids.add(tid)
        return vs_ids

    def _compute_embedding_rewards(
        self,
        prefix_ids: torch.Tensor,          # (B, L) left-padded
        prefix_ages: torch.Tensor,          # (B, L) left-padded
        prefix_times: torch.Tensor,         # (B, L) left-padded
        prefix_lengths: torch.Tensor,       # (B,)
        future_ids: torch.Tensor,           # (B, F) right-padded
        future_ages: torch.Tensor,          # (B, F) right-padded
        future_times: torch.Tensor,         # (B, F) right-padded
        future_lengths: torch.Tensor,       # (B,)
        rollout_token_strs: List[List[str]],
        K: int,
        dev: torch.device,
    ) -> torch.Tensor:
        """
        Compute per-rollout embedding rewards (B, K).

        For each patient i:
          1. Build the full original sequence (unpadded prefix + future tokens) and
             pass it through the frozen ref model to get last-layer hidden states.
          2. Identify [VS] positions in the *future* portion and extract their
             hidden-state vectors → orig_vs_embeddings[i].
          3. For each rollout k, build (unpadded prefix + generated tokens), run
             through ref model, identify [VS] positions in the *generated* portion,
             extract hidden states → gen_vs_embeddings[i, k].
          4. R_{i,k} = Σ_v exp(-||e_orig_v − e_gen_v||_2) summed over original visits;
             missing generated visits contribute 0, extra ones are ignored.

        Note: if the model uses causal_sfm, ``output.hidden_states[-1]`` has one
        extra sequence position inserted at ``demographics_size`` for the random
        vector; VS position indices would be shifted by 1 after that point.  This
        is not corrected here and is unlikely to affect models that do not use
        causal_sfm.
        """
        B = prefix_ids.shape[0]
        max_prefix_len = prefix_ids.shape[1]
        vs_token_ids = self._get_vs_token_ids()
        pad_id = self.cehrgpt_tokenizer.pad_token_id
        chunk_size = self.rl_args.generation_chunk_size or (B + B * K)

        # ------------------------------------------------------------------
        # Step 1: extract VS embeddings from the original (ground-truth) sequences
        # ------------------------------------------------------------------
        # Build one entry per patient: (i, full_ids, full_ages, full_times, prefix_len_i)
        orig_entries: List[Tuple] = []
        for i in range(B):
            plen = int(prefix_lengths[i].item())
            flen = int(future_lengths[i].item())
            p_ids   = prefix_ids[i, -plen:].to(dev)
            p_ages  = prefix_ages[i, -plen:].to(dev)
            p_times = prefix_times[i, -plen:].to(dev)
            f_ids   = future_ids[i, :flen].to(dev)
            f_ages  = future_ages[i, :flen].to(dev)
            f_times = future_times[i, :flen].to(dev)
            full_ids   = torch.cat([p_ids,   f_ids])
            full_ages  = torch.cat([p_ages,  f_ages])
            full_times = torch.cat([p_times, f_times])
            orig_entries.append((i, full_ids, full_ages, full_times, plen))

        # orig_vs_per_patient[i] = list of (hidden_dim,) tensors on CPU
        orig_vs_per_patient: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for chunk_start in range(0, len(orig_entries), chunk_size):
            chunk = orig_entries[chunk_start: chunk_start + chunk_size]
            max_len = max(e[1].shape[0] for e in chunk)
            N = len(chunk)
            b_ids   = torch.full((N, max_len), pad_id, dtype=torch.long,    device=dev)
            b_ages  = torch.zeros((N, max_len),         dtype=torch.long,    device=dev)
            b_times = torch.zeros((N, max_len),         dtype=torch.float32, device=dev)
            b_attn  = torch.zeros((N, max_len),         dtype=torch.long,    device=dev)
            for n, (_, fids, fages, ftimes, _) in enumerate(chunk):
                slen = fids.shape[0]
                b_ids[n,   -slen:] = fids
                b_ages[n,  -slen:] = fages
                b_times[n, -slen:] = ftimes
                b_attn[n,  -slen:] = 1

            with torch.no_grad():
                out = self.ref_model(
                    input_ids=b_ids, ages=b_ages, epoch_times=b_times,
                    attention_mask=b_attn, output_hidden_states=True,
                )
            last_hs = out.hidden_states[-1]  # (N, max_len, hidden_dim)

            for n, (i, fids, _, _, plen) in enumerate(chunk):
                seq_len = fids.shape[0]
                hs_n = last_hs[n, -seq_len:]       # (seq_len, hidden_dim)
                # VS tokens in the future portion (positions >= plen)
                fut_ids_list = fids[plen:].tolist()
                for j, tid in enumerate(fut_ids_list):
                    if tid in vs_token_ids:
                        orig_vs_per_patient[i].append(hs_n[plen + j].detach().cpu())

            del out, last_hs, b_ids, b_ages, b_times, b_attn
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Step 2: extract VS embeddings from each rollout sequence
        # ------------------------------------------------------------------
        # Build one entry per valid (i, k) rollout
        rollout_entries: List[Tuple] = []  # (i, k, full_ids, full_ages, full_times, plen)
        for i in range(B):
            plen = int(prefix_lengths[i].item())
            p_ids   = prefix_ids[i, -plen:].to(dev)
            p_ages  = prefix_ages[i, -plen:].to(dev)
            p_times = prefix_times[i, -plen:].to(dev)
            prefix_end_age  = int(p_ages[-1].item())
            prefix_end_time = float(p_times[-1].item())

            for k in range(K):
                bk_idx     = i * K + k
                new_tokens = rollout_token_strs[bk_idx][max_prefix_len:]
                if not new_tokens:
                    rollout_entries.append((i, k, None, None, None, plen))
                    continue
                new_ids = self.cehrgpt_tokenizer.convert_tokens_to_ids(new_tokens)
                if not new_ids:
                    rollout_entries.append((i, k, None, None, None, plen))
                    continue
                new_ids_t  = torch.tensor(new_ids, dtype=torch.long,    device=dev)
                r_ages, r_times = self._reconstruct_rollout_context(
                    new_tokens, prefix_end_age, prefix_end_time
                )
                r_ages_t   = torch.tensor(r_ages,  dtype=torch.long,    device=dev)
                r_times_t  = torch.tensor(r_times, dtype=torch.float32, device=dev)
                full_ids   = torch.cat([p_ids,   new_ids_t])
                full_ages  = torch.cat([p_ages,  r_ages_t])
                full_times = torch.cat([p_times, r_times_t])
                rollout_entries.append((i, k, full_ids, full_ages, full_times, plen))

        # gen_vs_per_rollout[(i, k)] = list of (hidden_dim,) tensors on CPU
        gen_vs_per_rollout: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        valid_entries = [(idx, e) for idx, e in enumerate(rollout_entries) if e[2] is not None]

        for chunk_start in range(0, len(valid_entries), chunk_size):
            chunk = valid_entries[chunk_start: chunk_start + chunk_size]
            max_len = max(e[1][2].shape[0] for e in chunk)
            N = len(chunk)
            b_ids   = torch.full((N, max_len), pad_id, dtype=torch.long,    device=dev)
            b_ages  = torch.zeros((N, max_len),         dtype=torch.long,    device=dev)
            b_times = torch.zeros((N, max_len),         dtype=torch.float32, device=dev)
            b_attn  = torch.zeros((N, max_len),         dtype=torch.long,    device=dev)
            for n, (_, (_, _, fids, fages, ftimes, _)) in enumerate(chunk):
                slen = fids.shape[0]
                b_ids[n,   -slen:] = fids
                b_ages[n,  -slen:] = fages
                b_times[n, -slen:] = ftimes
                b_attn[n,  -slen:] = 1

            with torch.no_grad():
                out = self.ref_model(
                    input_ids=b_ids, ages=b_ages, epoch_times=b_times,
                    attention_mask=b_attn, output_hidden_states=True,
                )
            last_hs = out.hidden_states[-1]  # (N, max_len, hidden_dim)

            for n, (_, (i, k, fids, _, _, plen)) in enumerate(chunk):
                seq_len = fids.shape[0]
                hs_n = last_hs[n, -seq_len:]       # (seq_len, hidden_dim)
                # VS tokens in the generated portion (positions >= plen)
                bk_idx     = i * K + k
                gen_tokens = rollout_token_strs[bk_idx][max_prefix_len:]
                embs: List[torch.Tensor] = []
                for j, tok in enumerate(gen_tokens):
                    if is_visit_start(tok):
                        embs.append(hs_n[plen + j].detach().cpu())
                gen_vs_per_rollout[(i, k)] = embs

            del out, last_hs, b_ids, b_ages, b_times, b_attn
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Step 3: compute scalar reward per (patient, rollout)
        # ------------------------------------------------------------------
        rewards = torch.zeros(B, K, dtype=torch.float32, device=dev)
        for i in range(B):
            orig_embs = orig_vs_per_patient[i]
            for k in range(K):
                gen_embs = gen_vs_per_rollout.get((i, k), [])
                rewards[i, k] = compute_visit_embedding_reward(orig_embs, gen_embs)
        return rewards

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
        Sample rollout trajectories with an explicit attention mask.

        When ``rl_args.generation_chunk_size`` is set the B*K sequences are
        generated in sub-batches of that size, with ``torch.cuda.empty_cache()``
        called between chunks.  This avoids KV-cache OOM when eval_num_rollouts
        is large (default 50), where generating all B*K sequences at once can
        exhaust GPU memory.

        Returns a dict with:
          - "sequences": List[List[str]] — decoded concept token strings (prefix + generated)
          - "sequence_vals": (B*K, full_len) LongTensor of value bin IDs, or None
          - "sequence_val_masks": (B*K, full_len) BoolTensor, or None
        """
        chunk = self.rl_args.generation_chunk_size
        N = rep_ids.shape[0]

        if chunk is None or N <= chunk:
            return self._generate_rollouts_chunk(model, rep_ids, rep_ages, rep_attention_mask, rep_values, rep_val_masks)

        # Process in sub-batches to bound KV-cache memory
        all_seqs: List[List[str]] = []
        all_vals: List[torch.Tensor] = []
        all_vmasks: List[torch.Tensor] = []

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            result = self._generate_rollouts_chunk(
                model,
                rep_ids[start:end],
                rep_ages[start:end],
                rep_attention_mask[start:end],
                rep_values[start:end] if rep_values is not None else None,
                rep_val_masks[start:end] if rep_val_masks is not None else None,
            )
            all_seqs.extend(result["sequences"])
            sv = result.get("sequence_vals")
            if sv is not None:
                all_vals.append(sv)
                all_vmasks.append(result["sequence_val_masks"])
            torch.cuda.empty_cache()

        merged_vals = merged_vmasks = None
        if all_vals:
            max_len = max(t.shape[1] for t in all_vals)
            merged_vals = torch.cat(
                [F.pad(t, (0, max_len - t.shape[1])) for t in all_vals], dim=0
            )
            merged_vmasks = torch.cat(
                [F.pad(t, (0, max_len - t.shape[1])) for t in all_vmasks], dim=0
            )

        return {"sequences": all_seqs, "sequence_vals": merged_vals, "sequence_val_masks": merged_vmasks}

    def _generate_rollouts_chunk(
        self,
        model,
        rep_ids: torch.Tensor,
        rep_ages: torch.Tensor,
        rep_attention_mask: torch.Tensor,
        rep_values: Optional[torch.Tensor] = None,
        rep_val_masks: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Single-batch rollout generation (no chunking)."""
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
            "sequence_vals": getattr(outputs, "sequence_vals", None),
            "sequence_val_masks": getattr(outputs, "sequence_val_masks", None),
        }

    # ------------------------------------------------------------------
    # Advantage estimation
    # ------------------------------------------------------------------

    def _compute_advantages(
        self,
        rewards_t: torch.Tensor,                          # (B, K)
        raw_model,                                        # unwrapped policy model
        prefix_input_ids: torch.Tensor,                   # (B, L)
        prefix_ages: torch.Tensor,                        # (B, L)
        prefix_epoch_times: torch.Tensor,                 # (B, L)
        prefix_attention_mask: torch.Tensor,              # (B, L)
        prefix_values: Optional[torch.Tensor],            # (B, L) or None
        prefix_value_indicators: Optional[torch.Tensor],  # (B, L) or None
        prefix_lengths: torch.Tensor,                     # (B,)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        GRPO per-rollout advantage: normalise within each patient's K rollouts.

            A_{i,k} = (R_{i,k} − mean_k R_{i,k}) / (std_k R_{i,k} + ε)

        Patients where all K rollouts receive the same reward (σ ≈ 0) get zero
        advantage and contribute nothing to the gradient — correct behaviour when
        the policy is deterministic for that patient.

        Returns (advantages, None) — shape (B, K), no value loss for GRPO.
        Subclasses (e.g. CehrGptPPOTrainer) override this to use a learned
        value network instead.
        """
        mu    = rewards_t.mean(dim=1, keepdim=True)   # (B, 1)
        sigma = rewards_t.std(dim=1, keepdim=True)    # (B, 1)
        advantages = (rewards_t - mu) / (sigma + 1e-8)  # (B, K)
        return advantages, None

    # ------------------------------------------------------------------
    # PG + KL loss computation
    # ------------------------------------------------------------------

    def _compute_pg_loss(
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
        advantages: torch.Tensor,                       # (B, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched PG + KL loss.

        KL is computed exactly at every rollout position by summing over the full
        vocabulary:
            KL(π_θ || π_ref)(t) = Σ_v π_θ(v|·) · (log π_θ(v|·) − log π_ref(v|·))

        This is always ≥ 0 and requires no approximation or clamping.

        Returns (pg_loss, kl_loss) where total_loss = pg_loss + β · kl_loss.
        """
        B = prefix_ids.shape[0]
        dev = prefix_ids.device

        entries = self._build_entries(
            B, K, prefix_ids, prefix_ages, prefix_times,
            prefix_values, prefix_value_indicators, prefix_lengths,
            rollout_token_strs, rollout_seq_vals, rollout_seq_val_masks, dev,
        )
        if not entries:
            zero = prefix_ids.new_zeros(1, dtype=torch.float32).squeeze().requires_grad_(True)
            return zero, zero

        # Re-use generation_chunk_size to bound the (chunk, max_len, vocab)
        # logit tensors per forward pass, the same way it limits generation.
        chunk_size = self.rl_args.generation_chunk_size or len(entries)

        # Precompute per-entry loss weights for correct two-level averaging:
        #   L = (1/B) Σ_i  (1/K_i) Σ_k  term_{i,k}
        # Each entry (i, k) contributes with weight 1 / (n_patients * K_i).
        rollouts_per_patient: Dict[int, int] = {}
        for entry in entries:
            rollouts_per_patient[entry[0]] = rollouts_per_patient.get(entry[0], 0) + 1
        n_patients = len(rollouts_per_patient)

        # During training (grad enabled) we backward immediately after each chunk
        # so that only one chunk's activation graph is live at a time.  This keeps
        # peak GPU memory proportional to chunk_size rather than len(entries).
        # During eval (torch.no_grad) there is no graph, so we just accumulate.
        use_chunked_bwd = torch.is_grad_enabled()

        pg_running = 0.0  # training path — scalar accumulators
        kl_running = 0.0
        pg_per_patient: Dict[int, List[Tuple[int, torch.Tensor]]] = {}  # eval path
        kl_per_patient: Dict[int, List[torch.Tensor]] = {}

        for chunk_start in range(0, len(entries), chunk_size):
            chunk = entries[chunk_start: chunk_start + chunk_size]

            batch_ids, batch_ages, batch_times, batch_vals, batch_vmask, batch_attn, has_values = \
                self._left_pad_entries(chunk, dev)

            fwd_kwargs: Dict[str, Any] = {}
            if has_values:
                fwd_kwargs["values"]           = batch_vals
                fwd_kwargs["value_indicators"] = batch_vmask

            # Current model — dropout disabled for deterministic log-probs;
            # gradients still flow since requires_grad is unaffected by eval mode.
            model.eval()
            curr_logits = model(
                input_ids=batch_ids, ages=batch_ages, epoch_times=batch_times,
                attention_mask=batch_attn, **fwd_kwargs,
            ).logits  # (chunk, max_chunk_len, vocab)
            model.train()

            # Reference model — no gradients
            with torch.no_grad():
                ref_logits = self.ref_model(
                    input_ids=batch_ids, ages=batch_ages, epoch_times=batch_times,
                    attention_mask=batch_attn, **fwd_kwargs,
                ).logits  # (chunk, max_chunk_len, vocab)

            chunk_pg_terms: List[torch.Tensor] = []
            chunk_kl_terms: List[torch.Tensor] = []

            for n, (i, k, _, _, _, _, _, rollout_len, new_ids_t) in enumerate(chunk):
                sl = slice(-(rollout_len + 1), -1)

                curr_log_probs = F.log_softmax(curr_logits[n, sl, :], dim=-1)          # (T, vocab)
                ref_log_probs  = F.log_softmax(ref_logits[n,  sl, :], dim=-1).detach() # (T, vocab)

                # PG loss: log-prob of the sampled tokens
                idx    = new_ids_t.unsqueeze(1)                                          # (T, 1)
                seq_lp = curr_log_probs.gather(1, idx).squeeze(1).mean()

                # Exact per-position KL(π_θ || π_ref) = Σ_v π_θ(v)·(log π_θ(v) − log π_ref(v))
                kl_approx = (curr_log_probs.exp() * (curr_log_probs - ref_log_probs)).sum(dim=-1).mean()

                if use_chunked_bwd:
                    w = 1.0 / (n_patients * rollouts_per_patient[i])
                    chunk_pg_terms.append(w * (-advantages[i, k] * seq_lp))
                    chunk_kl_terms.append(w * kl_approx)
                else:
                    pg_per_patient.setdefault(i, []).append((k, seq_lp))
                    kl_per_patient.setdefault(i, []).append(kl_approx)

            if use_chunked_bwd:
                # Sum weighted contributions from this chunk and backward immediately.
                # After backward the chunk's activation graph is freed, so the next
                # chunk's forward pass does not compete with it for GPU memory.
                chunk_pg = torch.stack(chunk_pg_terms).sum()
                chunk_kl = torch.stack(chunk_kl_terms).sum()
                self.accelerator.backward(chunk_pg + self.rl_args.kl_beta * chunk_kl)
                pg_running += chunk_pg.detach().item()
                kl_running += chunk_kl.detach().item()

            del ref_logits, batch_ids, batch_ages, batch_times, batch_attn, batch_vals, batch_vmask, fwd_kwargs
            torch.cuda.empty_cache()

        if use_chunked_bwd:
            # Gradients are already accumulated. Return leaf tensors so that the
            # Trainer's subsequent accelerator.backward(total_loss) is a harmless
            # no-op (the leaf has no parameter graph).
            return (
                torch.tensor(pg_running, device=dev, requires_grad=True),
                torch.tensor(kl_running, device=dev),
            )

        # Eval path: standard two-level aggregation (torch.no_grad() — no graph).
        pg_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        for i in sorted(pg_per_patient):
            pg_terms.append(
                torch.stack([-advantages[i, k] * seq_lp for k, seq_lp in pg_per_patient[i]]).mean()
            )
            kl_terms.append(torch.stack(kl_per_patient[i]).mean())

        if not pg_terms:
            zero = prefix_ids.new_zeros(1, dtype=torch.float32).squeeze().requires_grad_(True)
            return zero, zero

        return torch.stack(pg_terms).mean(), torch.stack(kl_terms).mean()

    # ------------------------------------------------------------------
    # Shared helpers used by both GRPO and PPO trainers
    # ------------------------------------------------------------------

    def _build_entries(
        self,
        B: int,
        K: int,
        prefix_ids: torch.Tensor,
        prefix_ages: torch.Tensor,
        prefix_times: torch.Tensor,
        prefix_values: Optional[torch.Tensor],
        prefix_value_indicators: Optional[torch.Tensor],
        prefix_lengths: torch.Tensor,
        rollout_token_strs: List[List[str]],
        rollout_seq_vals: Optional[torch.Tensor],
        rollout_seq_val_masks: Optional[torch.Tensor],
        dev: torch.device,
    ) -> List[Tuple]:
        """
        Build per-(patient, rollout) full-sequence tensors.

        Each entry: (patient_idx, rollout_idx, full_ids_1d, full_ages_1d, full_times_1d,
                     full_vals_1d_or_None, full_valmask_1d_or_None,
                     rollout_len, new_ids_t)
        """
        # outputs.sequences from .generate() is [pad × pad_len, actual_prefix, generated]
        # with total length = max_prefix_len + num_new.  We must skip max_prefix_len
        # (the full padded width) to reach the generated tokens, NOT prefix_len_i
        # (the actual non-padded length), which lands inside the padding / prefix region.
        max_prefix_len = prefix_ids.shape[1]

        entries: List[Tuple] = []
        for i in range(B):
            prefix_len_i     = int(prefix_lengths[i].item())
            prefix_ids_i     = prefix_ids[i, -prefix_len_i:]
            prefix_ages_i    = prefix_ages[i, -prefix_len_i:]
            prefix_times_i   = prefix_times[i, -prefix_len_i:]
            prefix_vals_i    = prefix_values[i, -prefix_len_i:]             if prefix_values is not None else None
            prefix_valmask_i = prefix_value_indicators[i, -prefix_len_i:]   if prefix_value_indicators is not None else None

            prefix_end_age  = int(prefix_ages_i[-1].item())
            prefix_end_time = float(prefix_times_i[-1].item())

            for k in range(K):
                bk_idx        = i * K + k
                rollout_tokens = rollout_token_strs[bk_idx]
                new_tokens     = rollout_tokens[max_prefix_len:]
                if not new_tokens:
                    continue

                new_ids = self.cehrgpt_tokenizer.convert_tokens_to_ids(new_tokens)
                if not new_ids:
                    continue

                rollout_len     = len(new_ids)
                new_ids_t       = torch.tensor(new_ids, dtype=torch.long, device=dev)
                rollout_ages, rollout_times = self._reconstruct_rollout_context(
                    new_tokens, prefix_end_age, prefix_end_time
                )
                rollout_ages_t  = torch.tensor(rollout_ages,  dtype=torch.long,    device=dev)
                rollout_times_t = torch.tensor(rollout_times, dtype=torch.float32,  device=dev)

                if rollout_seq_vals is not None:
                    gen_vals_t    = rollout_seq_vals[bk_idx, max_prefix_len: max_prefix_len + rollout_len].to(dev)
                    gen_valmask_t = rollout_seq_val_masks[bk_idx, max_prefix_len: max_prefix_len + rollout_len].to(dev)
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
                    i, k, full_ids_1d, full_ages_1d, full_times_1d,
                    full_vals_1d, full_valmask_1d, rollout_len, new_ids_t,
                ))
        return entries

    def _left_pad_entries(
        self,
        entries: List[Tuple],
        dev: torch.device,
    ) -> Tuple:
        """
        Left-pad all (prefix + rollout) sequences to a common length.

        Returns (batch_ids, batch_ages, batch_times, batch_vals,
                 batch_vmask, batch_attn, has_values).
        """
        N            = len(entries)
        max_full_len = max(e[2].shape[0] for e in entries)
        pad_id       = self.cehrgpt_tokenizer.pad_token_id
        has_values   = entries[0][5] is not None

        batch_ids   = torch.full((N, max_full_len), pad_id, dtype=torch.long,    device=dev)
        batch_ages  = torch.zeros((N, max_full_len),         dtype=torch.long,    device=dev)
        batch_times = torch.zeros((N, max_full_len),         dtype=torch.float32, device=dev)
        batch_vals  = torch.zeros((N, max_full_len),         dtype=torch.long,    device=dev)
        batch_vmask = torch.zeros((N, max_full_len),         dtype=torch.bool,    device=dev)
        batch_attn  = torch.zeros((N, max_full_len),         dtype=torch.long,    device=dev)

        for n, (_, _, fids, fages, ftimes, fvals, fvmask, _, _) in enumerate(entries):
            slen = fids.shape[0]
            batch_ids[n,   -slen:] = fids
            batch_ages[n,  -slen:] = fages
            batch_times[n, -slen:] = ftimes
            if fvals is not None:
                batch_vals[n,  -slen:] = fvals
                batch_vmask[n, -slen:] = fvmask
            batch_attn[n, -slen:] = 1

        return batch_ids, batch_ages, batch_times, batch_vals, batch_vmask, batch_attn, has_values

    def _forward(
        self,
        model,
        batch_ids: torch.Tensor,
        batch_ages: torch.Tensor,
        batch_times: torch.Tensor,
        batch_attn: torch.Tensor,
        fwd_kwargs: Dict[str, Any],
        no_grad: bool = False,
    ) -> torch.Tensor:
        """Single batched forward pass; returns logits (N, seq_len, vocab).

        Always runs in eval mode to disable dropout, ensuring deterministic
        log-probs consistent with rollout generation.  Gradients still flow
        when no_grad=False because requires_grad on parameters is unaffected
        by train/eval mode.
        """
        was_training = model.training
        model.eval()
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        try:
            with ctx:
                return model(
                    input_ids=batch_ids, ages=batch_ages, epoch_times=batch_times,
                    attention_mask=batch_attn, **fwd_kwargs,
                ).logits
        finally:
            if was_training:
                model.train()


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
