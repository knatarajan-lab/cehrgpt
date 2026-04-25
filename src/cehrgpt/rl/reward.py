"""
Reward computation for CEHR-GPT RL fine-tuning.

Implements the visit-embedding reward:

    R_i = Σ_{v=1}^{N_orig}  w(t_v) · exp(-||e_orig_v − e_gen_v||_2)

where e_orig_v / e_gen_v are the ref-model last-layer hidden states at
the v-th [VE] token in the true future sequence and the generated rollout
respectively, and

    w(t_v) = 1 + log(1 + t_v / D) / log(D)

up-weights visits that are further in the future (t_v = days from prefix end,
D = time_weight_denom defaults to 30 days).  Dividing by log(D) normalises
the increment so it is expressed in the same log base as D.  Missing generated
visits contribute 0; extra ones are ignored.
"""

import math
from typing import List, Optional

import torch
from transformers.utils import logging

LOG = logging.get_logger("transformers")


def compute_visit_embedding_reward(
    orig_vs_embeddings: List[torch.Tensor],
    gen_vs_embeddings: List[torch.Tensor],
    orig_vs_days: Optional[List[float]] = None,
    time_weight_denom: float = 30.0,
) -> float:
    """
    Embedding-based reward for a single rollout.

    For each future visit v in the original sequence we compute
    ``w(t_v) · exp(-||e_orig_v - e_gen_v||_2)`` and sum across all original visits:

        R_emb = Σ_{v=1}^{N_orig}  w(t_v) · exp(-||e_orig_v - e_gen_v||_2)

    where ``w(t_v) = 1 + log(1 + t_v / time_weight_denom)`` up-weights visits
    that are further in the future.  When ``orig_vs_days`` is None all weights
    are 1 (no time weighting).

    Rules:
      * If the generated sequence has *fewer* future visits than the original,
        the missing visits contribute 0.
      * Extra generated visits beyond N_orig are ignored.
      * If there are no original future visits, returns 0.

    Args:
        orig_vs_embeddings: Hidden-state vectors at each future ``[VE]`` position
            in the real patient sequence, one tensor of shape ``(hidden_dim,)``
            per visit.
        gen_vs_embeddings: Hidden-state vectors at each generated ``[VE]`` position
            in the rollout, one tensor of shape ``(hidden_dim,)`` per visit.
        orig_vs_days: Days from the prefix end for each original visit (same
            length as ``orig_vs_embeddings``).  When provided, each visit
            contribution is multiplied by ``1 + log(1 + days / D) / log(D)``.
        time_weight_denom: Reference time in days for the log weight (default 30).

    Returns:
        Scalar reward (higher is better).
    """
    total = 0.0
    for v, e_orig in enumerate(orig_vs_embeddings):
        if v < len(gen_vs_embeddings):
            dist = torch.dist(e_orig.float(), gen_vs_embeddings[v].float()).item()
            sim = math.exp(-dist)
            if orig_vs_days is not None:
                weight = 1.0 + math.log(1.0 + orig_vs_days[v] / time_weight_denom) / math.log(time_weight_denom)
                sim *= weight
            total += sim
        # else: generated sequence ran out of visits → reward contribution is 0
    return total
