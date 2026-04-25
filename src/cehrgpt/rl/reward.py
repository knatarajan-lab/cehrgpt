"""
Reward computation for CEHR-GPT RL fine-tuning.

Implements the visit-embedding reward:

    R_i = Σ_{v=1}^{N_orig}  exp(-||e_orig_v − e_gen_v||_2)

where e_orig_v / e_gen_v are the ref-model last-layer hidden states at
the v-th [VS] token in the true future sequence and the generated rollout
respectively.  Missing generated visits contribute 0; extra ones are ignored.
"""

import math
from typing import List

import torch
from transformers.utils import logging

LOG = logging.get_logger("transformers")


def compute_visit_embedding_reward(
    orig_vs_embeddings: List[torch.Tensor],
    gen_vs_embeddings: List[torch.Tensor],
) -> float:
    """
    Embedding-based reward for a single rollout.

    For each future visit v in the original sequence we compute
    ``exp(-||e_orig_v - e_gen_v||_2)`` and sum across all original visits:

        R_emb = Σ_{v=1}^{N_orig}  exp(-||e_orig_v - e_gen_v||_2)

    Rules:
      * If the generated sequence has *fewer* future visits than the original,
        the missing visits contribute 0.
      * Extra generated visits beyond N_orig are ignored.
      * If there are no original future visits, returns 0.

    Args:
        orig_vs_embeddings: Hidden-state vectors at each future ``[VS]`` position
            in the real patient sequence, one tensor of shape ``(hidden_dim,)``
            per visit.
        gen_vs_embeddings: Hidden-state vectors at each generated ``[VS]`` position
            in the rollout, one tensor of shape ``(hidden_dim,)`` per visit.

    Returns:
        Scalar reward (higher is better).
    """
    total = 0.0
    for v, e_orig in enumerate(orig_vs_embeddings):
        if v < len(gen_vs_embeddings):
            dist = torch.dist(e_orig.float(), gen_vs_embeddings[v].float()).item()
            total += math.exp(-dist)
        # else: generated sequence ran out of visits → reward contribution is 0
    return total
