"""
Attention rollout (Abnar & Zuidema, 2020) for the patient representations of CEHR-GPT.

The scores follow the attention weights only: where the model looked, not the effect of a token.
"""

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

# The share of the residual connection in a layer, as in the paper
RESIDUAL_WEIGHT = 0.5


def _restrict_and_normalize(attention: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Zero the weights a query cannot have and make every row sum to 1 again."""
    kept = attention.to(torch.float32) * allowed
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def linear_prob_attention_rollout(
    self_attentions: Sequence[torch.Tensor],
    cross_attentions: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    (batch, T) importance of every input token for the representation at the last token of a row.

    The attentions are (batch, T, T) with the heads averaged, first layer first, and a row holds
    one patient. Without cross attentions the representation is the last decoder state.
    """
    real = attention_mask.to(torch.bool)
    batch_size, length = real.shape
    rows = torch.arange(batch_size, device=real.device)
    # The last token of the patient, where the features of the linear probing are taken
    positions = length - 1 - real.flip(-1).int().argmax(dim=-1)
    causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=real.device))
    allowed = real.unsqueeze(2) & real.unsqueeze(1) & causal
    num_layers = len(self_attentions)
    # The probe block i of K reads the decoder layer L - K + 1 + i
    first_read_layer = num_layers - len(cross_attentions)

    scores = torch.zeros(real.shape, dtype=torch.float32, device=real.device)
    if not cross_attentions:
        scores[rows, positions] = 1.0
    # From the last decoder layer down: scores is how much of the representation comes from
    # each position of the output of the layer being visited.
    for layer in range(num_layers - 1, -1, -1):
        if layer >= first_read_layer:
            cross_rows = _restrict_and_normalize(
                cross_attentions[layer - first_read_layer][rows, positions], real
            )
            # A probe block is discounted by the residual of every block after it
            scores = scores + RESIDUAL_WEIGHT ** (num_layers - 1 - layer) * cross_rows
        attention = _restrict_and_normalize(self_attentions[layer], allowed)
        attended = torch.bmm(scores.unsqueeze(1), attention).squeeze(1)
        scores = RESIDUAL_WEIGHT * scores + (1.0 - RESIDUAL_WEIGHT) * attended
    return scores / scores.sum(dim=-1, keepdim=True)


def uniform_attention_baseline(
    attention_mask: torch.Tensor, num_layers: int, num_probe_layers: int
) -> torch.Tensor:
    """
    The rollout of a model that attends uniformly: what a token gets from its position alone.

    Causal attention reaches the first tokens by more paths, so compare the scores with this.
    """
    length = attention_mask.shape[-1]
    uniform = torch.ones((attention_mask.shape[0], length, length), device=attention_mask.device)
    return linear_prob_attention_rollout(
        [uniform] * num_layers, [uniform] * num_probe_layers, attention_mask
    )


class AttentionCapture:
    """
    Collects the attention weights of a forward pass, averaged over the heads.

    The average is taken in a hook, so the (batch, heads, T, T) weights of a layer are freed
    before the next layer runs.
    """

    def __init__(self, model: nn.Module):
        config = model.config
        if (
            getattr(config, "_attn_implementation", "eager") != "eager"
            or config.use_local_attention
            or config.causal_sfm
        ):
            raise ValueError(
                "Attention rollout needs attn_implementation='eager' and supports neither "
                "use_local_attention nor causal_sfm"
            )
        decoder = [block.attn for block in model.cehrgpt.h]
        probe = (
            [block.crossattention for block in model.linear_prob.linear_prob]
            if model.linear_prob is not None
            else []
        )
        self.self_attentions: List[Optional[torch.Tensor]] = [None] * len(decoder)
        self.cross_attentions: List[Optional[torch.Tensor]] = [None] * len(probe)
        self._stores = [(self.self_attentions, decoder), (self.cross_attentions, probe)]
        self._handles = []

    @staticmethod
    def _ask_for_weights(module, args, kwargs):
        kwargs["output_attentions"] = True
        return args, kwargs

    @staticmethod
    def _collect(store: List[Optional[torch.Tensor]], index: int):
        def hook(module, args, kwargs, output):
            store[index] = output[-1].detach().mean(dim=1)
            return output[:-1]

        return hook

    def __enter__(self) -> "AttentionCapture":
        for store, modules in self._stores:
            for index, module in enumerate(modules):
                self._handles.append(
                    module.register_forward_pre_hook(self._ask_for_weights, with_kwargs=True)
                )
                self._handles.append(
                    module.register_forward_hook(self._collect(store, index), with_kwargs=True)
                )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()


@torch.no_grad()
def compute_token_importance(
    model: nn.Module, model_inputs: Dict[str, torch.Tensor], target: str = "global"
) -> torch.Tensor:
    """
    Run the model on a batch, one patient per row, and roll its attention out to the tokens.

    target is "global" for the state of the linear probing module and "local" for the last
    decoder state, which is also the global one of a model without the probe.
    """
    with AttentionCapture(model) as capture:
        model(**model_inputs)
    cross_attentions = capture.cross_attentions if target == "global" else []
    return linear_prob_attention_rollout(
        capture.self_attentions, cross_attentions, model_inputs["attention_mask"]
    )
