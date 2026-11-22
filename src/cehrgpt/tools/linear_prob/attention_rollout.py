"""
Attention rollout for the patient representations of CEHR-GPT.

Attention rollout (Abnar & Zuidema, 2020, "Quantifying Attention Flow in Transformers") follows
the attention weights down the layers to say how much of a state comes from every input token.
A layer with a residual connection is written as ``alpha * I + (1 - alpha) * A`` (``A`` the
attention weights with the heads fused, ``alpha`` = 0.5 in the paper) and the layers are
multiplied.

The global representation of CEHR-GPT is not a state of the decoder. It is built by the linear
probing module (``LinearProbModule``), and the rollout follows that structure:

* The decoder has ``L`` blocks of causal self attention. ``R_l = A^_l ... A^_1`` (``R_0 = I``)
  says how much of the output of the decoder layer ``l`` at a position comes from every input
  token. The attention is causal and never crosses a patient of a packed sequence, so every
  ``R_l`` is lower triangular inside a patient and zero outside.
* The probe has ``K`` blocks (``linear_prob_n_layer``, ``L`` by default). The block ``i`` cross
  attends to the output of the decoder layer ``m_i = L - K + 1 + i``: the probe reads one
  decoder layer per block, not the last layer only. The probe state of every position starts
  as the same learned token and the positions of the probe do not attend to each other, so the
  cross attention is the only way a patient gets into the global representation.
* With ``c_i`` the cross attention of the block ``i`` at the prediction time (a row over the
  positions of the decoder) the probe rolls out as ``q_{i+1} = alpha * q_i + (1 - alpha) * c_i
  R_{m_i}`` from ``q_0 = 0``: the learned token carries nothing of the patient. What is left
  on that token (``alpha^K``) is the same for every patient, so it is dropped and the scores
  are normalised to sum to 1 over the input tokens of the patient.

Only the rows at the prediction time are needed, so nothing is multiplied as matrices: a row
vector is taken down the decoder, picking up the cross attention of the probe block that reads
each layer on the way, which costs ``O(L T^2)`` for a sequence of ``T`` tokens.

The rollout follows the attention only. It does not see the value vectors, the MLPs or the
layer normalisations, and the fused heads and the fixed ``alpha`` are approximations: read the
scores as where the model looked, not as the causal effect of a token on a prediction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

HEAD_FUSIONS = ("mean", "max", "min")
PROBE_LAYER_WEIGHTINGS = ("rollout", "uniform")
TARGETS = ("global", "local")


@dataclass
class TokenImportance:
    """
    The importance of the input tokens for the representation of N (patient, prediction time).

    batch_indices: (N,) the row of the batch each one is in.
    query_positions: (N,) the position of the prediction time in that row.
    scores: (N, T) the importance of every position of the row, non-negative, summing to 1,
        zero for padding and for the other patients of a packed row.
    token_mask: (N, T) the positions that belong to the patient (the same packed segment).
    """

    batch_indices: torch.Tensor
    query_positions: torch.Tensor
    scores: torch.Tensor
    token_mask: torch.Tensor


def fuse_heads(attention: torch.Tensor, head_fusion: str = "mean") -> torch.Tensor:
    """(batch, heads, queries, keys) attention weights as (batch, queries, keys)."""
    if attention.dim() == 3:
        return attention
    if attention.dim() != 4:
        raise ValueError(
            f"Attention weights are (batch, heads, queries, keys) or (batch, queries, keys), "
            f"received {tuple(attention.shape)}"
        )
    if head_fusion == "mean":
        return attention.mean(dim=1)
    if head_fusion == "max":
        return attention.amax(dim=1)
    if head_fusion == "min":
        return attention.amin(dim=1)
    raise ValueError(f"head_fusion is one of {HEAD_FUSIONS}, received {head_fusion}")


def get_segment_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    The patient every position belongs to, -1 for padding.

    Sample packing puts several patients in one row with a padding token between them, so a
    patient is a run of 1s in the attention mask (the same rule as
    ``create_sample_packing_attention_mask``). It holds for a row with one patient too, padded
    on either side.
    """
    real = attention_mask.to(torch.bool)
    segment_ids = (~real).cumsum(dim=-1)
    return torch.where(real, segment_ids, torch.full_like(segment_ids, -1))


def find_prediction_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    The (row, position) of the last token of every patient, in the order of the batch.

    This is the position the features of the linear probing are taken from
    (``extract_features_from_packed_sequence``, or the last position without sample packing).
    """
    real = attention_mask.to(torch.bool)
    next_is_real = torch.zeros_like(real)
    next_is_real[:, :-1] = real[:, 1:]
    return torch.nonzero(real & ~next_is_real, as_tuple=False)


def _allowed_attention(
    attention_mask: torch.Tensor, causal: bool
) -> torch.Tensor:
    """(batch, T, T) True where a query may attend to a key: same patient, and the past if causal."""
    segment_ids = get_segment_ids(attention_mask)
    real = segment_ids >= 0
    allowed = (
        (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1))
        & real.unsqueeze(2)
        & real.unsqueeze(1)
    )
    if causal:
        length = attention_mask.shape[-1]
        allowed = allowed & torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=allowed.device)
        )
    return allowed


def _restrict_and_normalize(
    attention: torch.Tensor,
    allowed: torch.Tensor,
    name: str,
    leak_tolerance: Optional[float],
) -> torch.Tensor:
    """
    Zero the weights a query cannot have and make every row sum to 1 again.

    A softmax under a mask leaves exactly nothing on the masked keys, so this only removes
    rounding (half precision) and the rows of the padding queries, which attend to nothing and
    come out of the softmax uniform. With `leak_tolerance` the weights are checked first: a
    real query with more than that on keys it cannot see means the weights were not computed
    under the mask this rollout assumes, which is an error and not something to normalise away.
    """
    attention = attention.to(torch.float32)
    kept = attention * allowed
    if leak_tolerance is not None:
        has_keys = allowed.any(dim=-1)
        leaked = ((attention.sum(dim=-1) - kept.sum(dim=-1)) * has_keys).abs().max()
        if leaked.item() > leak_tolerance:
            raise ValueError(
                f"{name}: {leaked.item():.4f} of the attention of a token is on the future, on "
                f"padding or on another patient. The weights do not match the attention mask."
            )
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def decoder_attention_rollout(
    self_attentions: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    residual_weight: float = 0.5,
    head_fusion: str = "mean",
    leak_tolerance: Optional[float] = 1e-3,
) -> List[torch.Tensor]:
    """
    The rollout matrices ``R_1 .. R_L`` of the decoder, each (batch, T, T).

    ``R_l[b, t, j]`` is how much of the output of the layer ``l`` at the position ``t`` comes
    from the input token ``j``. The rows of real tokens sum to 1, the rows of padding are zero.

    This multiplies (T, T) matrices and is meant for short sequences and for checking
    ``linear_prob_attention_rollout``, which gets the same numbers for the rows it needs
    without the products.
    """
    _check_residual_weight(residual_weight)
    allowed = _allowed_attention(attention_mask, causal=True)
    real = attention_mask.to(torch.bool)
    identity = torch.diag_embed(real.to(torch.float32))
    rollout = identity
    rollouts = []
    for layer, attention in enumerate(self_attentions):
        attention = _restrict_and_normalize(
            fuse_heads(attention, head_fusion),
            allowed,
            f"self attention of the decoder layer {layer + 1}",
            leak_tolerance,
        )
        mixed = residual_weight * identity + (1.0 - residual_weight) * attention
        rollout = torch.bmm(mixed, rollout)
        rollouts.append(rollout)
    return rollouts


def _check_residual_weight(residual_weight: float) -> None:
    if not 0.0 <= residual_weight < 1.0:
        raise ValueError(
            f"residual_weight is the share of the residual connection, in [0, 1), "
            f"received {residual_weight}"
        )


def probe_layer_weights(
    num_probe_layers: int, residual_weight: float = 0.5, weighting: str = "rollout"
) -> torch.Tensor:
    """
    The share of the global representation each probe block brings, summing to 1.

    "rollout": the block ``i`` of ``K`` brings ``(1 - alpha) * alpha^(K - 1 - i)``, which is
    what ``q_{i+1} = alpha * q_i + (1 - alpha) * c_i R`` gives once the share of the learned
    probe token (``alpha^K``) is dropped. The last blocks count the most.
    "uniform": every block counts the same. The probe state is the learned token plus the sum
    of what the blocks add, so this is the reading where no block is discounted.
    """
    _check_residual_weight(residual_weight)
    if weighting == "uniform":
        weights = torch.ones(num_probe_layers)
    elif weighting == "rollout":
        exponents = torch.arange(num_probe_layers - 1, -1, -1, dtype=torch.float32)
        weights = (1.0 - residual_weight) * residual_weight**exponents
    else:
        raise ValueError(
            f"probe_layer_weighting is one of {PROBE_LAYER_WEIGHTINGS}, received {weighting}"
        )
    return weights / weights.sum()


def linear_prob_attention_rollout(
    self_attentions: Sequence[torch.Tensor],
    cross_attentions: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    query_positions: Optional[torch.Tensor] = None,
    residual_weight: float = 0.5,
    probe_layer_weighting: str = "rollout",
    head_fusion: str = "mean",
    leak_tolerance: Optional[float] = 1e-3,
) -> TokenImportance:
    """
    The importance of every input token for a patient representation at the prediction time.

    Args:
        self_attentions: the self attention weights of the ``L`` decoder layers, first layer
            first, each (batch, heads, T, T) or with the heads fused already (batch, T, T).
        cross_attentions: the cross attention weights of the ``K <= L`` probe blocks, first
            block first, same shapes (queries are the probe positions, keys the decoder
            positions). The block ``i`` reads the decoder layer ``L - K + 1 + i``, as
            ``LinearProbModule`` does. Empty for the local representation, the state of the
            last decoder layer at the prediction time: the result is then the row of the
            decoder rollout.
        attention_mask: (batch, T), 1 for a token and 0 for padding. Several patients packed
            in a row are told apart by the padding between them, and each is rolled out on
            its own tokens only.
        query_positions: (N, 2) the (row, position) of the prediction times. The last token
            of every patient by default, which is where the features of the linear probing
            are taken. At an earlier position the cross attention is used as it is given: the
            eager attention of the probe is not causal (the flash one is), so there it also
            covers the tokens of the patient after the position.
        residual_weight: ``alpha``, the share of the residual connection in a layer.
        probe_layer_weighting: how the probe blocks are weighted, see ``probe_layer_weights``.
        head_fusion: how the heads are fused, "mean" (the paper), "max" or "min".
        leak_tolerance: fail when a token has more than this much attention where the mask
            does not let it attend. None to skip the check.
    """
    _check_residual_weight(residual_weight)
    num_layers, num_probe_layers = len(self_attentions), len(cross_attentions)
    if num_layers == 0:
        raise ValueError("The self attention of at least one decoder layer is needed")
    if num_probe_layers > num_layers:
        raise ValueError(
            f"The probe has {num_probe_layers} blocks and the decoder {num_layers} layers: "
            f"every probe block reads its own decoder layer"
        )
    if query_positions is None:
        query_positions = find_prediction_positions(attention_mask)
    batch_indices, positions = query_positions[:, 0], query_positions[:, 1]
    segment_ids = get_segment_ids(attention_mask)
    if (segment_ids[batch_indices, positions] < 0).any():
        raise ValueError("A prediction time is on a padding token")

    num_queries, length = query_positions.shape[0], attention_mask.shape[-1]
    device = attention_mask.device
    allowed_self = _allowed_attention(attention_mask, causal=True)
    # The probe is not causal on the eager path, the mask only keeps it inside the patient
    allowed_cross = _allowed_attention(attention_mask, causal=False)[batch_indices, positions]

    scores = torch.zeros((num_queries, length), dtype=torch.float32, device=device)
    if num_probe_layers == 0:
        scores[torch.arange(num_queries, device=device), positions] = 1.0
    else:
        weights = probe_layer_weights(num_probe_layers, residual_weight, probe_layer_weighting)
    first_read_layer = num_layers - num_probe_layers

    rows = [
        torch.nonzero(batch_indices == b, as_tuple=False).flatten()
        for b in range(attention_mask.shape[0])
    ]
    # From the last decoder layer down: scores holds, for every prediction time, how much of
    # the representation comes from each position of the output of the layer being visited.
    for layer in range(num_layers - 1, -1, -1):
        probe_layer = layer - first_read_layer
        if probe_layer >= 0:
            cross_attention = fuse_heads(cross_attentions[probe_layer], head_fusion)
            cross_rows = _restrict_and_normalize(
                cross_attention[batch_indices, positions],
                allowed_cross,
                f"cross attention of the probe block {probe_layer}",
                leak_tolerance,
            )
            scores = scores + weights[probe_layer].item() * cross_rows
        attention = _restrict_and_normalize(
            fuse_heads(self_attentions[layer], head_fusion),
            allowed_self,
            f"self attention of the decoder layer {layer + 1}",
            leak_tolerance,
        )
        attended = torch.zeros_like(scores)
        for b, row in enumerate(rows):
            if row.numel() > 0:
                attended[row] = scores[row] @ attention[b]
        scores = residual_weight * scores + (1.0 - residual_weight) * attended

    token_mask = segment_ids[batch_indices] == segment_ids[batch_indices, positions].unsqueeze(1)
    scores = scores * token_mask
    scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return TokenImportance(
        batch_indices=batch_indices,
        query_positions=positions,
        scores=scores,
        token_mask=token_mask,
    )


def uniform_attention_baseline(
    attention_mask: torch.Tensor,
    num_layers: int,
    num_probe_layers: int,
    query_positions: Optional[torch.Tensor] = None,
    residual_weight: float = 0.5,
    probe_layer_weighting: str = "rollout",
) -> TokenImportance:
    """
    The rollout of a model that attends uniformly, for the same patients and layers.

    Causal attention alone puts more on the first tokens of a patient: every later position
    can attend to them, so they are reached by more paths. This is the importance a token gets
    from its position and nothing else. Divide the scores of the model by it (or subtract it)
    before reading a token at the start of the history as important.
    """
    uniform = torch.ones(
        (attention_mask.shape[0], attention_mask.shape[-1], attention_mask.shape[-1]),
        dtype=torch.float32,
        device=attention_mask.device,
    )
    return linear_prob_attention_rollout(
        self_attentions=[uniform] * num_layers,
        cross_attentions=[uniform] * num_probe_layers,
        attention_mask=attention_mask,
        query_positions=query_positions,
        residual_weight=residual_weight,
        probe_layer_weighting=probe_layer_weighting,
        leak_tolerance=None,
    )


class AttentionCapture:
    """
    Collects the attention weights of a CEHR-GPT forward pass, with the heads fused.

    Inside the ``with`` block every self attention of the decoder and every cross attention of
    the linear probing module is made to return its weights. They are fused over the heads
    as soon as they exist and kept in ``self_attentions`` (one (batch, T, T) per decoder
    layer) and ``cross_attentions`` (one per probe block), and taken out of what the attention
    returns unless the caller asked for them, so the (batch, heads, T, T) weights of a layer
    are freed before the next layer runs.

    The weights only exist on the eager path: flash attention, SDPA and xformers never compute
    them. The model has to be loaded with ``attn_implementation="eager"``, which is the same
    function of the same parameters. Visit-local attention (``use_local_attention``) has no
    eager path and is not supported, and neither is ``causal_sfm``, which inserts a position
    in the decoder that the probe does not have.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean"):
        if head_fusion not in HEAD_FUSIONS:
            raise ValueError(f"head_fusion is one of {HEAD_FUSIONS}, received {head_fusion}")
        config = model.config
        attn_implementation = getattr(config, "_attn_implementation", "eager")
        if attn_implementation != "eager" or getattr(config, "use_local_attention", False):
            raise ValueError(
                "The attention weights are only computed by the eager attention, this model "
                f"uses attn_implementation={attn_implementation!r} with use_local_attention="
                f"{getattr(config, 'use_local_attention', False)}. Load it with "
                "attn_implementation='eager' (visit-local attention is not supported)."
            )
        if getattr(config, "causal_sfm", False):
            raise NotImplementedError("Attention rollout does not support causal_sfm")
        self.model = model
        self.head_fusion = head_fusion
        self.self_attentions: List[Optional[torch.Tensor]] = []
        self.cross_attentions: List[Optional[torch.Tensor]] = []
        self._handles = []
        self._asked_by_caller: Dict[int, bool] = {}

    def _modules(self):
        decoder = [block.attn for block in self.model.cehrgpt.h]
        linear_prob = getattr(self.model, "linear_prob", None)
        probe = (
            [block.crossattention for block in linear_prob.linear_prob]
            if linear_prob is not None
            else []
        )
        return decoder, probe

    def _ask_for_weights(self, module, args, kwargs):
        self._asked_by_caller[id(module)] = bool(kwargs.get("output_attentions", False))
        kwargs["output_attentions"] = True
        return args, kwargs

    def _collect(self, store: List[Optional[torch.Tensor]], index: int):
        def hook(module, args, kwargs, output):
            weights = output[-1]
            if weights is None or not torch.is_tensor(weights) or weights.dim() != 4:
                raise RuntimeError(
                    f"{type(module).__name__} did not return attention weights. Load the "
                    "model with attn_implementation='eager'."
                )
            store[index] = fuse_heads(weights.detach(), self.head_fusion)
            if self._asked_by_caller.pop(id(module), False):
                return output
            return output[:-1]

        return hook

    def __enter__(self) -> "AttentionCapture":
        decoder, probe = self._modules()
        self.self_attentions = [None] * len(decoder)
        self.cross_attentions = [None] * len(probe)
        for store, modules in (
            (self.self_attentions, decoder),
            (self.cross_attentions, probe),
        ):
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
        self._handles = []
        self._asked_by_caller = {}


@torch.no_grad()
def compute_token_importance(
    model: nn.Module,
    model_inputs: Dict[str, torch.Tensor],
    target: str = "global",
    query_positions: Optional[torch.Tensor] = None,
    residual_weight: float = 0.5,
    probe_layer_weighting: str = "rollout",
    head_fusion: str = "mean",
    leak_tolerance: Optional[float] = 1e-3,
) -> TokenImportance:
    """
    Run a ``CEHRGPT2LMHeadModel`` on a batch and roll its attention out to the input tokens.

    Args:
        model: the model, loaded with ``attn_implementation="eager"`` and in eval mode.
        model_inputs: what the model takes (input_ids, attention_mask, values, ...). The
            batch can be sample packed, but the eager attention of a row of T tokens takes
            (heads, T, T) per layer, so long packed rows do not fit in memory.
        target: "global" for the state of the linear probing module (the first half of the
            saved features), "local" for the state of the last decoder layer (the second half).
            A model without the probe (trained without MOTOR) has the last decoder state as
            its global features, and gets the local rollout for both.
        The other arguments are those of ``linear_prob_attention_rollout``.
    """
    if target not in TARGETS:
        raise ValueError(f"target is one of {TARGETS}, received {target}")
    if model.training:
        raise ValueError("Call model.eval() first: dropout would be applied to the attention")
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("model_inputs must have an attention_mask")
    with AttentionCapture(model, head_fusion) as capture:
        outputs = model(**model_inputs, output_hidden_states=True)
    cross_attentions: Sequence[torch.Tensor] = ()
    if target == "global" and outputs.linear_prob_hidden_states is not None:
        cross_attentions = capture.cross_attentions
        if any(attention is None for attention in cross_attentions):
            raise RuntimeError("A block of the linear probing module did not run")
    return linear_prob_attention_rollout(
        self_attentions=capture.self_attentions,
        cross_attentions=cross_attentions,
        attention_mask=attention_mask,
        query_positions=query_positions,
        residual_weight=residual_weight,
        probe_layer_weighting=probe_layer_weighting,
        head_fusion=head_fusion,
        leak_tolerance=leak_tolerance,
    )
