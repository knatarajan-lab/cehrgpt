"""
Qwen2-style decoder block for CEHR-GPT.

This mirrors `cehrgpt.models.gpt2.GPT2Block` exactly at the interface level - same
`forward` signature, same tuple return shape, same legacy `(key, value)` cache format -
so `CEHRGPT2Model` can swap one for the other without any change to the embedding stack,
the task heads, the output dataclasses, or the generation loop.

What differs from the GPT-2 block:
  * RMSNorm instead of LayerNorm;
  * `nn.Linear` q/k/v/o projections (Qwen2 convention: bias on q/k/v, none on o) instead
    of fused `Conv1D` `c_attn`;
  * per-head rotary embeddings over *sequential* positions, computed directly from
    `position_ids` rather than gathered from a precomputed table;
  * grouped-query attention support via `config.num_key_value_heads`.

Two behaviours are deliberately kept from the GPT-2 block because `CEHRGPT2Model` depends
on them:

  1. Causality is enforced *inside* this module via the triangular `bias` buffer, exactly
     as HF's `GPT2Attention` does. `CEHRGPT2Model.forward` hands us an additive mask that
     encodes padding and sample-packing segments but NOT causality, so a backbone that
     relied on the mask alone for causality (like upstream `Qwen2Model`) would silently
     become bidirectional. Registering `bias` also keeps
     `CEHRGPT2Model.update_attn_bias` working unchanged.
  2. The flash-attention path uses the sample-packing-aware `_get_unpad_data` from
     `cehrgpt.models.gpt2`, which derives per-segment `cu_seqlens` for packed batches.
     Upstream Qwen2 uses the stock version, which treats each row as a single sequence and
     would let packed patients attend across each other's boundaries.

Dropout is read from `attn_pdrop` / `resid_pdrop` (not Qwen2's `attention_dropout`) so the
existing CEHR-GPT dropout configuration applies to both backbones.
"""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.utils import is_flash_attn_2_available, logging

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from cehrgpt.models.activations import RMSNorm
from cehrgpt.models.gpt2 import GPT2MLP, LlamaMLP, _get_unpad_data

logger = logging.get_logger("transformers")


def build_sdpa_params(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float = 0.0,
    is_causal: bool = False,
):
    """
    Construct `torch.backends.cuda.SDPAParams` across torch versions.

    torch 2.4 takes six positional arguments; newer versions (verified on 2.8.0) append a
    trailing `enable_gqa` flag. `enable_gqa=False` is always right here: key/value are
    expanded with `repeat_kv` before reaching SDPA, so the kernel only ever sees as many
    key/value heads as query heads, whatever `num_key_value_heads` is set to.
    """
    try:
        return torch.backends.cuda.SDPAParams(
            query, key, value, attn_mask, dropout_p, is_causal, False
        )
    except TypeError:
        return torch.backends.cuda.SDPAParams(
            query, key, value, attn_mask, dropout_p, is_causal
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (Llama/Qwen2 convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to the query and key tensors.

    `cos`/`sin` are expected to already be selected for the positions of `q`/`k`, i.e.
    shape `(batch, seq_len, head_dim)`; no table lookup by index happens here, so position
    values are never used as an index into a cache.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for grouped-query attention; a no-op when `n_rep == 1`."""
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class Qwen2RotaryEmbedding(nn.Module):
    """
    Rotary embeddings computed directly from `position_ids`.

    Unlike upstream `Qwen2RotaryEmbedding`, which builds a `cos_cached`/`sin_cached` table
    of length `seq_len` and then indexes it with `position_ids`, this computes the angles
    from the position values themselves. Sequential positions are what CEHR-GPT passes for
    this backbone, but computing rather than gathering means arbitrary position magnitudes
    (e.g. ages) can never raise an out-of-range index.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (batch, seq_len) -> freqs: (batch, seq_len, dim // 2)
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        positions = position_ids.to(device=x.device, dtype=torch.float32)
        freqs = positions[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class Qwen2Attention(nn.Module):
    """Qwen2-style self-attention with CEHR-GPT's causal-mask and sample-packing behaviour."""

    # Class-level so the math-fallback warning is emitted once per process, not per layer.
    _sdpa_backend_checked = False

    def __init__(self, config, is_cross_attention: bool = False, layer_idx=None):
        super().__init__()
        if is_cross_attention:
            raise NotImplementedError(
                "The Qwen2 backbone does not support cross attention; set "
                "config.add_cross_attention=False."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got hidden_size: "
                f"{self.embed_dim} and num_attention_heads: {self.num_heads})."
            )

        max_positions = config.max_position_embeddings
        # Same causal buffer as HF's GPT2Attention. CEHRGPT2Model.update_attn_bias
        # re-registers this under the same name when sample packing needs a wider window.
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

        # Qwen2 convention: bias on the q/k/v projections, none on the output projection.
        self.q_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.embed_dim, bias=False
        )

        # QK-norm, as introduced in Qwen3. Qwen2 has no normalisation between the q/k
        # projections and the attention logits, so nothing bounds the scale of those
        # logits during from-scratch training. Qwen3 normalises each head's query and key
        # vector over `head_dim` before the rotary embedding, which is what this
        # reproduces. Note the learned per-dimension gain does mean the rotation no longer
        # commutes exactly with the normalisation, so relative-offset invariance is
        # approximate rather than exact - that is a property of the Qwen3 design itself,
        # not of this port.
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim, base=getattr(config, "rope_theta", 10000.0)
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _causal_additive_mask(self, query, query_length, key_length):
        """Build an additive (-inf above the diagonal) causal mask from the `bias` buffer."""
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        return torch.zeros(
            causal_mask.shape, dtype=query.dtype, device=query.device
        ).masked_fill(~causal_mask, torch.finfo(query.dtype).min)

    def _warn_if_sdpa_falls_back_to_math(self, query, key, value, attention_mask):
        """
        Warn once if neither fused SDPA kernel accepts these inputs.

        PyTorch silently falls back to SDPBackend.MATH in that case, which materialises a
        `(heads, L, L)` score matrix - the very thing sdpa was chosen to avoid. The usual
        cause is a dtype the fused kernels do not support on the current GPU:
        mem-efficient attention needs compute capability 8.0+ for bfloat16, so on Volta
        (sm_70) a bf16 run falls back even though the mask itself is perfectly acceptable.
        """
        if Qwen2Attention._sdpa_backend_checked or not query.is_cuda:
            return
        Qwen2Attention._sdpa_backend_checked = True
        try:
            params = build_sdpa_params(query, key, value, attention_mask)
            efficient = torch.backends.cuda.can_use_efficient_attention(params, False)
            flash = torch.backends.cuda.can_use_flash_attention(params, False)
        except Exception as error:  # noqa: BLE001 - introspection differs across versions
            # Do not fail silently: without this probe a math fallback is invisible.
            logger.warning(
                "Could not determine the SDPA backend (%s: %s). If training memory grows "
                "with the square of the packed sequence length, PyTorch is falling back "
                "to the math backend.",
                type(error).__name__,
                error,
            )
            return
        if not (efficient or flash):
            capability = torch.cuda.get_device_capability(query.device)
            logger.warning(
                "SDPA has no fused kernel for these inputs (dtype=%s, sm_%d%d) and will "
                "fall back to the math backend, which materialises a (heads, seq, seq) "
                "attention matrix. Memory will scale with the square of the packed "
                "sequence length. Note bfloat16 requires sm_80+ for mem-efficient "
                "attention; float16 is supported further back.",
                query.dtype,
                capability[0],
                capability[1],
            )

    def _sdpa_attn(self, query, key, value, attention_mask=None):
        """
        Scaled-dot-product attention.

        `attention_mask` must already be additive AND already include causality:
        `CEHRGPT2Model.forward` folds the causal mask into the padding/sample-packing mask
        once per forward pass for this path, so the (potentially large) combined mask is
        built once rather than per layer. SDPA cannot combine `is_causal=True` with an
        explicit `attn_mask`, so causality has to live in the mask whenever a mask exists.
        """
        query_length, key_length = query.size(-2), key.size(-2)
        self._warn_if_sdpa_falls_back_to_math(query, key, value, attention_mask)
        if attention_mask is None:
            if query_length == key_length:
                return nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            # `is_causal=True` assumes square attention; with a KV cache the query is
            # shorter than the key, so build the offset mask explicitly instead.
            attention_mask = self._causal_additive_mask(
                query, query_length, key_length
            )
        return nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / (
            self.head_dim**0.5
        )

        # ---- causality, enforced here rather than delegated to attention_mask ----
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

        # ---- padding / sample-packing mask, already additive (-inf) ----
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast the softmax for numerical stability (Qwen2 behaviour).
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "The Qwen2 backbone does not support cross attention."
            )

        batch_size, seq_length, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # QK-norm before the rotary embedding, matching Qwen3. Applied on the last
        # dimension, which is head_dim either side of the transpose, so normalisation is
        # per head. Value is deliberately left unnormalised.
        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        # Positions of the *current* query tokens. CEHRGPT2Model supplies sequential
        # positions already offset by the cache length; fall back to a local arange.
        past_length = layer_past[0].shape[-2] if layer_past is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0)
        if position_ids.shape[-1] != seq_length:
            raise ValueError(
                f"position_ids has length {position_ids.shape[-1]} but the current input "
                f"has length {seq_length}."
            )
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        attn_implementation = getattr(self.config, "_attn_implementation", "eager")
        if attn_implementation == "flash_attention_2":
            attn_output = self._flash_attention_forward(
                query,
                repeat_kv(key, self.num_key_value_groups),
                repeat_kv(value, self.num_key_value_groups),
                attention_mask,
                query.size(-2),
                self.attn_dropout.p if self.training else 0.0,
                softmax_scale=None,
            )
            attn_weights = None
        elif attn_implementation == "sdpa":
            if output_attentions:
                raise ValueError(
                    "output_attentions=True is not supported with "
                    "attn_implementation='sdpa' because SDPA never materialises the "
                    "attention weights. Use attn_implementation='eager' instead."
                )
            if head_mask is not None:
                raise ValueError(
                    "head_mask is not supported with attn_implementation='sdpa'."
                )
            attn_output = self._sdpa_attn(
                query,
                repeat_kv(key, self.num_key_value_groups),
                repeat_kv(value, self.num_key_value_groups),
                attention_mask,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._attn(
                query,
                repeat_kv(key, self.num_key_value_groups),
                repeat_kv(value, self.num_key_value_groups),
                attention_mask,
                head_mask,
            )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.embed_dim)
        )
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Flash-attention forward that understands sample-packed batches.

        `attention_mask` here is the raw 2D `(batch, seq_len)` mask - `CEHRGPT2Model`
        skips the additive conversion for the flash path. Segment boundaries are recovered
        by `cehrgpt.models.gpt2._get_unpad_data`, which is packing-aware.
        """
        dtype = query_states.dtype
        query_states = query_states.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        key_states = key_states.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        value_states = value_states.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )
            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )
        return attn_output.permute(0, 2, 1, 3).contiguous().to(dtype)

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Qwen2Block(nn.Module):
    """Pre-norm Qwen2 decoder layer, interface-compatible with `GPT2Block`."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.ln_1 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(hidden_size, eps=rms_norm_eps)

        if config.add_cross_attention:
            raise NotImplementedError(
                "The Qwen2 backbone does not support cross attention; set "
                "config.add_cross_attention=False."
            )

        decoder_mlp_function = getattr(config, "decoder_mlp", "LlamaMLP")
        if decoder_mlp_function == "LlamaMLP":
            self.mlp = LlamaMLP(inner_dim, config)
        elif decoder_mlp_function == "GPT2MLP":
            self.mlp = GPT2MLP(inner_dim, config)
        else:
            raise RuntimeError("You must set decoder_mlp to one of (GPT2MLP, LlamaMLP)")

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)
