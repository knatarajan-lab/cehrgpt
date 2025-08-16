from typing import Optional, Tuple, Union

import torch.nn.functional as f
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import Conv1D, GPT2Attention
from transformers.utils import is_flash_attn_2_available, logging

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from cehrgpt.models.activations import RMSNorm

logger = logging.get_logger("transformers")


def is_sample_pack(attention_mask: torch.Tensor) -> bool:
    """
    Determines whether any sequence in the batch is likely sample-packed.

    A sample-packed sequence is one where there are non-padding (1) tokens
    after a padding (0) token, indicating multiple sequences packed together
    with padding as a separator.

    Args:
        attention_mask (torch.Tensor): A tensor of shape (batch_size, seq_len)
            where 1 indicates a real token and 0 indicates padding.

    Returns:
        bool: True if any sample in the batch is sample-packed, False otherwise.
    """

    # If the attention_maks is left padded, we will flip it so we can use the same logic below
    if (attention_mask[:, 0] == 0).any():
        attention_mask = attention_mask.flip(dims=[1])

    nonzero_counts = attention_mask.sum(dim=1)
    max_token_positions = torch.argmax(
        attention_mask.to(torch.int32).flip(dims=[1]), dim=1
    )
    max_indices = attention_mask.shape[1] - 1 - max_token_positions
    return torch.any(nonzero_counts < (max_indices + 1)).item()


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    # This infers sample packing
    if is_sample_pack(attention_mask):
        # Assume input: attention_mask shape = (batch, seq_len)
        attention_mask = attention_mask.flatten()  # shape: (seq_len,)

        # Compute max_index of the last non-zero element
        nonzero = torch.nonzero(attention_mask, as_tuple=False).flatten()
        max_index = nonzero[-1].item()

        # Pad the truncated attention mask
        padded_attention_mask = f.pad(attention_mask[: max_index + 1], (0, 1), value=0)

        # Indices of all tokens
        indices = torch.nonzero(attention_mask, as_tuple=False).flatten()

        # Find where 0s occur (segment boundaries)
        cumsum_seqlens_in_batch = torch.cumsum(padded_attention_mask, dim=0)[
            padded_attention_mask == 0
        ]

        # Compute seqlens per segment
        seqlens_in_batch = (
            cumsum_seqlens_in_batch
            - f.pad(cumsum_seqlens_in_batch, (1, 0), value=0)[:-1]
        ).to(torch.int)

        max_seqlen_in_batch = (
            seqlens_in_batch.max().item() if seqlens_in_batch.numel() > 0 else 0
        )
        cu_seqlens = f.pad(cumsum_seqlens_in_batch, (1, 0)).to(torch.int)
    else:
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = f.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )

    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.linspace(0, 2, steps=dim // 2))).reshape(
            1, 1, dim // 2
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 2:
            time = time[..., None]
        t = self.inv_freq.to(time.device) * time
        sin, cos = torch.sin(t), torch.cos(t)
        sin = torch.stack((sin, sin), dim=-1).reshape(x.shape)
        cos = torch.stack((cos, cos), dim=-1).reshape(x.shape)
        flat_x = x.reshape(-1, x.shape[-1])
        x1 = flat_x[:, ::2]
        x2 = flat_x[:, 1::2]
        return (x * cos) + (torch.stack((-x2, x1), dim=-1).reshape(x.shape) * sin)


class GPT2AttentionRoPE(GPT2Attention):
    """
    GPT2FlashAttention inherits from `GPT2Attention`.

    The primary change is in the forward pass, where it correctly
    calls the public API of flash attention and handles padding tokens.
    """

    def __init__(
        self, config, is_cross_attention=False, layer_idx=None, apply_rotary=False
    ):
        super().__init__(config, is_cross_attention, layer_idx)
        self.apply_rotary = apply_rotary
        if self.apply_rotary:
            self.rope = RotaryPositionEmbedding(config.hidden_size)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        position_ids: Optional[Tuple[torch.FloatTensor]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        if self.apply_rotary and position_ids is not None:
            query = self.rope(query, position_ids)
            key = self.rope(key, position_ids)
            # value = self.rope(value, position_ids)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2FlashAttention(GPT2Attention):
    """
    GPT2FlashAttention inherits from `GPT2Attention`.

    The primary change is in the forward pass, where it correctly
    calls the public API of flash attention and handles padding tokens.
    """

    def __init__(
        self, config, is_cross_attention=False, layer_idx=None, apply_rotary=False
    ):
        super().__init__(config, is_cross_attention, layer_idx)
        self.apply_rotary = apply_rotary
        if self.apply_rotary:
            self.rope = RotaryPositionEmbedding(config.hidden_size)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        position_ids: Optional[Tuple[torch.FloatTensor]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        # Prepare query, key, and value
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        if self.apply_rotary and position_ids is not None:
            query = self.rope(query, position_ids)
            key = self.rope(key, position_ids)
            # value = self.rope(value, position_ids)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # Apply Flash Attention Forward
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            # Flash Attention forward pass
            attn_output = self._flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query.size(-2),
                self.attn_dropout.p,
                softmax_scale=None,
            )

        # Merge heads and project back to hidden size
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
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
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token.

        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        dtype = query_states.dtype
        query_states = query_states.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        key_states = key_states.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        value_states = value_states.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)

        # Contains at least one padding token in the sequence
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
            # (batch, seq_length, n_heads, head_dim)
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
        # re-order the tensor back to (batch, n_heads, seq_length, head_dim)
        return attn_output.permute(0, 2, 1, 3).contiguous().to(dtype)

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
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
            )  # There is a memcpy here, that is very bad.
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


class LlamaMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.activation_function]

    def forward(self, x: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        attention_class = (
            GPT2FlashAttention
            if getattr(config, "_attn_implementation", "eager") == "flash_attention_2"
            else GPT2AttentionRoPE
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = attention_class(
            config=config, layer_idx=layer_idx, apply_rotary=config.apply_rotary
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = attention_class(
                config=config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        decoder_mlp_function = getattr(config, "decoder_mlp", "GPT2MLP")
        if decoder_mlp_function == "GPT2MLP":
            self.mlp = GPT2MLP(inner_dim, config)
        elif getattr(config, "decoder_mlp", "GPT2Block") == "LlamaMLP":
            self.mlp = LlamaMLP(inner_dim, config)
        else:
            raise RuntimeError("You must set decoder_mlp to one of (GPT2MLP, LlamaMLP)")

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        position_ids: Optional[Tuple[torch.FloatTensor]] = None,
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
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
