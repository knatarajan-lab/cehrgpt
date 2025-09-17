from typing import Optional, Tuple

import torch.utils.checkpoint
from torch import nn

from cehrgpt.models.gpt2 import GPT2MLP, GPT2AttentionRoPE, GPT2FlashAttention


class LinearProbBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        # inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        attention_class = (
            GPT2FlashAttention
            if getattr(config, "_attn_implementation", "eager") == "flash_attention_2"
            else GPT2AttentionRoPE
        )
        self.crossattention = attention_class(
            config=config, is_cross_attention=True, layer_idx=layer_idx
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(hidden_size, config)

    def forward(
        self,
        linear_prob_hidden_states: Optional[Tuple[torch.FloatTensor]],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = linear_prob_hidden_states
        linear_prob_hidden_states = self.ln_1(linear_prob_hidden_states)
        # We disable the causal attention between linear probing tokens
        cross_attn_outputs = self.crossattention(
            linear_prob_hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        linear_prob_hidden_states = residual + attn_output

        linear_prob_hidden_states = self.ln_2(linear_prob_hidden_states)

        feed_forward_hidden_states = self.mlp(linear_prob_hidden_states)
        # residual connection
        linear_prob_hidden_states = residual + feed_forward_hidden_states
        return linear_prob_hidden_states
