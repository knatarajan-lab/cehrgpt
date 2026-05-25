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
            if (
                getattr(config, "_attn_implementation", "eager") == "flash_attention_2"
                or getattr(config, "use_local_attention", False)
            )
            else GPT2AttentionRoPE
        )
        self.crossattention = attention_class(
            config=config, is_cross_attention=True, layer_idx=layer_idx
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(hidden_size, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # We disable the causal attention between linear probing tokens
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
        )
        hidden_states = residual + cross_attn_outputs[0]
        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states
