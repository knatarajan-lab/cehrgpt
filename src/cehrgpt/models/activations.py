# From https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
# coding=utf-8

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import transformers.pytorch_utils


class SwiGLU(nn.Module):
    def __init__(self, intermediate_size: int):
        super().__init__()
        self.linear1 = nn.Linear(intermediate_size, intermediate_size, bias=False)
        self.linear2 = nn.Linear(intermediate_size, intermediate_size, bias=False)
        self.linear3 = nn.Linear(intermediate_size, intermediate_size, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.linear3(self.swish(self.linear2(x)) * self.linear1(x))


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """MistralRMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


transformers.pytorch_utils.ALL_LAYERNORM_LAYERS.extend([SwiGLU, RMSNorm])
