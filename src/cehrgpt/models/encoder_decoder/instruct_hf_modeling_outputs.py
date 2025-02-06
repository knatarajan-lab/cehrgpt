from typing import Optional, Tuple

import torch

from cehrgpt.models.hf_modeling_outputs import CehrGptCausalLMOutput


class InstructCehrGptCausalLMOutput(CehrGptCausalLMOutput):
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
