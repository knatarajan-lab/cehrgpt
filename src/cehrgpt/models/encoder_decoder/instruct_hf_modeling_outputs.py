from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from cehrgpt.models.hf_modeling_outputs import CehrGptCausalLMOutput


@dataclass
class InstructCehrGptCausalLMOutput(CehrGptCausalLMOutput):
    """
    Subclass of CehrGptCausalLMOutput that includes encoder-specific outputs.

    This subclass adds the final hidden states and attention details from the encoder,
    used in settings where an encoder-decoder architecture is involved.

    Attributes:
        encoder_last_hidden_state (Optional[torch.FloatTensor]): Last hidden state of the encoder.
        encoder_attentions (Optional[Tuple[torch.FloatTensor, ...]]): Attention weights from the encoder.
    """

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
