from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
)


class PatientLanguageModelConfig(PretrainedConfig):
    """
    Configuration of a language model that is conditioned on a patient representation.

    Args:
        lm_type: "gpt2" or "llama", the architecture of the decoder (trained from scratch)
        vocab_size: size of the text vocabulary
        hidden_size: width of the decoder
        num_hidden_layers: depth of the decoder
        num_attention_heads: attention heads of the decoder
        intermediate_size: width of the Llama MLP (ignored by GPT-2, which uses 4x hidden)
        max_position_embeddings: longest text sequence, excluding the patient prefix
        patient_feature_dim: size of the patient vector, e.g. the linear probing hidden size
        num_prefix_tokens: how many soft tokens the patient vector is expanded into
    """

    model_type = "patient_language_model"

    def __init__(
        self,
        lm_type: str = "gpt2",
        vocab_size: int = 8000,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 256,
        patient_feature_dim: int = 768,
        num_prefix_tokens: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        if lm_type not in ("gpt2", "llama"):
            raise ValueError(f"lm_type must be 'gpt2' or 'llama', got {lm_type}")
        if num_prefix_tokens < 1:
            raise ValueError("num_prefix_tokens must be at least 1")
        self.lm_type = lm_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.patient_feature_dim = patient_feature_dim
        self.num_prefix_tokens = num_prefix_tokens
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def build_lm_config(self) -> PretrainedConfig:
        # The prefix takes positions in front of the text
        positions = self.max_position_embeddings + self.num_prefix_tokens
        common = dict(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )
        if self.lm_type == "gpt2":
            return GPT2Config(
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                n_positions=positions,
                n_ctx=positions,
                **common,
            )
        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=positions,
            **common,
        )


class PatientLanguageModel(PreTrainedModel):
    """
    A decoder-only language model whose context starts with the patient representation.

    The patient vector (for example the last hidden state of the CEHR-GPT linear probing
    module) is projected into `num_prefix_tokens` soft tokens that are placed in front of
    the question. The loss is only computed on the labelled (answer) tokens.
    """

    config_class = PatientLanguageModelConfig
    base_model_prefix = "lm"

    def __init__(self, config: PatientLanguageModelConfig):
        super().__init__(config)
        self.lm = AutoModelForCausalLM.from_config(config.build_lm_config())
        # LayerNorm first: the patient states are not normalised to the scale of token embeddings
        self.patient_projector = nn.Sequential(
            nn.LayerNorm(config.patient_feature_dim),
            nn.Linear(config.patient_feature_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size * config.num_prefix_tokens),
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.lm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lm.set_output_embeddings(new_embeddings)

    def _embed_with_prefix(
        self,
        patient_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        batch_size = input_ids.shape[0]
        expected = (batch_size, self.config.patient_feature_dim)
        if tuple(patient_features.shape) != expected:
            raise ValueError(
                f"patient_features must have shape {expected} (one linear probing hidden state "
                f"per question), got {tuple(patient_features.shape)}. If it is twice as wide, "
                "the features were extracted with the global and local states combined; "
                "re-run the extraction with --disable_combine_global_local."
            )
        prefix = self.patient_projector(patient_features.to(self.dtype)).view(
            batch_size, self.config.num_prefix_tokens, self.config.hidden_size
        )
        inputs_embeds = torch.cat(
            [prefix, self.lm.get_input_embeddings()(input_ids)], dim=1
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = attention_mask.new_ones(batch_size, self.config.num_prefix_tokens)
        return inputs_embeds, torch.cat([prefix_mask, attention_mask], dim=1)

    def forward(
        self,
        patient_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        The only inputs are the question and the linear probing hidden state; the answer
        is a training label and is never given at inference.

        Args:
            patient_features: (batch, patient_feature_dim)
            input_ids: (batch, length) text tokens
            attention_mask: (batch, length), 1 for real tokens
            labels: (batch, length) with -100 on tokens that are not predicted
        """
        inputs_embeds, full_mask = self._embed_with_prefix(
            patient_features, input_ids, attention_mask
        )
        if labels is not None:
            ignore = labels.new_full(
                (labels.shape[0], self.config.num_prefix_tokens), -100
            )
            labels = torch.cat([ignore, labels], dim=1)
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def next_token_logits(
        self,
        patient_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Logits of the token that follows the prompt, shape (batch, vocab).

        Prompts must be left padded (see PatientQACollator.collate_prompts) so that the last
        position is a real token. Positions are derived from the mask, so padding does not
        change the result.
        """
        inputs_embeds, full_mask = self._embed_with_prefix(
            patient_features, input_ids, attention_mask
        )
        position_ids = (full_mask.cumsum(-1) - 1).clamp(min=0)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        return out.logits[:, -1]

    @torch.no_grad()
    def generate(
        self,
        patient_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate the answer; returns only the new tokens (the prompt is not repeated)."""
        inputs_embeds, full_mask = self._embed_with_prefix(
            patient_features, input_ids, attention_mask
        )
        # The decoder has a fixed number of positions; a longer answer would index past them
        room = (
            self.config.max_position_embeddings
            + self.config.num_prefix_tokens
            - full_mask.shape[1]
        )
        if room < 1:
            raise ValueError(
                f"The prompt of {input_ids.shape[1]} tokens leaves no room to generate within "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )
        max_new_tokens = generate_kwargs.pop("max_new_tokens", 64)
        return self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            max_new_tokens=min(max_new_tokens, room),
            **generate_kwargs,
        )
