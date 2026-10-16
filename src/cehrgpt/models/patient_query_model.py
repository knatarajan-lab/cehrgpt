import math
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModel,
    GPT2Config,
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import ModelOutput

from dataclasses import dataclass


@dataclass
class PatientQueryOutput(ModelOutput):
    # (batch,) raw score; sigmoid(logits) is the probability that the answer is yes
    logits: torch.Tensor = None
    loss: Optional[torch.Tensor] = None


class PatientQueryModelConfig(PretrainedConfig):
    """
    Configuration of a model that answers yes/no queries about a patient.

    A query is a concept (which also covers all its descendants) and a time window in days
    after the index date. The patient is given as the linear probing hidden state.

    Args:
        lm_type: "gpt2" or "llama", the architecture of the trunk (trained from scratch)
        hidden_size: width of the trunk
        num_hidden_layers: depth of the trunk
        num_attention_heads: attention heads of the trunk
        intermediate_size: width of the Llama MLP (ignored by GPT-2, which uses 4x hidden)
        patient_feature_dim: size of the patient vector, e.g. the linear probing hidden size
        num_prefix_tokens: how many soft tokens the patient vector is expanded into
        query_concept_ids: the concepts that can be queried, in embedding order
        concept_embedding_dim: width of the concept embeddings (the trunk width if None); set
            it to the width of pretrained vectors, e.g. 768 for the CEHR-GPT token embeddings
    """

    model_type = "patient_query_model"

    def __init__(
        self,
        lm_type: str = "gpt2",
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: Optional[int] = None,
        patient_feature_dim: int = 768,
        num_prefix_tokens: int = 1,
        query_concept_ids: Optional[List[str]] = None,
        concept_embedding_dim: Optional[int] = None,
        **kwargs,
    ):
        if lm_type not in ("gpt2", "llama"):
            raise ValueError(f"lm_type must be 'gpt2' or 'llama', got {lm_type}")
        if num_prefix_tokens < 1:
            raise ValueError("num_prefix_tokens must be at least 1")
        self.lm_type = lm_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.patient_feature_dim = patient_feature_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.query_concept_ids = list(query_concept_ids or [])
        self.concept_embedding_dim = concept_embedding_dim or hidden_size
        super().__init__(**kwargs)

    @property
    def sequence_length(self) -> int:
        # patient prefix, concept token, window token
        return self.num_prefix_tokens + 2

    def build_trunk_config(self) -> PretrainedConfig:
        common = dict(vocab_size=1)  # the trunk only sees embeddings, never token ids
        if self.lm_type == "gpt2":
            return GPT2Config(
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                n_positions=self.sequence_length,
                n_ctx=self.sequence_length,
                **common,
            )
        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.sequence_length,
            **common,
        )


def window_features(start_days: torch.Tensor, end_days: torch.Tensor) -> torch.Tensor:
    """(batch, 3): log scaled start, end and width, so 30 vs 90 days is as visible as 365 vs 730."""
    start = start_days.float()
    end = end_days.float()
    return torch.stack(
        [torch.log1p(start), torch.log1p(end), torch.log1p((end - start).clamp(min=0))],
        dim=-1,
    ) / math.log1p(3650.0)


class PatientQueryModel(PreTrainedModel):
    """
    A GPT-2 or Llama trunk over three kinds of input embeddings:

        [patient prefix tokens] [query concept] [query window]

    The last position is read out by a linear head into one score: the log odds that the
    patient has the concept, or any descendant of it, inside the window. There is no text,
    so there is no vocabulary to generate from and the answer is a probability.
    """

    config_class = PatientQueryModelConfig
    base_model_prefix = "trunk"

    def __init__(self, config: PatientQueryModelConfig):
        super().__init__(config)
        hidden = config.hidden_size
        self.trunk = AutoModel.from_config(config.build_trunk_config())
        # LayerNorm first: the patient states are not normalised to the scale of the other inputs
        self.patient_projector = nn.Sequential(
            nn.LayerNorm(config.patient_feature_dim),
            nn.Linear(config.patient_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden * config.num_prefix_tokens),
        )
        self.concept_embedding = nn.Embedding(
            max(len(config.query_concept_ids), 1), config.concept_embedding_dim
        )
        self.concept_projector = (
            nn.Identity()
            if config.concept_embedding_dim == hidden
            else nn.Linear(config.concept_embedding_dim, hidden)
        )
        self.window_projector = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.concept_index = {c: i for i, c in enumerate(config.query_concept_ids)}
        self.post_init()

    def concept_indices(self, concept_ids: Sequence[str]) -> torch.Tensor:
        """Embedding rows of the given concepts; raises for a concept that cannot be queried."""
        missing = [c for c in concept_ids if c not in self.concept_index]
        if missing:
            raise KeyError(f"Concepts outside the query vocabulary: {missing[:5]}")
        return torch.tensor([self.concept_index[c] for c in concept_ids], dtype=torch.long)

    @torch.no_grad()
    def init_concept_embeddings(self, matrix: np.ndarray, freeze: bool = False) -> None:
        """
        Start the concept embeddings from given vectors, e.g. the CEHR-GPT token embeddings,
        so related concepts begin close together and unseen concepts are not random noise.

        Args:
            matrix: (num_query_concepts, concept_embedding_dim) in the order of
                config.query_concept_ids
            freeze: keep the embeddings fixed during training
        """
        vectors = torch.as_tensor(matrix, dtype=torch.float32)
        expected = (
            len(self.config.query_concept_ids),
            self.config.concept_embedding_dim,
        )
        if tuple(vectors.shape) != expected:
            raise ValueError(
                f"Expected embeddings of shape {expected} (one row per query concept, "
                f"concept_embedding_dim wide), got {tuple(vectors.shape)}"
            )
        self.concept_embedding.weight.copy_(vectors)
        self.concept_embedding.weight.requires_grad_(not freeze)

    def forward(
        self,
        patient_features: torch.Tensor,
        concept_index: torch.Tensor,
        start_days: torch.Tensor,
        end_days: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> PatientQueryOutput:
        """
        The only inputs are the patient state and the query; the answer is the training label.

        Args:
            patient_features: (batch, patient_feature_dim)
            concept_index: (batch,) rows of the query concepts, see concept_indices
            start_days, end_days: (batch,) the window (start, end] after the index date
            labels: (batch,) 1 for yes, 0 for no
        """
        batch_size = patient_features.shape[0]
        expected = (batch_size, self.config.patient_feature_dim)
        if tuple(patient_features.shape) != expected:
            raise ValueError(
                f"patient_features must have shape {expected} (one linear probing hidden state "
                f"per query), got {tuple(patient_features.shape)}. If it is twice as wide, "
                "the features were extracted with the global and local states combined; "
                "re-run the extraction with --disable_combine_global_local."
            )
        prefix = self.patient_projector(patient_features.to(self.dtype)).view(
            batch_size, self.config.num_prefix_tokens, self.config.hidden_size
        )
        concept = self.concept_projector(self.concept_embedding(concept_index)).unsqueeze(1)
        window = self.window_projector(
            window_features(start_days, end_days).to(self.dtype)
        ).unsqueeze(1)
        hidden = self.trunk(
            inputs_embeds=torch.cat([prefix, concept, window], dim=1), return_dict=True
        ).last_hidden_state[:, -1]
        logits = self.head(hidden).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float()
            )
        return PatientQueryOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def predict_proba(
        self,
        patient_features: torch.Tensor,
        concept_index: torch.Tensor,
        start_days: torch.Tensor,
        end_days: torch.Tensor,
    ) -> torch.Tensor:
        """Probability that the answer is yes, shape (batch,)."""
        return torch.sigmoid(
            self(patient_features, concept_index, start_days, end_days).logits.float()
        )
