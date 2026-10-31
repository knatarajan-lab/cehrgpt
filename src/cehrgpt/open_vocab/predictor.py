"""
Answers of the patient timeline QA head at inference: the chance that a question happens in a time window.

Like the zero-shot tool of the MOTOR head, it loads only the weights of the head from a checkpoint, applies them
to the features that compute_cehrgpt_features writes (global and local state, which is its default) and turns the
hazards into probabilities. A question enters as QuestionVectors, which only a QuestionEncoder that was verified
against the embedder of the model can make; raw vectors, or vectors of another embedder, are refused.
"""

import os
from typing import List, Optional, Sequence, Tuple

import torch

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.open_vocab.head import OpenVocabularyHead, window_probabilities
from cehrgpt.open_vocab.question_encoder import (
    EmbedderMismatchError,
    QuestionVectors,
    fingerprint,
)

HEAD_PREFIX = "open_vocab_head."


class OpenVocabPredictor:
    def __init__(
        self,
        head: OpenVocabularyHead,
        bins: Sequence[float],
        embedder: Optional[dict],
        device: Optional[str] = None,
    ):
        if not embedder:
            raise EmbedderMismatchError(
                "The model has no record of the encoder of its question embeddings (config.open_vocab_embedder)"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.head = head.to(self.device).eval()
        self.bins = [float(b) for b in bins]
        self.fingerprint = fingerprint(embedder)
        self.embedder = embedder

    @classmethod
    def from_model(cls, model, device: Optional[str] = None) -> "OpenVocabPredictor":
        """The predictor of a loaded CEHRGPT2LMHeadModel that has the head."""
        if model.open_vocab_head is None:
            raise ValueError("The model has no patient timeline QA head")
        head = OpenVocabularyHead(
            feature_dim=model.open_vocab_head.feature_dim,
            question_dim=model.open_vocab_head.question_dim,
            hidden_dim=model.open_vocab_head.hidden_dim,
            num_time_pieces=model.open_vocab_head.num_time_pieces,
            residual_dim=model.config.open_vocab_residual_dim,
        )
        head.load_state_dict(model.open_vocab_head.state_dict())
        return cls(head.float(), model.config.motor_time_bins, model.config.open_vocab_embedder, device)

    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[str] = None) -> "OpenVocabPredictor":
        # the loader of the weights is shared with the zero-shot tool of the MOTOR head
        from cehrgpt.tools.zero_shot_motor_predictions import load_head_state

        model_path = os.path.expanduser(model_path)
        config = CEHRGPTConfig.from_pretrained(model_path)
        if not getattr(config, "include_open_vocab", False):
            raise ValueError(f"{model_path} was not trained with the patient timeline QA head (include_open_vocab)")
        bins = [float(b) for b in config.motor_time_bins]
        head = OpenVocabularyHead(
            feature_dim=2 * config.n_embd,
            question_dim=config.open_vocab_question_dim,
            hidden_dim=config.n_embd,
            num_time_pieces=config.motor_num_time_pieces,
            residual_dim=config.open_vocab_residual_dim,
        )
        head.load_state_dict(load_head_state(model_path, HEAD_PREFIX, "patient timeline QA head"))
        return cls(head.float(), bins, config.open_vocab_embedder, device)

    def _check(self, questions: QuestionVectors):
        if not isinstance(questions, QuestionVectors):
            raise TypeError(
                "Questions must be embedded by a QuestionEncoder (QuestionEncoder.from_model); got "
                f"{type(questions).__name__}, which nothing ties to the embedder the head was trained with"
            )
        if questions.fingerprint != self.fingerprint:
            raise EmbedderMismatchError(
                f"The questions were embedded by another embedder ({questions.fingerprint}) than the one of the "
                f"model ({self.fingerprint})"
            )
        if questions.vectors.shape[-1] != self.head.question_dim:
            raise EmbedderMismatchError(
                f"{questions.vectors.shape[-1]} dimensional question vectors for a head of {self.head.question_dim}"
            )

    @torch.no_grad()
    def log2_hazards(self, features: torch.Tensor, questions: QuestionVectors) -> torch.Tensor:
        """(rows, pieces, questions): the log2 events per day. Features are the global and local state."""
        self._check(questions)
        if features.shape[-1] != self.head.feature_dim:
            raise ValueError(
                f"The features are {features.shape[-1]}-dimensional but the head reads {self.head.feature_dim} "
                "(the global and the local state: extract with compute_cehrgpt_features without "
                "--disable_combine_global_local)"
            )
        features = features.to(self.device, dtype=torch.float32)
        return self.head(features, questions.vectors.to(self.device, dtype=torch.float32))

    @torch.no_grad()
    def probabilities(
        self,
        features: torch.Tensor,
        questions: QuestionVectors,
        windows: Sequence[Tuple[float, float]],
        gap_days: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(rows, windows, questions): the chance of each question in each window (start, end] after the prediction time."""
        hazards = self.log2_hazards(features, questions)
        if gap_days is not None:
            gap_days = gap_days.to(self.device, dtype=torch.float32)
        return window_probabilities(hazards, self.bins, windows, gap_days)
