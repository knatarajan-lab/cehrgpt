"""
The encoder of a question at inference, held to the one that made the question embeddings of the model.

The head was trained on the frozen embeddings of the question names. A new question only means something to it
if it is embedded the same way: same encoder (and revision), same template, pooling and normalisation. A look
alike (another encoder with the same width, another template) gives vectors the head never saw and answers
without any sign that something is wrong, so it is refused:

  * the description of the embedder is stored in the model config (open_vocab_embedder) when the head is
    first trained, with a few probes: the text that was embedded and its vector;
  * QuestionEncoder is built from that description (the encoder is loaded at the revision that is recorded)
    and, whoever gave it the encoder, checks the description of the encoder against the record and
    that it reproduces every probe vector. Anything else raises EmbedderMismatchError;
  * what it embeds is a QuestionVectors, which carries the fingerprint of the embedder. The predictor of the
    head (OpenVocabPredictor) accepts nothing else, and compares the fingerprint with the model's own.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

LOG = logging.getLogger(__name__)

# What makes an embedding the same one. The version of transformers is left out: it is kept in the record for
# information, and a change that matters shows in the probes.
IDENTITY_FIELDS = ("model_name", "revision", "pooling", "normalize", "max_length", "template", "dim")
PROBE_SIMILARITY = 0.999


class EmbedderMismatchError(RuntimeError):
    """The encoder is not the one that made the question embeddings of the model."""


def identity(description: Dict) -> Dict:
    return {field: description.get(field) for field in IDENTITY_FIELDS}


def fingerprint(description: Dict) -> str:
    """A short hash of the identity of the embedder: two embedders with the same fingerprint are the same."""
    text = json.dumps(identity(description), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def differences(expected: Dict, actual: Dict) -> List[str]:
    """The identity fields on which two descriptions disagree. A revision that is not known on one side is not one."""
    found = []
    for field in IDENTITY_FIELDS:
        a, b = expected.get(field), actual.get(field)
        if field == "revision" and (a is None or b is None):
            continue
        if a != b:
            found.append(f"{field}: the model was trained with {a!r}, this is {b!r}")
    return found


def require_same_embedder(expected: Optional[Dict], actual: Dict, what: str = "the question embeddings"):
    """Raise EmbedderMismatchError unless the two descriptions are of the same embedder."""
    if expected is None:
        raise EmbedderMismatchError(
            "The model has no record of the encoder of its question embeddings (config.open_vocab_embedder), "
            f"so {what} cannot be checked. Train the head with embeddings made by question_embeddings."
        )
    found = differences(expected, actual)
    if found:
        raise EmbedderMismatchError(f"{what} do not come from the embedder of the model: " + "; ".join(found))


@dataclass
class QuestionVectors:
    """Embedded questions, with the embedder that made them. Built by QuestionEncoder.embed."""

    vectors: torch.Tensor  # (questions, dim)
    fingerprint: str
    texts: Sequence[str] = ()

    def __len__(self) -> int:
        return self.vectors.shape[0]


class QuestionEncoder:
    """
    Embeds the text of a question the way the model's question embeddings were made, or refuses to exist.

    Args:
        description: config.open_vocab_embedder of the model (QuestionEmbeddings.embedder)
        encoder: an object with describe() and encode(texts); default: the HuggingFaceEncoder named in the
            description, at its recorded revision. An encoder given here is verified like the default one.
        probe_similarity: the lowest cosine similarity between a probe and its stored vector
    """

    def __init__(
        self,
        description: Optional[Dict],
        encoder=None,
        probe_similarity: float = PROBE_SIMILARITY,
        device: Optional[str] = None,
        fp16: bool = False,
    ):
        if not description:
            raise EmbedderMismatchError(
                "The model has no record of the encoder of its question embeddings (config.open_vocab_embedder); "
                "a new question cannot be embedded in a way the head is known to understand."
            )
        if not description.get("probes"):
            raise EmbedderMismatchError("The record of the embedder has no probes to verify an encoder with")
        self.description = description
        self.template = description["template"]
        self.fingerprint = fingerprint(description)
        if encoder is None:
            from cehrgpt.analysis.concept_name_embeddings import HuggingFaceEncoder

            encoder = HuggingFaceEncoder(
                description["model_name"],
                max_length=int(description["max_length"]),
                revision=description.get("revision"),
                device=device,
                fp16=fp16,
            )
        self.encoder = encoder
        self._verify(probe_similarity)

    @classmethod
    def from_model(cls, model, **kwargs) -> "QuestionEncoder":
        """The encoder of a loaded CEHRGPT2LMHeadModel with the open vocabulary head."""
        return cls(getattr(model.config, "open_vocab_embedder", None), **kwargs)

    def _verify(self, probe_similarity: float):
        actual = {**self.encoder.describe(), "template": self.template, "dim": self.description.get("dim")}
        found = differences(self.description, actual)
        if found:
            raise EmbedderMismatchError("The encoder is not the one of the question embeddings: " + "; ".join(found))
        probes = self.description["probes"]
        encoded = np.asarray(self.encoder.encode([p["text"] for p in probes]), dtype=np.float64)
        stored = np.asarray([p["vector"] for p in probes], dtype=np.float64)
        if encoded.shape != stored.shape:
            raise EmbedderMismatchError(
                f"The encoder gives {encoded.shape[-1]} dimensional vectors, the question embeddings have {stored.shape[-1]}"
            )
        cosine = (encoded * stored).sum(-1) / (np.linalg.norm(encoded, axis=-1) * np.linalg.norm(stored, axis=-1))
        worst = float(cosine.min())
        if worst < probe_similarity:
            raise EmbedderMismatchError(
                f"The encoder does not reproduce the question embeddings: a probe ({probes[int(cosine.argmin())]['text']!r}) "
                f"has cosine similarity {worst:.4f} with its stored vector, below {probe_similarity}. It is another "
                "model, revision, precision or preprocessing than the one the head was trained with."
            )
        LOG.info("The question encoder reproduces the %s probes (lowest cosine similarity %.5f)", len(probes), worst)

    def embed(self, texts: Sequence[str]) -> QuestionVectors:
        """The vectors of questions given as text, with the template of the embeddings applied."""
        texts = list(texts)
        if not texts:
            raise ValueError("No questions")
        rendered = [self.template.format(text=text) for text in texts]
        vectors = torch.as_tensor(np.asarray(self.encoder.encode(rendered), dtype=np.float32))
        return QuestionVectors(vectors, self.fingerprint, texts)
