import pickle
from pathlib import Path
from typing import Optional

import numpy as np

PRETRAINED_EMBEDDING_VECTOR_FILE_NAME = "pretrained_embedding_vectors.npy"
PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME = "pretrained_embedding_concepts.pkl"


class PretrainedEmbeddings:
    """A class to handle pretrained embedding vectors and their associated concepts."""

    def __init__(self, model_folder: Optional[str]):
        if model_folder:
            model_path = Path(model_folder)
            self.vector_file = model_path / PRETRAINED_EMBEDDING_VECTOR_FILE_NAME
            self.concept_file = model_path / PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME
            self.exists = self.vector_file.exists() and self.concept_file.exists()
        else:
            self.exists = False
        self._initialize_embeddings() if self.exists else self._initialize_empty()

    def _initialize_embeddings(self):
        """Load embeddings and associated concepts from files."""
        self.pretrained_embeddings = np.load(self.vector_file)
        with open(self.concept_file, "rb") as f:
            self.pretrained_concepts = pickle.load(f)

        self.concept_ids = [
            concept["concept_id"] for concept in self.pretrained_concepts
        ]
        self.reverse_concept_id_map = {
            concept_id: i for i, concept_id in enumerate(self.concept_ids)
        }
        self.concept_names = [
            concept["concept_name"] for concept in self.pretrained_concepts
        ]
        self.embed_dim = self.pretrained_embeddings.shape[1]

        assert len(self.pretrained_embeddings) == len(
            self.pretrained_concepts
        ), "The number of embeddings does not match the number of concepts."

    def _initialize_empty(self):
        """Initialize empty attributes for when files do not exist."""
        self.pretrained_embeddings = None
        self.pretrained_concepts = None
        self.concept_ids = None
        self.concept_names = None
        self.reverse_concept_id_map = None
        self.embed_dim = 0

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.pretrained_embeddings) if self.exists else 0

    def is_concept_available(self, concept_id: str) -> bool:
        """Check if a given concept ID is available."""
        return self.exists and concept_id in self.concept_ids

    def get_concept_embeddings(self, concept_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the embedding vector for a given concept ID.

        Returns None if the concept ID is not available.
        """
        if self.is_concept_available(concept_id):
            return self.pretrained_embeddings[self.reverse_concept_id_map[concept_id]]
        return None

    def save(self, model_folder: str):
        """Save the embeddings and concepts to the specified folder."""
        if self.exists:
            model_path = Path(model_folder)
            np.save(
                model_path / PRETRAINED_EMBEDDING_VECTOR_FILE_NAME,
                self.pretrained_embeddings,
            )
            with open(model_path / PRETRAINED_EMBEDDING_CONCEPT_FILE_NAME, "wb") as f:
                pickle.dump(self.pretrained_concepts, f)
