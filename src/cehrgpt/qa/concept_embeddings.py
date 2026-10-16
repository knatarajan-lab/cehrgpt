"""Starting vectors for the query concepts, taken from the CEHR-GPT token embeddings."""

from typing import Dict, Sequence, Tuple

import numpy as np

from cehrgpt.qa.concept_hierarchy import ConceptHierarchy


def token_vectors_from_weights(
    vocab: Dict[str, int], weights: np.ndarray, tokens: Sequence[str]
) -> Dict[str, np.ndarray]:
    """The embedding rows of the tokens that exist in the vocabulary."""
    return {t: weights[vocab[t]] for t in tokens if t in vocab}


def load_cehrgpt_token_vectors(
    model_path: str, tokens: Sequence[str]
) -> Dict[str, np.ndarray]:
    """
    The input embeddings of a pretrained CEHR-GPT, by concept token.

    These live in the same space as the patient states the model produces, so related
    concepts start close together.
    """
    from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
    from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer

    tokenizer = CehrGptTokenizer.from_pretrained(model_path)
    model = CEHRGPT2LMHeadModel.from_pretrained(model_path)
    weights = model.cehrgpt.wte.weight.detach().float().cpu().numpy()
    return token_vectors_from_weights(tokenizer._tokenizer.get_vocab(), weights, tokens)


def build_concept_embedding_matrix(
    concept_ids: Sequence[str],
    hierarchy: ConceptHierarchy,
    token_vectors: Dict[str, np.ndarray],
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    One starting vector per query concept.

    A concept that is a token uses its own vector. A broader concept that is not a token
    (say "Diabetes mellitus") uses the mean of the vectors of the tokens it covers, so a
    query about it begins at the centre of its descendants. A concept with neither gets small
    random noise.

    Returns:
        the (len(concept_ids), d) matrix in the order of concept_ids, and the share of
        concepts that got a real vector
    """
    if not token_vectors:
        raise ValueError("token_vectors is empty")
    dimension = len(next(iter(token_vectors.values())))
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for token, vector in token_vectors.items():
        for ancestor in hierarchy.ancestors(token):
            if ancestor in sums:
                sums[ancestor] += vector
            else:
                sums[ancestor] = np.array(vector, dtype=np.float64)
            counts[ancestor] = counts.get(ancestor, 0) + 1
    rng = np.random.RandomState(seed)
    matrix = np.zeros((len(concept_ids), dimension), dtype=np.float32)
    found = 0
    for row, concept_id in enumerate(concept_ids):
        if concept_id in token_vectors:
            matrix[row] = token_vectors[concept_id]
            found += 1
        elif concept_id in sums:
            matrix[row] = sums[concept_id] / counts[concept_id]
            found += 1
        else:
            matrix[row] = 0.02 * rng.randn(dimension)
    return matrix, found / len(concept_ids) if len(concept_ids) else 0.0
