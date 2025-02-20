from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl


class ConceptTransitionTokenizer:
    def __init__(self, prob_df: pl.DataFrame):
        """
        Initialize tokenizer with conditional probability dataframe.

        Args:
            prob_df: Polars DataFrame with columns [concept_id_1, concept_id_2, prob]
        """
        # Get unique concepts from both columns
        # Get unique concepts from both columns
        concepts1 = prob_df.select("concept_id_1").unique()
        concepts2 = prob_df.select("concept_id_2").unique()

        # Rename columns to match before concatenation
        concepts1 = concepts1.rename({"concept_id_1": "concept"})
        concepts2 = concepts2.rename({"concept_id_2": "concept"})

        unique_concepts = pl.concat([concepts1, concepts2]).unique()

        # Create token mappings
        self.concept_to_token = {
            concept: idx
            for idx, concept in enumerate(unique_concepts.to_numpy().flatten())
        }
        self.token_to_concept = {v: k for k, v in self.concept_to_token.items()}
        self.vocab_size = len(self.concept_to_token)

        # Create transition matrix
        self.transition_matrix = np.zeros((self.vocab_size, self.vocab_size))

        # Fill transition matrix with probabilities
        for row in prob_df.iter_rows():
            i = self.concept_to_token[row[0]]  # concept_id_1
            j = self.concept_to_token[row[1]]  # concept_id_2
            self.transition_matrix[i, j] = row[2]  # prob

        # Normalize probabilities for each concept_id_1
        row_sums = self.transition_matrix.sum(axis=1)
        # Avoid division by zero for rows that sum to 0
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

    def encode(self, concepts: List[str]) -> List[int]:
        """
        Convert concept IDs to token indices.

        Args:
            concepts: List of concept strings to encode

        Returns:
            List of token indices

        Raises:
            KeyError: If a concept is not in the vocabulary
        """
        try:
            return [self.concept_to_token[c] for c in concepts]
        except KeyError as e:
            raise KeyError(f"Concept {e} not found in vocabulary")

    def decode(self, tokens: List[int]) -> List[str]:
        """
        Convert token indices back to concept IDs.

        Args:
            tokens: List of token indices to decode

        Returns:
            List of concept strings

        Raises:
            KeyError: If a token index is not valid
        """
        try:
            return [self.token_to_concept[t] for t in tokens]
        except KeyError as e:
            raise KeyError(f"Token {e} not found in vocabulary")

    def get_transition_prob(self, concept_id_1: str, concept_id_2: str):
        """
        Get transition probability between two concepts.

        Args:
            concept_id_1: Source concept string
            concept_id_2: Target concept string

        Returns:
            Transition probability from concept_id_1 to concept_id_2

        Raises:
            KeyError: If either concept is not in the vocabulary
        """
        try:
            token1 = self.concept_to_token[concept_id_1]
            token2 = self.concept_to_token[concept_id_2]
            return self.transition_matrix[token1, token2]
        except KeyError as e:
            raise KeyError(f"Concept {e} not found in vocabulary")

    def get_next_concepts(
        self, concept_id: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most likely next concepts and their probabilities.

        Args:
            concept_id: Source concept string
            top_k: Number of top concepts to return

        Returns:
            List of tuples (concept_string, probability) sorted by probability

        Raises:
            KeyError: If the concept is not in the vocabulary
            ValueError: If top_k is less than 1 or greater than vocabulary size
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > self.vocab_size:
            raise ValueError(
                f"top_k cannot be larger than vocabulary size ({self.vocab_size})"
            )

        try:
            token = self.concept_to_token[concept_id]
            probs = self.transition_matrix[token]
            top_indices = np.argsort(probs)[-top_k:][::-1]
            return [(self.token_to_concept[idx], probs[idx]) for idx in top_indices]
        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")

    def sample(
        self,
        concept_id: str,
        top_k: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Randomly sample a next concept based on transition probabilities.

        If top_k is provided, sample only from the top-k most likely concepts.

        Args:
            concept_id: Source concept string
            top_k: Optional, if provided, sample only from top-k most likely concepts
            random_state: Optional random seed for reproducibility

        Returns:
            Tuple of (sampled_concept, probability)

        Raises:
            KeyError: If the concept is not in the vocabulary
            ValueError: If top_k is less than 1 or greater than vocabulary size
        """
        try:
            if random_state is not None:
                np.random.seed(random_state)

            token = self.concept_to_token[concept_id]
            probs = self.transition_matrix[token]

            if top_k is not None:
                if top_k < 1:
                    raise ValueError("top_k must be at least 1")
                if top_k > self.vocab_size:
                    raise ValueError(
                        f"top_k cannot be larger than vocabulary size ({self.vocab_size})"
                    )

                # Get top-k indices and their probabilities
                top_indices = np.argsort(probs)[-top_k:]
                top_probs = probs[top_indices]

                # Renormalize probabilities of top-k concepts
                top_probs = top_probs / top_probs.sum()

                # Sample from top-k
                sampled_idx = np.random.choice(top_indices, p=top_probs)
                sampled_concept = self.token_to_concept[sampled_idx]
                sampled_prob = probs[sampled_idx]  # Return original probability
            else:
                # Sample from all concepts
                sampled_idx = np.random.choice(self.vocab_size, p=probs)
                sampled_concept = self.token_to_concept[sampled_idx]
                sampled_prob = probs[sampled_idx]

            return sampled_concept, sampled_prob

        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")
        except ValueError as e:
            if "probabilities do not sum to 1" in str(e):
                # Handle numerical precision issues
                probs = probs / probs.sum()
                return self.sample(concept_id, top_k, random_state)
            raise e


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser("Data generator")
    parser.add_argument("--probability_table", dest="probability_table", required=True)
    parser.add_argument("--output_folder", dest="output_folder", required=True)
    args = parser.parse_args()
    n_patients = 10_000
    probability_table = pl.read_parquet(
        os.path.join(args.probability_table, "*.parquet")
    )
    tokenizer = ConceptTransitionTokenizer(probability_table)
    batched_seqs = []
    for i in range(n_patients):
        current_token = "[START]"
        tokens = [current_token]
        while current_token != "[END]":
            current_token, _ = tokenizer.sample(current_token, top_k=100)
            tokens.append(current_token)
            if len(tokens) > 1024:
                break
        batched_seqs.append({"concept_ids": tokens})
    pd.DataFrame(batched_seqs, columns=["concept_ids"]).to_parquet(
        os.path.join(args.output_folder, "batched_seqs.parquet")
    )
