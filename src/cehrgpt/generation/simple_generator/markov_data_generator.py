import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm  # Add this import

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.omop.vocab_utils import generate_concept_maps
from cehrgpt.tools.generate_causal_patient_split_by_age import age_group_func


class ConceptTransitionTokenizer:
    """A tokenizer for handling concept transitions with probability-based sampling."""

    def __init__(self, prob_df: pl.DataFrame):
        """
        Initialize tokenizer with conditional probability dataframe.

        Args:
            prob_df: Polars DataFrame with columns [concept_id_1, concept_id_2, prob]
                    representing transition probabilities between concepts
        """
        self.vocab_size = 0
        self.concept_to_token = {}
        self.token_to_concept = {}
        self.transition_matrix = None
        self.visit_type_probability = None
        self.demographic_transition_matrix = None
        self._initialize_vocabulary(prob_df)
        self._build_transition_matrix(prob_df)

    def _initialize_vocabulary(self, prob_df: pl.DataFrame) -> None:
        """Initialize vocabulary mappings from probability dataframe."""
        # Get unique concepts from both columns
        concepts1 = (
            prob_df.select("concept_id_1").unique().rename({"concept_id_1": "concept"})
        )
        concepts2 = (
            prob_df.select("concept_id_2").unique().rename({"concept_id_2": "concept"})
        )
        unique_concepts = pl.concat([concepts1, concepts2]).unique()

        # Create token mappings
        self.concept_to_token = {
            concept: idx
            for idx, concept in enumerate(unique_concepts.to_numpy().flatten())
        }
        self.token_to_concept = {v: k for k, v in self.concept_to_token.items()}
        self.vocab_size = len(self.concept_to_token)

    def _build_transition_matrix(self, prob_df: pl.DataFrame) -> None:
        """Build and normalize the transition probability matrix."""

        self.transition_matrix: Dict[Tuple[str, str], np.ndarray] = {}
        self.demographic_transition_matrix: np.ndarray = np.zeros(
            (self.vocab_size, self.vocab_size)
        )
        self.visit_type_probability: np.ndarray = np.zeros(self.vocab_size)

        # Fill demographic transition matrix with probabilities
        for row in prob_df.filter(pl.col("age_group") == "age:-10-0").iter_rows():
            concept_id_1 = row[2]
            concept_id_2 = row[3]
            prob = row[6]
            i, j = (
                self.concept_to_token[concept_id_1],
                self.concept_to_token[concept_id_2],
            )
            self.demographic_transition_matrix[i, j] = prob

        # Fill visit type probability distribution with probabilities
        for row in prob_df.filter(pl.col("concept_id_1") == "[VS]").iter_rows():
            concept_id_2 = row[3]
            prob = row[6]
            self.visit_type_probability[self.concept_to_token[concept_id_2]] = prob

        # Fill transition matrix with probabilities
        for row in prob_df.filter(pl.col("age_group") != "age:-10-0").iter_rows():
            visit_concept_id = str(row[0])
            age_group = row[1]
            concept_id_1 = row[2]
            concept_id_2 = row[3]
            prob = row[6]
            if (visit_concept_id, age_group) not in self.transition_matrix:
                self.transition_matrix[(visit_concept_id, age_group)] = np.zeros(
                    (self.vocab_size, self.vocab_size)
                )
            i, j = (
                self.concept_to_token[concept_id_1],
                self.concept_to_token[concept_id_2],
            )
            self.transition_matrix[(visit_concept_id, age_group)][i, j] = prob

        # Normalize probabilities
        for demographic_group in self.transition_matrix.keys():
            row_sums = self.transition_matrix[demographic_group].sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.transition_matrix[demographic_group] = (
                self.transition_matrix[demographic_group] / row_sums[:, np.newaxis]
            )

        # Normalize demographic transition matrix
        row_sums = self.demographic_transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.demographic_transition_matrix = (
            self.demographic_transition_matrix / row_sums[:, np.newaxis]
        )

        # Normalize visit type probability
        if self.visit_type_probability.sum() > 0:
            self.visit_type_probability = (
                self.visit_type_probability / self.visit_type_probability.sum()
            )

    def encode(self, concepts: List[str]) -> List[int]:
        """Convert concept IDs to token indices."""
        try:
            return [self.concept_to_token[c] for c in concepts]
        except KeyError as e:
            raise KeyError(f"Concept {e} not found in vocabulary")

    def decode(self, tokens: List[int]) -> List[str]:
        """Convert token indices back to concept IDs."""
        try:
            return [self.token_to_concept[t] for t in tokens]
        except KeyError as e:
            raise KeyError(f"Token {e} not found in vocabulary")

    def get_transition_prob(
        self,
        visit_concept_id: str,
        age_group: str,
        concept_id_1: str,
        concept_id_2: str,
    ) -> float:
        """Get transition probability between two concepts."""
        try:
            token1 = self.concept_to_token[concept_id_1]
            token2 = self.concept_to_token[concept_id_2]
            return self.transition_matrix[(visit_concept_id, age_group)][token1, token2]
        except KeyError as e:
            raise KeyError(f"Concept {e} not found in vocabulary")

    def get_next_medical_concepts(
        self, visit_concept_id: str, age_group: str, concept_id: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k most likely next concepts and their probabilities."""
        if not 1 <= top_k <= self.vocab_size:
            raise ValueError(f"top_k must be between 1 and {self.vocab_size}")

        try:
            token = self.concept_to_token[concept_id]
            probs = self.transition_matrix[(visit_concept_id, age_group)][token]
            top_indices = np.argsort(probs)[-top_k:][::-1]
            return [(self.token_to_concept[idx], probs[idx]) for idx in top_indices]
        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")

    def get_next_demographic_concepts(
        self, concept_id: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k most likely next concepts and their probabilities."""
        if not 1 <= top_k <= self.vocab_size:
            raise ValueError(f"top_k must be between 1 and {self.vocab_size}")

        try:
            token = self.concept_to_token[concept_id]
            probs = self.demographic_transition_matrix[token]
            top_indices = np.argsort(probs)[-top_k:][::-1]
            return [(self.token_to_concept[idx], probs[idx]) for idx in top_indices]
        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")

    def sample_first_visit_type(
        self, top_k: Optional[int] = None, random_state: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Randomly sample a visit type based on visit type probabilities.

        Args:
            top_k: If provided, sample only from top-k most likely visit types
            random_state: Optional random seed for reproducibility

        Returns:
            Tuple of (sampled_visit_type, probability)

        Raises:
            ValueError: If top_k is less than 1 or greater than vocabulary size
        """
        if random_state is not None:
            np.random.seed(random_state)

        probs = self.visit_type_probability

        if top_k is not None:
            if not 1 <= top_k <= self.vocab_size:
                raise ValueError(f"top_k must be between 1 and {self.vocab_size}")

            # Get top-k indices and their probabilities
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]

            # Renormalize probabilities of top-k concepts
            top_probs = top_probs / top_probs.sum()

            # Sample from top-k
            sampled_idx = np.random.choice(top_indices, p=top_probs)
        else:
            # Ensure probabilities sum to 1
            probs = probs / probs.sum()
            sampled_idx = np.random.choice(self.vocab_size, p=probs)

        return self.token_to_concept[sampled_idx], probs[sampled_idx]

    def sample_demographic(
        self,
        concept_id: str,
        top_k: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Randomly sample a demographic concept based on demographic transition probabilities.

        Args:
            concept_id: Source concept string
            top_k: If provided, sample only from top-k most likely demographic concepts
            random_state: Optional random seed for reproducibility

        Returns:
            Tuple of (sampled_demographic_concept, probability)

        Raises:
            KeyError: If the concept is not in the vocabulary
            ValueError: If top_k is less than 1 or greater than vocabulary size
        """
        if random_state is not None:
            np.random.seed(random_state)

        try:
            token = self.concept_to_token[concept_id]
            probs = self.demographic_transition_matrix[token]

            if top_k is not None:
                if not 1 <= top_k <= self.vocab_size:
                    raise ValueError(f"top_k must be between 1 and {self.vocab_size}")

                # Get top-k indices and their probabilities
                top_indices = np.argsort(probs)[-top_k:]
                top_probs = probs[top_indices]

                # Renormalize probabilities of top-k concepts
                top_probs = top_probs / top_probs.sum()

                # Sample from top-k
                sampled_idx = np.random.choice(top_indices, p=top_probs)
            else:
                # Ensure probabilities sum to 1
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    sampled_idx = np.random.choice(self.vocab_size, p=probs)
                else:
                    raise ValueError(f"No valid transitions from concept {concept_id}")

            return self.token_to_concept[sampled_idx], probs[sampled_idx]

        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")
        except ValueError as e:
            if "probabilities do not sum to 1" in str(e):
                probs = probs / probs.sum()
                return self.sample_demographic(concept_id, top_k, random_state)
            raise e

    def sample(
        self,
        visit_concept_id: str,
        age_group: str,
        concept_id: str,
        top_k: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Randomly sample a next concept based on transition probabilities.

        Args:
            concept_id: Source concept string
            top_k: If provided, sample only from top-k most likely concepts
            random_state: Optional random seed for reproducibility

        Returns:
            Tuple of (sampled_concept, probability)
        """
        if (visit_concept_id, age_group) not in self.transition_matrix:
            return "0", 0.0

        if random_state is not None:
            np.random.seed(random_state)

        try:
            token = self.concept_to_token[concept_id]
            probs = self.transition_matrix[(visit_concept_id, age_group)][token]

            if top_k is not None:
                if not 1 <= top_k <= self.vocab_size:
                    raise ValueError(f"top_k must be between 1 and {self.vocab_size}")

                top_indices = np.argsort(probs)[-top_k:]
                top_probs = probs[top_indices]
                top_probs = top_probs / top_probs.sum()  # Renormalize

                sampled_idx = np.random.choice(top_indices, p=top_probs)
            else:
                sampled_idx = np.random.choice(self.vocab_size, p=probs)

            return self.token_to_concept[sampled_idx], probs[sampled_idx]

        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")
        except ValueError as e:
            if "probabilities do not sum to 1" in str(e):
                probs = probs / probs.sum()
                return self.sample(
                    visit_concept_id, age_group, concept_id, top_k, random_state
                )
            raise e


def generate_and_save_sequences(
    tokenizer: ConceptTransitionTokenizer,
    output_dir: Path,
    batch_size: int,
    n_patients: int,
    concept_domain_map: Dict[str, str],
    max_length: int = 1024,
    top_k: int = 100,
) -> None:
    """
    Generate synthetic patient sequences and save them in batches.

    Args:
        tokenizer: Initialized ConceptTransitionTokenizer
        output_dir: Output directory path
        batch_size: Number of sequences to generate before saving to disk
        n_patients: Total number of patients to generate
        concept_domain_map: concept to domain map
        max_length: Maximum sequence length
        top_k: Top k concepts to sample from
    """
    sequences = []
    batch_num = 0

    for i in tqdm(range(n_patients), total=n_patients):
        current_token = "[START]"
        tokens = []

        age_group = None
        visit_concept_id = None

        while len(tokens) < max_length:
            if len(tokens) < 4:
                current_token, _ = tokenizer.sample_demographic(
                    current_token, top_k=top_k
                )
                tokens.append(current_token)  # Add demographic tokens
                if current_token.startswith("age:"):
                    age_group = age_group_func(current_token)
            elif len(tokens) == 5:
                # The first visit type will be generated at the 6th position
                current_token, _ = tokenizer.sample_first_visit_type(
                    top_k=top_k
                )  # Added top_k parameter
                visit_concept_id = current_token
                tokens.append(current_token)  # Add visit type token
            elif visit_concept_id is not None and age_group is not None:
                current_token, _ = tokenizer.sample(
                    visit_concept_id, age_group, current_token, top_k=top_k
                )
                if current_token == "[END]":
                    tokens.append(current_token)  # Add END token
                    break
                else:
                    tokens.append(current_token)

        if len(tokens) < 5:  # Minimum sequence length check
            print(f"Sequence too short: {tokens}")
            continue

        cehrgpt_patient = get_cehrgpt_patient_converter(tokens, concept_domain_map)
        if cehrgpt_patient.is_validation_passed:
            sequences.append({"concept_ids": tokens})
        else:
            print(
                f"Invalid generated patient sequence due to: {cehrgpt_patient.get_error_messages()}"
            )

        # When batch is full or we've reached the end, save to disk
        if len(sequences) >= batch_size or i == n_patients - 1:
            batch_df = pd.DataFrame(sequences)
            output_file = output_dir / f"batch_{batch_num}_{uuid.uuid4()}.parquet"
            batch_df.to_parquet(output_file)

            # Clear sequences list for next batch
            sequences = []
            batch_num += 1
            print(f"Saved batch {batch_num}, processed {i + 1}/{n_patients} patients")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient sequences")
    parser.add_argument(
        "--probability_table", required=True, help="Path to probability table"
    )
    parser.add_argument("--vocabulary_dir", required=True, help="Vocabulary directory")
    parser.add_argument("--output_folder", required=True, help="Output directory")
    parser.add_argument(
        "--n_patients", required=True, type=int, help="Number of patients to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of sequences to generate before saving to disk",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probability table and initialize tokenizer
    print("Loading probability table...")
    prob_table = pl.read_parquet(os.path.join(args.probability_table, "*.parquet"))
    print("Building the transition matrix...")
    tokenizer = ConceptTransitionTokenizer(prob_table)
    print("Building the concept table...")
    concept = pl.read_parquet(os.path.join(args.vocabulary_dir, "concept", "*.parquet"))
    _, concept_domain_map = generate_concept_maps(concept)

    print(f"Generating {args.n_patients} sequences in batches of {args.batch_size}...")
    try:
        generate_and_save_sequences(
            tokenizer=tokenizer,
            output_dir=output_dir,
            batch_size=args.batch_size,
            n_patients=args.n_patients,
            concept_domain_map=concept_domain_map,
        )
        print("Generation completed successfully!")
    except Exception as e:
        print(f"Error during sequence generation: {e}")
        raise


if __name__ == "__main__":
    main()
