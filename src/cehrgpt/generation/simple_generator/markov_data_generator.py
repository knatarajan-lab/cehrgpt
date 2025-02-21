import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

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
    def __init__(self, prob_df: pl.DataFrame):
        """Initialize tokenizer with conditional probability dataframe."""
        self._initialize_vocabulary(prob_df)

        # Store probabilities in dictionaries instead of full matrices
        self.demographic_transitions = {}  # {(concept_id_1, concept_id_2): prob}
        self.medical_transitions = (
            {}
        )  # {(visit_type, age_group, concept_id_1, concept_id_2): prob}
        self.visit_type_transitions = {}  # {visit_type: prob}

        self._build_transition_probabilities(prob_df)

    def _initialize_vocabulary(self, prob_df: pl.DataFrame) -> None:
        """Initialize vocabulary with minimal memory usage."""
        concepts1 = prob_df.select("concept_id_1").unique()
        concepts2 = prob_df.select("concept_id_2").unique()

        # Use string concatenation instead of concat for memory efficiency
        unique_concepts = pl.concat(
            [
                concepts1.rename({"concept_id_1": "concept"}),
                concepts2.rename({"concept_id_2": "concept"}),
            ]
        ).unique()

        self.vocab = unique_concepts.to_numpy().flatten()
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.vocab)}

    def _build_transition_probabilities(self, prob_df: pl.DataFrame) -> None:
        """Build transition probabilities using sparse storage."""
        # Process demographic transitions
        demographic_df = prob_df.filter(pl.col("age_group") == "age:-10-0")
        for row in demographic_df.iter_rows():
            self.demographic_transitions[(row[2], row[3])] = row[6]

        # Process visit type probabilities
        visit_df = prob_df.filter(pl.col("concept_id_1") == "[VS]")
        for row in visit_df.iter_rows():
            self.visit_type_transitions[row[3]] = row[6]

        # Process medical transitions
        medical_df = prob_df.filter(pl.col("age_group") != "age:-10-0")
        for row in medical_df.iter_rows():
            key = (str(row[0]), row[1], row[2], row[3])
            self.medical_transitions[key] = row[6]

        # Normalize probabilities within groups
        self._normalize_probabilities()

    def _normalize_probabilities(self):
        """Normalize probabilities within their respective groups."""
        # Normalize visit type probabilities
        total = sum(self.visit_type_transitions.values())
        if total > 0:
            for k in self.visit_type_transitions:
                self.visit_type_transitions[k] /= total

        # Normalize demographic transitions by source concept
        demographic_sums = {}
        for (src, dst), prob in self.demographic_transitions.items():
            demographic_sums[src] = demographic_sums.get(src, 0) + prob
        for (src, dst), prob in self.demographic_transitions.items():
            if demographic_sums[src] > 0:
                self.demographic_transitions[(src, dst)] = prob / demographic_sums[src]

        # Normalize medical transitions by visit type, age group, and source concept
        medical_sums = {}
        for (visit, age, src, dst), prob in self.medical_transitions.items():
            key = (visit, age, src)
            medical_sums[key] = medical_sums.get(key, 0) + prob
        for (visit, age, src, dst), prob in self.medical_transitions.items():
            key = (visit, age, src)
            if medical_sums[key] > 0:
                self.medical_transitions[(visit, age, src, dst)] = (
                    prob / medical_sums[key]
                )

    def sample_first_visit_type(
        self, top_k: Optional[int] = None, random_state: Optional[int] = None
    ):
        """Sample visit type using dictionary-based probabilities."""
        if random_state is not None:
            np.random.seed(random_state)

        visits, probs = zip(*self.visit_type_transitions.items())

        if top_k is not None:
            top_k = min(top_k, len(visits))
            sorted_indices = np.argsort(probs)[-top_k:]
            visits = [visits[i] for i in sorted_indices]
            probs = [probs[i] for i in sorted_indices]
            probs = np.array(probs) / sum(probs)

        sampled_idx = np.random.choice(len(visits), p=probs)
        return visits[sampled_idx], probs[sampled_idx]

    def sample_demographic(
        self,
        concept_id: str,
        top_k: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Sample demographic transition using dictionary-based probabilities."""
        if random_state is not None:
            np.random.seed(random_state)

        # Get all possible transitions from current concept
        transitions = [
            (dst, prob)
            for (src, dst), prob in self.demographic_transitions.items()
            if src == concept_id
        ]

        if not transitions:
            raise ValueError(f"No valid transitions from concept {concept_id}")

        concepts, probs = zip(*transitions)

        if top_k is not None:
            top_k = min(top_k, len(concepts))
            sorted_indices = np.argsort(probs)[-top_k:]
            concepts = [concepts[i] for i in sorted_indices]
            probs = [probs[i] for i in sorted_indices]
            probs = np.array(probs) / sum(probs)

        sampled_idx = np.random.choice(len(concepts), p=probs)
        return concepts[sampled_idx], probs[sampled_idx]

    def sample(
        self,
        visit_concept_id: str,
        age_group: str,
        concept_id: str,
        top_k: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Sample medical transition using dictionary-based probabilities."""
        if random_state is not None:
            np.random.seed(random_state)

        # Get all possible transitions for the current state
        transitions = [
            (dst, prob)
            for (visit, age, src, dst), prob in self.medical_transitions.items()
            if visit == visit_concept_id and age == age_group and src == concept_id
        ]

        if not transitions:
            raise ValueError(
                f"No valid transitions for {visit_concept_id}, {age_group}, {concept_id}"
            )

        concepts, probs = zip(*transitions)

        if top_k is not None:
            top_k = min(top_k, len(concepts))
            sorted_indices = np.argsort(probs)[-top_k:]
            concepts = [concepts[i] for i in sorted_indices]
            probs = [probs[i] for i in sorted_indices]
            probs = np.array(probs) / sum(probs)

        sampled_idx = np.random.choice(len(concepts), p=probs)
        return concepts[sampled_idx], probs[sampled_idx]


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
