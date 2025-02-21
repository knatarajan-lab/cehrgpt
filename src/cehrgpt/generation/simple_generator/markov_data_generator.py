import argparse
import os
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from tqdm import tqdm
from transformers.utils import logging

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_visit_type_token,
)
from cehrgpt.omop.vocab_utils import generate_concept_maps

logger = logging.get_logger("transformers")


class ConceptTransitionTokenizer:
    def __init__(self, prob_df: pl.DataFrame, cache_dir: Optional[Path] = None):
        """
        Initialize tokenizer with conditional probability dataframe.

        Args:
            prob_df: Polars DataFrame with transition probabilities
            cache_dir: Optional directory to cache/load transition matrices
        """
        # Initialize matrices
        self.demographic_matrix = None
        self.visit_type_vector = None
        self.medical_matrices = {}

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Try to load from cache first
            if self._load_from_cache(cache_dir):
                logger.info("Loaded transition matrices from cache")
                return

        logger.info("Initializing vocabulary")
        self._initialize_vocabulary(prob_df)

        # Build matrices if not loaded from cache
        self._build_transition_matrices(prob_df)

        # Save to cache if directory provided
        if cache_dir is not None:
            self._save_to_cache(cache_dir)

    def _save_to_cache(self, cache_dir: Path):
        """Save transition matrices to cache directory."""
        import joblib

        # Save vocabulary
        vocab_file = cache_dir / "vocabulary.joblib"
        joblib.dump(
            {"vocab": self.vocab, "concept_to_idx": self.concept_to_idx}, vocab_file
        )

        # Save demographic matrix
        demographic_file = cache_dir / "demographic_matrix.npz"
        sparse.save_npz(demographic_file, self.demographic_matrix)

        # Save visit type vector
        visit_file = cache_dir / "visit_type_vector.npy"
        np.save(visit_file, self.visit_type_vector)

        # Save medical matrices
        medical_file = cache_dir / "medical_matrices.joblib"
        medical_dict = {k: v.tocoo() for k, v in self.medical_matrices.items()}
        joblib.dump(medical_dict, medical_file)

    def _load_from_cache(self, cache_dir: Path) -> bool:
        """
        Load transition matrices from cache directory.

        Returns:
            bool: True if successfully loaded from cache
        """
        try:
            import joblib

            # Check if all required files exist
            vocab_file = cache_dir / "vocabulary.joblib"
            demographic_file = cache_dir / "demographic_matrix.npz"
            visit_file = cache_dir / "visit_type_vector.npy"
            medical_file = cache_dir / "medical_matrices.joblib"

            if not all(
                f.exists()
                for f in [vocab_file, demographic_file, visit_file, medical_file]
            ):
                return False

            # Load vocabulary
            vocab_data = joblib.load(vocab_file)
            self.vocab = vocab_data["vocab"]
            self.vocab_size = len(self.vocab)
            self.concept_to_idx = vocab_data["concept_to_idx"]
            # Load matrices
            self.demographic_matrix = sparse.load_npz(demographic_file)
            self.visit_type_vector = np.load(visit_file)
            medical_dict = joblib.load(medical_file)
            self.medical_matrices = {k: v.tocsr() for k, v in medical_dict.items()}
            return True

        except Exception as e:
            logger.exception(f"Error loading from cache: {str(e)}")
            return False

    def _initialize_vocabulary(self, prob_df: pl.DataFrame) -> None:
        """Initialize vocabulary with minimal memory usage."""
        concepts1 = prob_df.select("concept_id_1").unique()
        concepts2 = prob_df.select("concept_id_2").unique()
        unique_concepts = pl.concat(
            [
                concepts1.rename({"concept_id_1": "concept"}),
                concepts2.rename({"concept_id_2": "concept"}),
            ]
        ).unique()

        self.vocab = unique_concepts.to_numpy().flatten()
        self.vocab_size = len(self.vocab)
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.vocab)}

    def _build_transition_matrices(self, prob_df: pl.DataFrame) -> None:
        """Build transition matrices using sparse matrices."""
        # Initialize sparse matrices
        self.visit_type_vector = np.zeros(self.vocab_size)

        # Process demographic transitions
        logger.info("Build the demographic transition matrix")
        demographic_df = prob_df.filter(pl.col("age_group") == "age:-10-0")
        row_indices = [
            self.concept_to_idx[row[2]] for row in demographic_df.iter_rows()
        ]
        col_indices = [
            self.concept_to_idx[row[3]] for row in demographic_df.iter_rows()
        ]
        values = [row[6] for row in demographic_df.iter_rows()]
        self.demographic_matrix = sparse.coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(self.vocab_size, self.vocab_size),
        ).tocsr()

        # Process visit type probabilities
        logger.info("Build the visit type probability distribution")
        visit_df = (
            prob_df.filter(pl.col("concept_id_1") == "[VS]")
            .group_by(pl.col("concept_id_1"), pl.col("concept_id_2"))
            .agg(pl.sum("count").alias("count"))
        )
        total_sum = visit_df.select(pl.col("count").sum()).to_numpy()[0, 0]
        for row in visit_df.iter_rows():
            j = self.concept_to_idx[row[1]]
            self.visit_type_vector[j] = row[2] / total_sum

        # Process medical transitions
        # Only keep age groups up to 90-100
        valid_age_groups = [f"age:{i}-{i + 10}" for i in range(0, 91, 10)]

        prob_df = prob_df.filter(
            (pl.col("age_group").is_in(valid_age_groups))
            | (pl.col("age_group") == "age:-10-0")
        )

        medical_df = prob_df.filter(pl.col("age_group") != "age:-10-0")
        # Group by visit type and age group
        for visit_type in medical_df["visit_concept_id"].unique().to_list():
            for age_group in medical_df["age_group"].unique().to_list():
                subset = medical_df.filter(
                    (pl.col("visit_concept_id") == visit_type)
                    & (pl.col("age_group") == age_group)
                )

                if len(subset) == 0:
                    continue
                logger.info(
                    f"Build the transition matrix for age: {age_group} visit: {visit_type}"
                )
                row_indices = [
                    self.concept_to_idx[row[2]] for row in subset.iter_rows()
                ]
                col_indices = [
                    self.concept_to_idx[row[3]] for row in subset.iter_rows()
                ]
                values = [row[6] for row in subset.iter_rows()]

                matrix = sparse.coo_matrix(
                    (values, (row_indices, col_indices)),
                    shape=(self.vocab_size, self.vocab_size),
                ).tocsr()

                # Normalize and add to dictionary
                self._normalize_sparse_matrix(matrix)
                self.medical_matrices[(str(visit_type), age_group)] = matrix

        # Normalize matrices
        self._normalize_sparse_matrix(self.demographic_matrix)
        self.demographic_matrix = self.demographic_matrix.tocsr()

        # Normalize visit type vector
        if self.visit_type_vector.sum() > 0:
            self.visit_type_vector = (
                self.visit_type_vector / self.visit_type_vector.sum()
            )

    def _normalize_sparse_matrix(self, matrix: sparse.lil_matrix) -> None:
        """Normalize sparse matrix rows to sum to 1."""
        # Convert to CSR for efficient row operations
        csr_matrix = matrix.tocsr()

        # Calculate row sums
        row_sums = np.array(csr_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1

        # Create diagonal matrix for normalization
        row_sums_inv = sparse.diags(1 / row_sums)

        # Normalize and convert back to LIL
        normalized = row_sums_inv @ csr_matrix
        matrix[:] = normalized.tolil()

    def sample_first_visit_type(
        self, top_k: Optional[int] = None, random_state: Optional[int] = None
    ) -> Tuple[str, float]:
        """Sample visit type using probability vector."""
        if random_state is not None:
            np.random.seed(random_state)

        probs = self.visit_type_vector
        non_zero_indices = np.nonzero(probs)[0]

        if len(non_zero_indices) == 0:
            raise ValueError("No valid visit types found")

        if top_k is not None:
            top_k = min(top_k, len(non_zero_indices))
            sorted_indices = non_zero_indices[
                np.argsort(probs[non_zero_indices])[-top_k:]
            ]
            selected_probs = probs[sorted_indices]
            selected_probs = selected_probs / selected_probs.sum()
            sampled_idx = sorted_indices[
                np.random.choice(len(sorted_indices), p=selected_probs)
            ]
        else:
            selected_probs = probs[non_zero_indices]
            selected_probs = selected_probs / selected_probs.sum()
            sampled_idx = non_zero_indices[
                np.random.choice(len(non_zero_indices), p=selected_probs)
            ]

        return self.vocab[sampled_idx], probs[sampled_idx]

    def sample_demographic(
        self,
        concept_id: str,
        top_k: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Sample demographic transition using sparse matrix."""
        if random_state is not None:
            np.random.seed(random_state)

        try:
            idx = self.concept_to_idx[concept_id]
            probs = self.demographic_matrix[idx].toarray().flatten()

            non_zero_indices = np.nonzero(probs)[0]
            if len(non_zero_indices) == 0:
                raise ValueError(f"No valid transitions from concept {concept_id}")

            if top_k is not None:
                top_k = min(top_k, len(non_zero_indices))
                sorted_indices = non_zero_indices[
                    np.argsort(probs[non_zero_indices])[-top_k:]
                ]
                selected_probs = probs[sorted_indices]
                selected_probs = selected_probs / selected_probs.sum()
                sampled_idx = sorted_indices[
                    np.random.choice(len(sorted_indices), p=selected_probs)
                ]
            else:
                selected_probs = probs[non_zero_indices]
                selected_probs = selected_probs / selected_probs.sum()
                sampled_idx = non_zero_indices[
                    np.random.choice(len(non_zero_indices), p=selected_probs)
                ]

            return self.vocab[sampled_idx], probs[sampled_idx]

        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")

    def sample(
        self,
        visit_concept_id: str,
        age_group: str,
        concept_id: str,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Sample medical transition using sparse matrix.

        Args:
            visit_concept_id: Current visit type
            age_group: Current age group
            concept_id: Current concept
            top_k: If provided, sample only from top-k most likely concepts
            temperature: Temperature for sampling (higher = more random)
            random_state: Optional random seed
        """
        if random_state is not None:
            np.random.seed(random_state)

        matrix = self.medical_matrices.get((visit_concept_id, age_group))
        if matrix is None:
            raise ValueError(
                f"No transitions for visit type {visit_concept_id} and age group {age_group}"
            )

        try:
            idx = self.concept_to_idx[concept_id]
            probs = matrix[idx].toarray().flatten()

            non_zero_indices = np.nonzero(probs)[0]
            if len(non_zero_indices) == 0:
                raise ValueError(f"No valid transitions for {concept_id}")

            # Apply temperature scaling to probabilities
            if temperature != 1.0:
                # Take log of probabilities to avoid numerical issues
                log_probs = np.log(probs + 1e-10)
                log_probs = log_probs / temperature
                probs = np.exp(log_probs)
                probs = probs / probs.sum()  # Renormalize

            if top_k is not None:
                top_k = min(top_k, len(non_zero_indices))
                sorted_indices = non_zero_indices[
                    np.argsort(probs[non_zero_indices])[-top_k:]
                ]
                selected_probs = probs[sorted_indices]
                selected_probs = selected_probs / selected_probs.sum()
                sampled_idx = sorted_indices[
                    np.random.choice(len(sorted_indices), p=selected_probs)
                ]
            else:
                selected_probs = probs[non_zero_indices]
                selected_probs = selected_probs / selected_probs.sum()
                sampled_idx = non_zero_indices[
                    np.random.choice(len(non_zero_indices), p=selected_probs)
                ]

            return self.vocab[sampled_idx], probs[sampled_idx]

        except KeyError:
            raise KeyError(f"Concept {concept_id} not found in vocabulary")


def create_age_group_udf(age):
    group_number = age // 10
    return f"age:{group_number * 10}-{(group_number + 1) * 10}"


def generate_and_save_sequences(
    tokenizer: ConceptTransitionTokenizer,
    output_dir: Path,
    batch_size: int,
    n_patients: int,
    concept_domain_map: Dict[str, str],
    max_length: int = 1024,
    top_k: int = 100,
    validate: bool = True,
) -> None:
    """Generate synthetic patient sequences and save them in batches."""
    sequences = []
    batch_num = 0

    for i in tqdm(range(n_patients), total=n_patients):
        current_token = "[START]"
        tokens = []  # Don't initialize with START token
        current_age: Optional[int] = None
        start_year: Optional[int] = None

        try:
            # Generate 4 demographic tokens
            for _ in range(4):
                current_token, _ = tokenizer.sample_demographic(
                    current_token, top_k=top_k
                )
                tokens.append(current_token)
                if current_token.startswith("year:"):
                    start_year = int(current_token.split(":")[1])
                elif current_token.startswith("age:"):
                    current_age = int(current_token.split(":")[1].split("-")[0])

            tokens.append("[VS]")
            visit_concept_id, _ = tokenizer.sample_first_visit_type(top_k=top_k)
            tokens.append(visit_concept_id)
            current_token = visit_concept_id

            if any(x is None for x in [current_age, visit_concept_id, start_year]):
                continue

            # Initialize temporal tracking
            birth_year = start_year - current_age
            current_date = date(start_year, 1, 1)
            age_group = create_age_group_udf(current_age)

            logger.debug(f"Initial age_group: {age_group}")
            logger.debug(f"Initial visit_concept_id: {visit_concept_id}")
            while len(tokens) < max_length:
                try:
                    current_token, _ = tokenizer.sample(
                        visit_concept_id, age_group, current_token, top_k=top_k
                    )
                    if current_token == "[END]":
                        break
                    tokens.append(current_token)

                    if is_att_token(current_token):
                        day_delta = extract_time_interval_in_days(current_token)
                        current_date += timedelta(days=day_delta)
                        current_age = current_date.year - birth_year
                        age_group = create_age_group_udf(current_age)
                        logger.debug(f"Updated age_group: {age_group}")
                    elif is_visit_type_token(current_token):
                        visit_concept_id = current_token

                except ValueError as e:
                    logger.error(f"Error in sequence generation: {e}")
                    break

            if tokens[-1] == "[VE]":
                if validate:
                    cehrgpt_patient = get_cehrgpt_patient_converter(
                        tokens, concept_domain_map
                    )
                    if cehrgpt_patient.is_validation_passed:
                        sequences.append({"concept_ids": tokens})
                    else:
                        logger.warning(
                            f"The validation failed due to: {cehrgpt_patient.get_error_messages()}"
                        )
                else:
                    sequences.append({"concept_ids": tokens})

        except Exception as e:
            logger.error(f"Error generating sequence {i}: {str(e)}")
            continue

        # Save batch when full
        if len(sequences) >= batch_size or i == n_patients - 1:
            if sequences:
                batch_df = pd.DataFrame(sequences)
                output_file = output_dir / f"batch_{batch_num}_{uuid.uuid4()}.parquet"
                batch_df.to_parquet(output_file)
                sequences = []
                batch_num += 1
                logger.info(
                    f"Saved batch {batch_num}, processed {i + 1}/{n_patients} patients"
                )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient sequences")
    parser.add_argument(
        "--probability_table", required=True, help="Path to probability table"
    )
    parser.add_argument("--vocabulary_dir", required=True, help="Vocabulary directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--cache_dir", required=True, help="Output directory")
    parser.add_argument(
        "--n_patients", required=True, type=int, help="Number of patients to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of sequences to generate before saving to disk",
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Top k tokens to sample from"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (higher = more random, lower = more deterministic)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir = output_dir / f"top_k{args.top_k}_temp_{int(args.temperature * 100)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probability table and initialize tokenizer
    logger.info("Loading probability table...")
    prob_table = pl.read_parquet(os.path.join(args.probability_table, "*.parquet"))
    logger.info("Building the transition matrix...")
    tokenizer = ConceptTransitionTokenizer(prob_table, args.cache_dir)
    logger.info("Building the concept table...")
    concept = pl.read_parquet(os.path.join(args.vocabulary_dir, "concept", "*.parquet"))
    _, concept_domain_map = generate_concept_maps(concept)

    logger.info(
        f"Generating {args.n_patients} sequences in batches of {args.batch_size}..."
    )
    try:
        generate_and_save_sequences(
            tokenizer=tokenizer,
            output_dir=output_dir,
            batch_size=args.batch_size,
            n_patients=args.n_patients,
            concept_domain_map=concept_domain_map,
            validate=args.validate,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        logger.info("Generation completed successfully!")
    except Exception as e:
        logger.error(f"Error during sequence generation: {e}")
        raise


if __name__ == "__main__":
    main()
