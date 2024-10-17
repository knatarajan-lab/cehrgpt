import logging
import os
import random
import sys
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import yaml

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

from .utils import (
    batched_pairwise_euclidean_distance_indices,
    create_demographics,
    create_gender_encoder,
    create_race_encoder,
    create_vector_representations_for_attribute,
    find_match,
    find_match_self,
    scale_age,
)

RANDOM_SEE = 42
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG = logging.getLogger("attribute_inference")


def safe_divide(numerator_vector, denominator_vector):
    return np.where(denominator_vector > 0, numerator_vector / denominator_vector, 0)


def cal_f1_score(vector_a, vector_b, index_matrix):
    # vector_a is train data and vector_b is synthetic data or iteself
    shared_vector = np.logical_and(
        vector_a[: len(index_matrix)], vector_b[index_matrix]
    ).astype(int)
    shared_vector_sum = np.sum(shared_vector, axis=1)

    precision = safe_divide(shared_vector_sum, np.sum(vector_b, axis=1))
    recall = safe_divide(shared_vector_sum, np.sum(vector_a, axis=1))

    f1 = safe_divide(2 * recall * precision, recall + precision)
    return f1, precision, recall


def main(args):
    try:
        with open(args.attribute_config, "r") as file:
            data = yaml.safe_load(file)
        if "common_attributes" in data:
            common_attributes = data["common_attributes"]
        if "sensitive_attributes" in data:
            sensitive_attributes = data["sensitive_attributes"]
    except Union[FileNotFoundError, PermissionError, OSError] as e:
        sys.exit(e)

    attribute_inference_folder = os.path.join(args.output_folder, "attribute_inference")
    if not os.path.exists(attribute_inference_folder):
        LOG.info(
            f"Creating the attribute_inference output folder at {attribute_inference_folder}"
        )
        os.makedirs(attribute_inference_folder, exist_ok=True)

    LOG.info(f"Started loading tokenizer at {args.tokenizer_path}")
    concept_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_path)

    LOG.info(f"Started loading training data at {args.training_data_folder}")
    train_data = pd.read_parquet(args.training_data_folder)

    LOG.info(f"Started loading synthetic_data at {args.synthetic_data_folder}")
    synthetic_data = pd.read_parquet(args.synthetic_data_folder)

    LOG.info(
        "Started extracting the demographic information from the patient sequences"
    )
    train_data = create_demographics(train_data)
    synthetic_data = create_demographics(synthetic_data)

    LOG.info("Started rescaling age columns")
    train_data = scale_age(train_data)
    synthetic_data = scale_age(synthetic_data)

    LOG.info("Started encoding gender")
    gender_encoder = create_gender_encoder(
        train_data,
        # TODO need to change this function to be generic
        train_data[:10],
        synthetic_data,
    )
    LOG.info("Completed encoding gender")

    LOG.info("Started encoding race")
    race_encoder = create_race_encoder(
        train_data,
        # TODO need to change this function to be generic
        train_data[:10],
        synthetic_data,
    )
    LOG.info("Completed encoding race")

    random.seed(RANDOM_SEE)
    for i in range(1, args.n_iterations + 1):
        LOG.info(f"Iteration {i}: Started creating data samples")
        train_data_sample = train_data.sample(args.num_of_samples)
        synthetic_data_sample = synthetic_data.sample(args.num_of_samples)
        LOG.info(f"Iteration {i}: Started creating train sample vectors")
        train_common_vectors, train_sensitive_vectors = (
            create_vector_representations_for_attribute(
                train_data_sample,
                concept_tokenizer,
                gender_encoder,
                race_encoder,
                common_attributes=common_attributes,
                sensitive_attributes=sensitive_attributes,
            )
        )

        LOG.info(f"Iteration {i}: Started creating synthetic vectors")
        synthetic_common_vectors, synthetic_sensitive_vectors = (
            create_vector_representations_for_attribute(
                synthetic_data_sample,
                concept_tokenizer,
                gender_encoder,
                race_encoder,
                common_attributes=common_attributes,
                sensitive_attributes=sensitive_attributes,
            )
        )

        LOG.info(
            f"Started calculating the distances between synthetic and training vectors"
        )
        if args.batched:
            train_synthetic_index = batched_pairwise_euclidean_distance_indices(
                train_common_vectors,
                synthetic_common_vectors,
                batch_size=args.batch_size,
            )
            train_train_index = batched_pairwise_euclidean_distance_indices(
                train_common_vectors,
                train_common_vectors,
                batch_size=args.batch_size,
                self_exclude=True,
            )
        else:
            train_synthetic_index = find_match(
                train_common_vectors, synthetic_common_vectors, return_index=True
            )
            train_train_index = find_match_self(
                train_common_vectors, train_common_vectors, return_index=True
            )

        f1_syn_train, precision_syn_train, recall_syn_train = cal_f1_score(
            train_sensitive_vectors, synthetic_sensitive_vectors, train_synthetic_index
        )
        f1_train_train, precision_train_train, recall_train_train = cal_f1_score(
            train_sensitive_vectors, train_sensitive_vectors, train_train_index
        )

        results = {
            "Precision Synthetic Train": precision_syn_train,
            "Recall Synthetic Train": recall_syn_train,
            "F1 Synthetic Train": f1_syn_train,
            "Precision Train Train": precision_train_train,
            "Recall Train Train": recall_train_train,
            "F1 Train Train": f1_train_train,
        }
        LOG.info(
            f"Attribute Inference: Average Precision Synthetic Train: {np.mean(precision_syn_train)} \n"
            f"Attribute Inference: Average Recall Synthetic Train:{np.mean(recall_syn_train)} \n"
            f"Attribute Inference: Average F1 Synthetic Train: {np.mean(f1_syn_train)} \n"
            f"Attribute Inference: Average Precision Train Train: {np.mean(precision_train_train)} \n"
            f"Attribute Inference: Average Recall Train Train: {np.mean(recall_train_train)} \n"
            f"Attribute Inference: Average F1 Train Train: {np.mean(f1_train_train)}"
        )
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        pd.DataFrame(
            results,
            columns=[
                "Precision Synthetic Train",
                "Recall Synthetic Train",
                "F1 Synthetic Train",
                "Precision Train Train",
                "Recall Train Train",
                "F1 Train Train",
            ],
        ).to_parquet(
            os.path.join(
                attribute_inference_folder,
                f"attribute_inference_{current_time}.parquet",
            )
        )


def create_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Attribute Inference Analysis Arguments"
    )
    parser.add_argument(
        "--training_data_folder",
        dest="training_data_folder",
        action="store",
        help="The path for where the training data folder",
        required=True,
    )
    parser.add_argument(
        "--synthetic_data_folder",
        dest="synthetic_data_folder",
        action="store",
        help="The path for where the synthetic data folder",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The output folder that stores the metrics",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        dest="tokenizer_path",
        action="store",
        help="The path to ConceptTokenizer",
        required=True,
    )
    parser.add_argument(
        "--attribute_config",
        dest="attribute_config",
        action="store",
        help="The configuration yaml file for common and sensitive attributes",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        action="store",
        type=int,
        default=1000,
        help="The batch size of the matching algorithm",
        required=False,
    )
    parser.add_argument(
        "--batched",
        dest="batched",
        action="store_true",
        help="Indicate whether we want to use the batch matrix operation",
    )
    parser.add_argument(
        "--num_of_samples",
        dest="num_of_samples",
        action="store",
        type=int,
        required=False,
        default=5000,
    )
    parser.add_argument(
        "--n_iterations",
        dest="n_iterations",
        action="store",
        type=int,
        required=False,
        default=1,
    )
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
