import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

LOG = logging.getLogger(__name__)

RANDOM_SEE = 42
NUM_OF_GENDERS = 3
NUM_OF_RACES = 21


def get_demographics(concept_ids):
    year_token, age_token, gender, race = concept_ids[0:4]
    try:
        year = int(year_token[5:])
    except ValueError:
        LOG.error(
            f"{year_token[5:]} cannot be converted to an integer, use the default value 1900"
        )
        year = 1900

    try:
        age = int(age_token[4:])
    except ValueError:
        LOG.error(
            f"{age_token[4:]} cannot be converted to an integer, use the default value 1900"
        )
        age = -1

    return year, age, gender, race


def create_race_encoder(
    train_data_sample, evaluation_data_sample, synthetic_data_sample
):
    race_encoder = OneHotEncoder()
    all_unique_races = np.unique(
        np.concatenate(
            [
                train_data_sample.race.unique(),
                evaluation_data_sample.race.unique(),
                synthetic_data_sample.race.unique(),
            ],
            axis=0,
        )
    )
    race_encoder.fit(all_unique_races[:, np.newaxis])
    return race_encoder


def create_gender_encoder(
    train_data_sample, evaluation_data_sample, synthetic_data_sample
):
    gender_encoder = OneHotEncoder()
    all_unique_genders = np.unique(
        np.concatenate(
            [
                train_data_sample.gender.unique(),
                evaluation_data_sample.gender.unique(),
                synthetic_data_sample.gender.unique(),
            ],
            axis=0,
        )
    )
    gender_encoder.fit(all_unique_genders[:, np.newaxis])
    return gender_encoder


def extract_medical_concepts(concept_ids):
    concept_ids = [_ for _ in concept_ids[4:] if str.isnumeric(_)]
    return list(set(concept_ids))


def create_binary_format(concept_ids, concept_tokenizer):
    indices = np.array(concept_tokenizer.encode(concept_ids)).flatten().astype(int)
    embeddings = np.zeros(concept_tokenizer.vocab_size)
    embeddings.put(indices, 1)
    return embeddings


def extract_common_sensitive_concepts(
    dataset, concept_tokenizer, common_attributes, sensitive_attributes
):
    common_embeddings_list = []
    sensitive_embeddings_list = []
    for _, pat_seq in dataset.concept_ids.items():
        concept_ids = extract_medical_concepts(pat_seq)
        common_concept_ids = [_ for _ in concept_ids if _ in common_attributes]
        sensitive_concept_ids = [_ for _ in concept_ids if _ in sensitive_attributes]

        common_embeddings_list.append(
            create_binary_format(common_concept_ids, concept_tokenizer)
        )
        sensitive_embeddings_list.append(
            create_binary_format(sensitive_concept_ids, concept_tokenizer)
        )
    return np.array(common_embeddings_list), np.array(sensitive_embeddings_list)


def transform_concepts(dataset, concept_tokenizer):
    embedding_list = []
    for _, pat_seq in dataset.concept_ids.items():
        embedding_list.append(
            create_binary_format(extract_medical_concepts(pat_seq), concept_tokenizer)
        )
    return np.asarray(embedding_list)


def scale_age(dataset):
    ages = dataset.concept_ids.apply(
        lambda concept_list: get_demographics(concept_list)[1]
    )
    dataset = dataset[ages >= 0]
    ages = ages[ages >= 0]
    max_age = ages.max()
    dataset["scaled_age"] = ages / max_age
    return dataset


def create_demographics(dataset):
    genders = dataset.concept_ids.apply(
        lambda concept_list: get_demographics(concept_list)[2]
    )
    races = dataset.concept_ids.apply(
        lambda concept_list: get_demographics(concept_list)[3]
    )
    dataset["gender"] = genders
    dataset["race"] = races
    return dataset


def create_vector_representations(
    dataset, concept_tokenizer, gender_encoder, race_encoder
):
    concept_vectors = transform_concepts(dataset, concept_tokenizer)
    gender_vectors = gender_encoder.transform(
        dataset.gender.to_numpy()[:, np.newaxis]
    ).todense()
    race_vectors = race_encoder.transform(
        dataset.race.to_numpy()[:, np.newaxis]
    ).todense()
    age_vectors = dataset.scaled_age.to_numpy()[:, np.newaxis]

    pat_vectors = np.concatenate(
        [age_vectors, gender_vectors, race_vectors, concept_vectors], axis=-1
    )

    return np.asarray(pat_vectors)


def create_vector_representations_for_attribute(
    dataset,
    concept_tokenizer,
    gender_encoder,
    race_encoder,
    common_attributes,
    sensitive_attributes,
):
    common_concept_vectors, sensitive_concept_vectors = (
        extract_common_sensitive_concepts(
            dataset, concept_tokenizer, common_attributes, sensitive_attributes
        )
    )
    gender_vectors = gender_encoder.transform(
        dataset.gender.to_numpy()[:, np.newaxis]
    ).todense()
    race_vectors = race_encoder.transform(
        dataset.race.to_numpy()[:, np.newaxis]
    ).todense()
    age_vectors = dataset.scaled_age.to_numpy()[:, np.newaxis]

    common_pat_vectors = np.concatenate(
        [age_vectors, gender_vectors, race_vectors, common_concept_vectors], axis=-1
    )

    return np.asarray(common_pat_vectors), np.asarray(sensitive_concept_vectors)


def batched_pairwise_euclidean_distance_indices(A, B, batch_size, self_exclude=False):
    # Initialize arrays to hold the minimum distances and indices for each point in A
    min_distances = np.full((A.shape[0],), np.inf)
    min_indices = np.full((A.shape[0],), -1, dtype=int)

    # Iterate over A in batches
    for i in tqdm(range(0, A.shape[0], batch_size), total=A.shape[0] // batch_size + 1):
        end_i = i + batch_size
        A_batch = A[i:end_i]

        # Adjust the identity matrix size based on the actual batch size
        actual_batch_size = A_batch.shape[0]

        # Iterate over B in batches
        for j in range(0, B.shape[0], batch_size):
            end_j = j + batch_size
            B_batch = B[j:end_j]

            # Compute distances between the current batches of A and B
            distances = np.sqrt(
                np.sum(
                    (A_batch[:, np.newaxis, :] - B_batch[np.newaxis, :, :]) ** 2, axis=2
                )
            )

            # Apply the identity matrix to exclude self-matches if required
            if self_exclude and i == j:
                identity_matrix = np.eye(actual_batch_size) * 10e8
                distances += identity_matrix

            # Find the minimum distance and corresponding indices for the A batch
            min_batch_indices = np.argmin(distances, axis=1) + j
            min_batch_distances = np.min(distances, axis=1)

            # Update the minimum distances and indices if the current batch distances are smaller
            update_mask = min_batch_distances < min_distances[i:end_i]
            min_distances[i:end_i][update_mask] = min_batch_distances[update_mask]
            min_indices[i:end_i][update_mask] = min_batch_indices[update_mask]

    return min_indices


def find_match(source, target, return_index: bool = False):
    a = np.sum(target**2, axis=1).reshape(target.shape[0], 1) + np.sum(
        source.T**2, axis=0
    )
    b = np.dot(target, source.T) * 2
    distance_matrix = a - b
    return (
        np.argmin(distance_matrix, axis=0)
        if return_index
        else np.min(distance_matrix, axis=0)
    )


def find_match_self(source, target, return_index: bool = False):
    a = np.sum(target**2, axis=1).reshape(target.shape[0], 1) + np.sum(
        source.T**2, axis=0
    )
    b = np.dot(target, source.T) * 2
    distance_matrix = a - b
    n_col = np.shape(distance_matrix)[1]

    if return_index:
        min_indices = np.zeros(n_col, dtype=int)
        for i in range(n_col):
            sorted_indices = np.argsort(distance_matrix[:, i])
            min_indices[i] = sorted_indices[1]  # Get index of second smallest value
        return min_indices
    else:
        min_distance = np.zeros(n_col)
        for i in range(n_col):
            sorted_column = np.sort(distance_matrix[:, i])
            min_distance[i] = sorted_column[1]
        return min_distance
