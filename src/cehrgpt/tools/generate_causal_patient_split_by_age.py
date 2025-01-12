import numpy as np
import pandas as pd

# Define race mapping
race_mapping = {
    "38003613": "8557",
    "38003610": "8557",
    "38003579": "8515",
    "44814653": "0",
}

# Invalid age groups
invalid_age_groups = [
    "age:100-110",
    "age:110-120",
    "age:120-130",
    "age:130-140",
    "age:140-150",
    "age:150-160",
    "age:160-170",
    "age:170-180",
    "age:180-190",
    "age:190-200",
    "age:640-650",
    "age:680-690",
    "age:730-740",
    "age:740-750",
    "age:890-900",
    "age:900-910",
    "age:-10-0",
]


def age_group_func(age_str):
    """
    Categorize an age into a 10-year age group.

    Args:
        age_str (str): A string containing the age in the format "age:XX".

    Returns:
        str: A string representing the 10-year age group "age:XX-XX".
    """
    age = int(age_str.split(":")[1])
    group_number = age // 10
    return f"age:{group_number * 10}-{(group_number + 1) * 10}"


def map_race(race):
    return race_mapping.get(race, race)


def main(args):
    # Load data
    patient_sequence = pd.read_parquet(args.patient_sequence)
    # Extract and preprocess demographics
    demographics = patient_sequence.concept_ids.apply(
        lambda concept_ids: concept_ids[:4]
    )
    patient_sequence["demographics"] = demographics
    year = demographics.apply(lambda concepts: concepts[0])
    age = demographics.apply(lambda concepts: concepts[1]).apply(age_group_func)
    gender = demographics.apply(lambda concepts: concepts[2])
    race = demographics.apply(lambda concepts: concepts[3])
    death = patient_sequence.concept_ids.apply(
        lambda concept_ids: int(concept_ids[-2] == "[DEATH]")
    )

    patient_sequence["year"] = year
    patient_sequence["age"] = age
    patient_sequence["gender"] = gender
    patient_sequence["race"] = race
    patient_sequence["death"] = death

    demographics = patient_sequence[
        ["person_id", "death", "year", "age", "gender", "race", "split"]
    ]
    demographics["race"] = demographics.race.apply(map_race)

    demographics_clean = demographics[
        (demographics.gender != "0") & (~demographics.age.isin(invalid_age_groups))
    ]
    patient_sequence_clean = patient_sequence[
        patient_sequence.person_id.isin(demographics_clean.person_id)
    ]

    # Calculate probabilities
    probs = (
        demographics_clean.groupby(["age"])["person_id"].count()
        / len(demographics_clean)
    ).reset_index()
    probs.rename(columns={"person_id": "prob"}, inplace=True)

    # Adjust probabilities
    np.random.seed(42)
    x = np.asarray(list(reversed(range(1, 11))))
    adjusted_probs = probs.prob * pd.Series(x)
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    probs["adjusted_prob"] = adjusted_probs

    demographics_for_sampling = patient_sequence_clean[
        ["year", "age", "race", "gender", "person_id"]
    ].merge(probs, on="age")
    demographics_for_sampling["adjusted_prob"] = (
        demographics_for_sampling.adjusted_prob
        / demographics_for_sampling.adjusted_prob.sum()
    )

    # Train/Validation Split
    causal_train_split = demographics_for_sampling.sample(
        args.num_patients, replace=False, weights="adjusted_prob", random_state=1
    )
    causal_train_split["split"] = "train"
    causal_val_split = demographics_for_sampling[
        ~demographics_for_sampling.person_id.isin(causal_train_split.person_id)
    ]
    causal_val_split["split"] = "validation"

    causal_train_val_split = pd.concat([causal_train_split, causal_val_split])

    # Save outputs
    causal_train_val_split.to_parquet(args.output_folder, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for a causal patient split by age groups"
    )
    parser.add_argument(
        "--patient_sequence",
        required=True,
    )
    parser.add_argument(
        "--num_patients",
        default=1_000_000,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--output_folder",
        required=True,
    )
    # Call the main function with parsed arguments
    main(parser.parse_args())
