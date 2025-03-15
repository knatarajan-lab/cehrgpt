import polars as pl
import pytest

from cehrgpt.omop.vocab_utils import generate_vocabulary_name_to_concept_id_map


def test_generate_vocabulary_name_to_concept_id_map():
    # Sample input DataFrame
    concept_pl = pl.DataFrame(
        {
            "vocabulary_id": ["ICD10", "RXNORM", "CPT", "ICD10"],
            "concept_name": ["Diabetes", "Metformin", "ProcedureX", "Hypertension"],
            "concept_id": [1001, 2002, 3003, 1004],
        }
    )

    # Expected output
    expected_output = {
        "icd10//diabetes": 1001,
        "rxnorm//metformin": 2002,
        "cpt//procedurex": 3003,
        "icd10//hypertension": 1004,
    }

    # Run function
    result = generate_vocabulary_name_to_concept_id_map(concept_pl)

    # Assertions
    assert isinstance(result, dict), "Output should be a dictionary"
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_vocabulary_name_to_concept_id_map_empty():
    # Empty DataFrame test
    concept_pl = pl.DataFrame(
        {"vocabulary_id": [], "concept_name": [], "concept_id": []}
    )

    # Run function
    result = generate_vocabulary_name_to_concept_id_map(concept_pl)

    # Assertions
    assert result == {}, "Expected empty dictionary for empty input"


def test_generate_vocabulary_name_to_concept_id_map_case_insensitive():
    # Test case-insensitive behavior
    concept_pl = pl.DataFrame(
        {
            "vocabulary_id": ["ICD10", "ICD10"],
            "concept_name": ["Diabetes", "diabetes"],
            "concept_id": [1001, 1002],  # The last one should override
        }
    )

    # Expected output (last occurrence should be used)
    expected_output = {"icd10//diabetes": 1002}

    # Run function
    result = generate_vocabulary_name_to_concept_id_map(concept_pl)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_vocabulary_name_to_concept_id_map_special_characters():
    # Handles special characters and spaces correctly
    concept_pl = pl.DataFrame(
        {
            "vocabulary_id": ["ICD10", "ICD10"],
            "concept_name": ["Type 2 Diabetes", "Type-2-Diabetes"],
            "concept_id": [1101, 1102],
        }
    )

    # Expected output
    expected_output = {"icd10//type 2 diabetes": 1101, "icd10//type-2-diabetes": 1102}

    # Run function
    result = generate_vocabulary_name_to_concept_id_map(concept_pl)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_vocabulary_name_to_concept_id_map_duplicates():
    # Duplicate entries: the last occurrence should override previous ones
    concept_pl = pl.DataFrame(
        {
            "vocabulary_id": ["ICD10", "ICD10"],
            "concept_name": ["Diabetes", "Diabetes"],
            "concept_id": [1001, 2002],  # The last occurrence should override
        }
    )

    # Expected output
    expected_output = {"icd10//diabetes": 2002}

    # Run function
    result = generate_vocabulary_name_to_concept_id_map(concept_pl)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()
