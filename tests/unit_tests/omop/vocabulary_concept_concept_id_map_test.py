import polars as pl
import pytest

from cehrgpt.omop.vocab_utils import generate_vocabulary_concept_to_concept_id_map


def test_generate_vocabulary_concept_to_concept_id_map():
    # Sample input DataFrame
    concept_pl = pl.DataFrame(
        {
            "vocabulary_id": ["ICD10", "RXNORM", "CPT", "ICD10"],
            "concept_code": ["A00", "12345", "99213", "B01"],
            "concept_id": [1001, 2002, 3003, 1004],
        }
    )

    # Expected output
    expected_output = {
        "ICD10//A00": 1001,
        "RXNORM//12345": 2002,
        "CPT//99213": 3003,
        "ICD10//B01": 1004,
    }

    # Run function
    result = generate_vocabulary_concept_to_concept_id_map(concept_pl)

    # Assertions
    assert isinstance(result, dict), "Output should be a dictionary"
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_vocabulary_concept_to_concept_id_map_empty():
    # Empty DataFrame case
    concept_pl = pl.DataFrame(
        {"vocabulary_id": [], "concept_code": [], "concept_id": []}
    )

    # Run function
    result = generate_vocabulary_concept_to_concept_id_map(concept_pl)

    # Assertions
    assert result == {}, "Expected empty dictionary for empty input"


def test_generate_vocabulary_concept_to_concept_id_map_duplicates():
    # Duplicate concept codes but different vocabularies
    concept_pl = pl.DataFrame(
        {
            "vocabulary_id": ["ICD10", "ICD10", "RXNORM"],
            "concept_code": ["A00", "A00", "12345"],
            "concept_id": [1111, 2222, 3333],
        }
    )

    # Expected output
    expected_output = {
        "ICD10//A00": 2222,  # Should keep the last occurrence
        "RXNORM//12345": 3333,
    }

    # Run function
    result = generate_vocabulary_concept_to_concept_id_map(concept_pl)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()
