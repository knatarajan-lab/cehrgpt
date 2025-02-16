import polars as pl
import pytest

from cehrgpt.omop.vocab_utils import generate_to_standard_concept_id_map


def test_generate_to_standard_concept_id_map():
    # Sample input DataFrame
    concept_relationship = pl.DataFrame(
        {
            "concept_id_1": [1001, 1001, 1002, 1003, 1003, 1003],
            "concept_id_2": [2001, 2002, 2003, 3001, 3002, 3003],
            "relationship_id": [
                "Maps to",
                "Maps to",
                "Maps to",
                "Maps to",
                "Maps to",
                "Maps to",
            ],
        }
    )

    # Expected output
    expected_output = {1001: [2001, 2002], 1002: [2003], 1003: [3001, 3002, 3003]}

    # Run function
    result = generate_to_standard_concept_id_map(concept_relationship)

    # Assertions
    assert isinstance(result, dict), "Output should be a dictionary"
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_to_standard_concept_id_map_empty():
    # Empty DataFrame test
    concept_relationship = pl.DataFrame(
        {"concept_id_1": [], "concept_id_2": [], "relationship_id": []}
    )

    # Run function
    result = generate_to_standard_concept_id_map(concept_relationship)

    # Assertions
    assert result == {}, "Expected empty dictionary for empty input"


def test_generate_to_standard_concept_id_map_single_entry():
    # Single entry test
    concept_relationship = pl.DataFrame(
        {"concept_id_1": [1001], "concept_id_2": [2001], "relationship_id": ["Maps to"]}
    )

    # Expected output
    expected_output = {1001: [2001]}

    # Run function
    result = generate_to_standard_concept_id_map(concept_relationship)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_to_standard_concept_id_map_duplicates():
    # Duplicate `concept_id_1` values test
    concept_relationship = pl.DataFrame(
        {
            "concept_id_1": [1001, 1001, 1001],
            "concept_id_2": [2001, 2002, 2003],
            "relationship_id": ["Maps to", "Maps to", "Maps to"],
        }
    )

    # Expected output
    expected_output = {1001: [2001, 2002, 2003]}

    # Run function
    result = generate_to_standard_concept_id_map(concept_relationship)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_to_standard_concept_id_map_unrelated_columns():
    # Unrelated column test (should ignore extra columns)
    concept_relationship = pl.DataFrame(
        {
            "concept_id_1": [1001, 1001, 1002],
            "concept_id_2": [2001, 2002, 2003],
            "relationship_id": ["Maps to", "Maps to", "Maps to"],
            "extra_column": ["A", "B", "C"],  # Extra column
        }
    )

    # Expected output
    expected_output = {1001: [2001, 2002], 1002: [2003]}

    # Run function
    result = generate_to_standard_concept_id_map(concept_relationship)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()
