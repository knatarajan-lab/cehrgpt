from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import polars as pl


def generate_ancestor_descendant_map(
    concept_ancestor_pl: pl.DataFrame,
    concept_ids: Iterable[str],
) -> Dict[str, List[str]]:
    concept_ids = [int(c) for c in concept_ids if c.isnumeric()]
    ancestor_descendant_map = defaultdict(list)
    for row in (
        concept_ancestor_pl.filter(pl.col("ancestor_concept_id").is_in(concept_ids))
        .to_pandas()
        .itertuples(index=False)
    ):
        ancestor_descendant_map[str(row.ancestor_concept_id)].append(
            str(row.descendant_concept_id)
        )
    return ancestor_descendant_map


def generate_concept_maps(
    concept_pl: pl.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate mappings from concept IDs to concept names and domain IDs using a Polars DataFrame.

    This function takes a Polars DataFrame containing at least the columns 'concept_id', 'concept_name', and 'domain_id'.
    It returns two dictionaries: one mapping concept IDs to their corresponding names and another mapping concept IDs to their domain IDs.

    Parameters:
    - concept_pl (pl.DataFrame): A Polars DataFrame with columns 'concept_id', 'concept_name', and 'domain_id'.
        'concept_id' should be unique identifiers for the concepts.
        'concept_name' is the descriptive name associated with each concept ID.
        'domain_id' refers to the domain classification of each concept ID.

    Returns:
    - Tuple[Dict[str, str], Dict[str, str]]: A tuple of two dictionaries. The first dictionary maps concept IDs to concept names,
      and the second maps concept IDs to domain IDs.

    Example:
    --------
    >>> concept_pl = pl.DataFrame({
    ...     "concept_id": [1, 2, 3],
    ...     "concept_name": ["Name1", "Name2", "Name3"],
    ...     "domain_id": ["Domain1", "Domain2", "Domain3"]
    ... })
    >>> concept_map, concept_domain = generate_concept_maps(concept_pl)
    >>> print(concept_map)
    {'1': 'Name1', '2': 'Name2', '3': 'Name3'}
    >>> print(concept_domain)
    {'1': 'Domain1', '2': 'Domain2', '3': 'Domain3'}
    """
    # Convert DataFrame columns to dictionaries
    concept_map = concept_pl.select(
        [pl.col("concept_id").cast(str), pl.col("concept_name")]
    ).to_dict(as_series=False)

    concept_domain = concept_pl.select(
        [pl.col("concept_id").cast(str), pl.col("domain_id")]
    ).to_dict(as_series=False)

    # Convert list of values to single dictionary per column pair
    concept_map = dict(zip(concept_map["concept_id"], concept_map["concept_name"]))
    concept_domain = dict(
        zip(concept_domain["concept_id"], concept_domain["domain_id"])
    )

    return concept_map, concept_domain


def create_drug_ingredient_to_brand_drug_map(
    concept: pl.DataFrame, concept_ancestor: pl.DataFrame
) -> Dict[int, List[int]]:
    drug_ingredient = concept.filter(
        (pl.col("domain_id") == "Drug") & (pl.col("concept_class_id") == "Ingredient")
    ).select(pl.col("concept_id").alias("ingredient_concept_id"))
    # Join with concept_ancestor where drug_concept_id matches ancestor_concept_id
    ingredient_drug_map = drug_ingredient.join(
        concept_ancestor,
        left_on="ingredient_concept_id",
        right_on="ancestor_concept_id",
    ).select(
        pl.col("ingredient_concept_id"),
        pl.col("descendant_concept_id").alias("drug_concept_id"),
    )
    drug_ingredient_to_brand_drug_map = defaultdict(list)
    for row in ingredient_drug_map.to_pandas().itertuples(index=False):
        ingredient_id = row.ingredient_concept_id
        drug_id = row.drug_concept_id
        drug_ingredient_to_brand_drug_map[ingredient_id].append(drug_id)
    return drug_ingredient_to_brand_drug_map
