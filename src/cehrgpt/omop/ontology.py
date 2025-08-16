from __future__ import annotations

import collections
import os
from typing import Any, Dict, Iterable, Optional, Set, Union

import polars as pl
from datasets import Dataset


# Adapted from femr.ontology
def _get_all_codes_map(batch: Dict[str, Any]) -> Dict[str, Any]:
    result = set()
    for concept_ids in batch["concept_ids"]:
        for concept_id in concept_ids:
            if concept_id.isnumeric():
                result.add(concept_id)
    return {"unique_concept_ids": list(result)}


class Ontology:
    def __init__(self, vocab_path: str):
        """Create an Ontology from an Athena download and an optional meds Code Metadata structure.

        NOTE: This is an expensive operation.
        It is recommended to create an ontology once and then save/load it as necessary.
        """
        # Load from code metadata
        self.parents_map: Dict[str, Set[str]] = collections.defaultdict(set)
        self.concept_vocabulary_map: Dict[str, str] = collections.defaultdict(str)
        self.concept_domain_map: Dict[str, str] = collections.defaultdict(str)

        # Load from the athena path ...
        concept = pl.scan_parquet(os.path.join(vocab_path, "concept/*parquet"))
        vocabulary_id_col = pl.col("vocabulary_id")
        concept_id_col = pl.col("concept_id").cast(pl.String)
        domain_id_col = pl.col("domain_id").cast(pl.String)

        processed_concepts = (
            concept.select(
                concept_id_col,
                domain_id_col,
                vocabulary_id_col,
                pl.col("standard_concept").is_null(),
            )
            .collect()
            .rows()
        )

        non_standard_concepts = set()

        for concept_id, domain_id, vocabulary_id, is_non_standard in processed_concepts:
            # We don't want to override code metadata
            if concept_id not in self.concept_vocabulary_map:
                self.concept_vocabulary_map[concept_id] = vocabulary_id

            if concept_id not in self.concept_domain_map:
                self.concept_domain_map[concept_id] = domain_id
            # We don't want to override code metadata
            if is_non_standard:
                non_standard_concepts.add(concept_id)

        relationship = pl.scan_parquet(
            os.path.join(vocab_path, "concept_relationship/*parquet")
        )
        relationship_id = pl.col("relationship_id")
        relationship = relationship.filter(
            relationship_id == "Maps to",
            pl.col("concept_id_1") != pl.col("concept_id_2"),
        )
        for concept_id_1, concept_id_2 in (
            relationship.select(
                pl.col("concept_id_1").cast(pl.String),
                pl.col("concept_id_2").cast(pl.String),
            )
            .collect()
            .rows()
        ):
            if concept_id_1 in non_standard_concepts:
                self.parents_map[concept_id_1].add(concept_id_2)

        ancestor = pl.scan_parquet(
            os.path.join(vocab_path, "concept_ancestor/*parquet")
        )
        ancestor = ancestor.filter(pl.col("min_levels_of_separation") == 1)
        for concept_id, parent_concept_id in (
            ancestor.select(
                pl.col("descendant_concept_id").cast(pl.String),
                pl.col("ancestor_concept_id").cast(pl.String),
            )
            .collect()
            .rows()
        ):
            self.parents_map[concept_id].add(parent_concept_id)
        self.all_parents_map: Dict[str, Set[str]] = {}

    def get_domain(self, concept_id: Union[str, int]) -> Optional[str]:
        return self.concept_domain_map.get(str(concept_id), None)

    def prune_to_dataset(
        self,
        dataset: Dataset,
        remove_ontologies: Set[str] = set(),
        num_proc: int = 4,
        batch_size: int = 1024,
    ) -> None:
        mapped_dataset = dataset.map(
            _get_all_codes_map,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )
        valid_concept_ids = set(mapped_dataset["unique_concept_ids"])
        all_parents = set()
        for concept_id in valid_concept_ids:
            all_parents |= self.get_all_parents(concept_id)

        def is_valid(c: str):
            ontology = self.concept_vocabulary_map.get(c, "")
            return (c in valid_concept_ids) or (
                (ontology not in remove_ontologies) and (c in all_parents)
            )

        concept_ids = set(self.parents_map.keys())
        for concept_id in concept_ids:
            m: Any
            if is_valid(concept_id):
                for m in (self.parents_map, self.concept_vocabulary_map):
                    m[concept_id] = {a for a in m[concept_id] if is_valid(a)}
            else:
                for m in (self.parents_map, self.concept_vocabulary_map):
                    if concept_id in m:
                        del m[concept_id]

        self.all_parents_map = {}

        # Prime the pump
        for concept_id in self.parents_map.keys():
            self.get_all_parents(concept_id)

    def get_parents(self, code: str) -> Iterable[str]:
        """Get the parents for a given code."""
        return self.parents_map.get(code, set())

    def get_all_parents(self, code: str) -> Set[str]:
        """Get all parents, including through the ontology."""
        if code not in self.all_parents_map:
            result = {code}
            for parent in self.parents_map.get(code, set()):
                result |= self.get_all_parents(parent)
            self.all_parents_map[code] = result

        return self.all_parents_map[code]
