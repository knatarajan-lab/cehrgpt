import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
from transformers.utils import logging

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptEvent
from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    PatientSequenceConverter,
    get_cehrgpt_patient_converter,
)
from cehrgpt.gpt_utils import ProbabilisticCache

logger = logging.get_logger("transformers")

DEFAULT_CLINICAL_STATEMENT = "Generate a patient"


class ConditionDrugKnowledgeGraph:
    def __init__(
        self,
        knowledge_graph: nx.Graph,
        drug_ingredient_to_brand_drug_map: Dict[int, List[int]],
    ):
        self.knowledge_graph = knowledge_graph
        self.drug_ingredient_to_brand_drug_map = drug_ingredient_to_brand_drug_map

    def get_drug_indications(self, condition_concept_id: int) -> Dict[int, int]:
        # Use set for better performance on lookups and additions
        drug_concept_ids = {}
        if condition_concept_id in self.knowledge_graph:
            for dest_concept_id, rels in self.knowledge_graph[
                condition_concept_id
            ].items():
                # Assumption: rels are dicts with potential multiple relationships
                for rel in rels.values():
                    if rel["rel_name"] == "inv_has_indication":
                        drug_concept_ids[dest_concept_id] = dest_concept_id
                        for branded_drug in self.drug_ingredient_to_brand_drug_map.get(
                            dest_concept_id, []
                        ):
                            drug_concept_ids[branded_drug] = dest_concept_id
        return drug_concept_ids


class ClinicalStatementGenerator:
    def __init__(
        self,
        condition_drug_knowledge_graph: ConditionDrugKnowledgeGraph,
        allowed_clinical_conditions: Optional[List[int]] = None,
        n_conditions: int = 1,
        capacity: int = 10000,
    ):
        self.allowed_clinical_conditions = allowed_clinical_conditions
        self.n_conditions = n_conditions
        self.condition_drug_map = defaultdict(dict)
        self.condition_drug_knowledge_graph = condition_drug_knowledge_graph
        self.cache = ProbabilisticCache[Tuple[int, int, int], PatientSequenceConverter](
            capacity
        )

    def get_indications(self, condition_concept_id: int) -> Dict[int, int]:
        if condition_concept_id not in self.condition_drug_map:
            indications = self.condition_drug_knowledge_graph.get_drug_indications(
                condition_concept_id
            )
            self.condition_drug_map[condition_concept_id] = indications
        return self.condition_drug_map[condition_concept_id]

    def is_allowed_condition(self, event: CehrGptEvent) -> bool:
        if event.domain.lower().startswith("condition"):
            if self.allowed_clinical_conditions is not None:
                return int(event.code) in self.allowed_clinical_conditions
            return True
        return False

    def generate_clinical_statement(
        self,
        concept_ids: List[str],
        concept_name_mapping: Dict[str, str],
        concept_domain_mapping: Dict[str, str],
        person_id: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        return_seed_concepts: bool = False,
    ) -> Optional[Union[str, Tuple[Optional[str], list[tuple[int, int, int]]]]]:
        clinical_statement = None
        patient_sequence_converter = self.cache.get_data((person_id, start, end))
        # If the element does not exist, we will generate it
        if not patient_sequence_converter:
            patient_sequence_converter = get_cehrgpt_patient_converter(
                concept_ids=concept_ids,
                concept_domain_mapping=concept_domain_mapping,
            )
            self.cache.add_data((person_id, start, end), patient_sequence_converter)

        age_condition_drug_tuples = list[Tuple[int, int, int]]()
        if patient_sequence_converter.is_validation_passed:
            cehrgpt_patient = patient_sequence_converter.get_patient(
                domain_map=concept_domain_mapping, concept_map=concept_name_mapping
            )
            conditions = []
            drugs = []
            birth_year = cehrgpt_patient.birth_datetime.year
            for event in cehrgpt_patient.get_events():
                if self.is_allowed_condition(event):
                    age_at_diagnosis = event.time.year - birth_year
                    conditions.append((int(event.code), age_at_diagnosis))
                elif event.domain.lower().startswith("drug"):
                    drugs.append(int(event.code))

            if conditions:
                for condition, age_at_diagnosis in random.sample(
                    conditions, self.n_conditions
                ):
                    indications = [
                        ingredient
                        for branded_drug, ingredient in self.get_indications(
                            condition
                        ).items()
                        if branded_drug in drugs
                    ]
                    random_indication = (
                        random.choice(indications) if indications else None
                    )
                    age_condition_drug_tuples.append(
                        (age_at_diagnosis, condition, random_indication)
                    )
                    logger.debug(
                        "Tuple[age, condition, drug]: %s, %s, %s",
                        age_at_diagnosis,
                        concept_name_mapping.get(str(condition), condition),
                        concept_name_mapping.get(
                            str(random_indication), random_indication
                        ),
                    )
            else:
                logger.debug(
                    "There are no conditions discovered\n.%s",
                    cehrgpt_patient.get_narrative(),
                )
            clinical_statement = f"Race: {cehrgpt_patient.race}\n"
            clinical_statement += f"Gender: {cehrgpt_patient.gender}\n"
            for i, (age, condition, drug) in enumerate(
                sorted(age_condition_drug_tuples, key=lambda x: x[0])
            ):
                if age:
                    clinical_statement += f"\n{i + 1}. Age: {age}\n"
                if condition:
                    clinical_statement += f"{i + 1}. Condition: {concept_name_mapping.get(str(condition), condition)}\n"
                if drug:
                    clinical_statement += (
                        f"{i + 1}. Drug: {concept_name_mapping.get(str(drug), drugs)}\n"
                    )
        else:
            logger.warning(
                "Failed to generate clinical statement, %s",
                patient_sequence_converter.get_error_messages(),
            )
        if return_seed_concepts:
            return clinical_statement, age_condition_drug_tuples
        else:
            return clinical_statement
