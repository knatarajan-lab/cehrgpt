import random
from collections import defaultdict
from typing import Dict, List, Optional

import networkx

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptEvent
from cehrgpt.generation.cehrgpt_patient.patient_narrative_converter import (
    PatientSequenceConverter,
)
from cehrgpt.generation.cehrgpt_patient.typed_tokens import translate_to_cehrgpt_tokens


class ConditionDrugKnowledgeGraph:
    def __init__(
        self,
        knowledge_graph: networkx.Graph,
        drug_ingredient_to_brand_drug_map: Dict[int, List[int]],
    ):
        self.knowledge_graph = knowledge_graph
        self.drug_ingredient_to_brand_drug_map = drug_ingredient_to_brand_drug_map

    def get_drug_indications(self, condition_concept_id: int) -> List[int]:
        drug_concept_ids = []
        for dest_concept_id, rel in self.knowledge_graph[condition_concept_id].items():
            if rel[0]["rel_name"] == "inv_has_indication":
                drug_concept_ids.append(dest_concept_id)
                drug_concept_ids.extend(
                    self.drug_ingredient_to_brand_drug_map.get(dest_concept_id, [])
                )
        return drug_concept_ids


class ClinicalStatementGenerator:
    def __init__(
        self,
        condition_drug_knowledge_graph: ConditionDrugKnowledgeGraph,
        allowed_conditions: Optional[List[int]] = None,
        n_conditions: int = 1,
    ):
        self.allowed_conditions = allowed_conditions
        self.n_conditions = n_conditions
        self.condition_drug_map = defaultdict(list)
        self.condition_drug_knowledge_graph = condition_drug_knowledge_graph

    def get_indications(self, condition_concept_id: int) -> List[int]:
        if condition_concept_id not in self.condition_drug_map:
            indications = self.condition_drug_knowledge_graph.get_drug_indications(
                condition_concept_id
            )
            self.condition_drug_map[condition_concept_id] = indications
        return self.condition_drug_map[condition_concept_id]

    def is_allowed_condition(self, event: CehrGptEvent) -> bool:
        if event.domain.lower().startswith("condition"):
            if self.allowed_conditions is not None:
                return int(event.code) in self.allowed_conditions
            return True
        return False

    def generate_clinical_statement(
        self,
        concept_ids: List[str],
        concept_name_mapping: Dict[str, str],
        concept_domain_mapping: Dict[str, str],
    ) -> Optional[str]:
        clinical_statement = None
        cehrgpt_tokens = translate_to_cehrgpt_tokens(
            concept_ids=concept_ids,
            concept_domain_mapping=concept_domain_mapping,
        )
        patient_sequence_converter = PatientSequenceConverter(tokens=cehrgpt_tokens)
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

            age_condition_drug_tuples = list()
            for condition, age_at_diagnosis in random.sample(
                conditions, self.n_conditions
            ):
                indications = [
                    indication
                    for indication in self.get_indications(condition)
                    if indication in drugs
                ]
                random_indication = random.choice(indications) if indications else None
                age_condition_drug_tuples.append(
                    (age_at_diagnosis, condition, random_indication)
                )

            clinical_statement = f"Race: {cehrgpt_patient.race}\n"
            clinical_statement += f"Gender: {cehrgpt_patient.gender}\n"
            for i, (age, condition, drug) in enumerate(
                sorted(age_condition_drug_tuples, key=lambda x: x[0])
            ):
                clinical_statement += f"\n{i + 1}. Diagnosis age {age}\n"
                clinical_statement += f"{i + 1}. Condition: {condition}\n"
                if drug:
                    clinical_statement += f"{i + 1}. Drug: {drug}\n"
        return clinical_statement
