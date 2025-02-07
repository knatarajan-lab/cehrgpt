import unittest
from datetime import datetime
from unittest.mock import MagicMock

import networkx as nx

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptEvent
from cehrgpt.generation.cehrgpt_patient.clinical_statement_generator import (
    ClinicalStatementGenerator,
    ConditionDrugKnowledgeGraph,
)


class TestClinicalStatementGenerator(unittest.TestCase):
    def setUp(self):
        # Mocking the knowledge graph
        self.knowledge_graph = nx.MultiDiGraph()
        self.knowledge_graph.add_edge(
            1, 2, edge_key=0, **{"rel_name": "inv_has_indication"}
        )
        self.drug_map = {2: [200, 201]}
        self.condition_drug_knowledge_graph = ConditionDrugKnowledgeGraph(
            knowledge_graph=self.knowledge_graph,
            drug_ingredient_to_brand_drug_map=self.drug_map,
        )
        self.generator = ClinicalStatementGenerator(
            condition_drug_knowledge_graph=self.condition_drug_knowledge_graph,
            allowed_conditions=[1, 3],
            n_conditions=1,
        )

    def test_get_indications(self):
        # Test to ensure caching and retrieval of indications
        indications = self.generator.get_indications(1)
        self.assertEqual(indications, [2, 200, 201])
        # Check if repeated calls use the cache
        indications = self.generator.get_indications(1)
        self.assertEqual(indications, [2, 200, 201])

    def test_is_allowed_condition(self):
        event_allowed = CehrGptEvent(
            domain="condition", code="1", time=datetime(year=2020, month=1, day=1)
        )
        event_not_allowed = CehrGptEvent(
            domain="condition", code="2", time=datetime(year=2020, month=1, day=1)
        )
        self.assertTrue(self.generator.is_allowed_condition(event_allowed))
        self.assertFalse(self.generator.is_allowed_condition(event_not_allowed))

    def test_generate_clinical_statement_no_conditions(self):
        # Testing with no conditions present
        self.generator.get_indications = MagicMock(return_value=[])
        clinical_statement = self.generator.generate_clinical_statement(
            concept_ids=["1"],
            concept_name_mapping={"1": "Diabetes"},
            concept_domain_mapping={"1": "condition"},
        )
        self.assertIsNone(clinical_statement)

    def test_generate_clinical_statement_with_data(self):
        # Testing generation of clinical statement with mock data
        self.generator.get_indications = MagicMock(return_value=[200])
        clinical_statement = self.generator.generate_clinical_statement(
            concept_ids=[
                "year:2021",
                "age:50",
                "8532",
                "8527",
                "[VS]",
                "9202",
                "1",
                "200",
                "[VE]",
            ],
            concept_name_mapping={
                "1": "Diabetes",
                "200": "Test Drug",
                "9202": "Outpatient visit",
                "8532": "FEMALE",
                "8527": "White",
            },
            concept_domain_mapping={"1": "condition", "9202": "visit", "200": "drug"},
        )
        self.assertIn("Race", clinical_statement)
        self.assertIn("Gender", clinical_statement)
        self.assertIn("Diagnosis age", clinical_statement)
        self.assertIn("Condition", clinical_statement)
        self.assertIn("Drug", clinical_statement)


if __name__ == "__main__":
    unittest.main()
