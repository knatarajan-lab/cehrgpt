import unittest

import polars as pl

from cehrgpt.open_vocab import question_universe as qu

# A small hierarchy (all SNOMED unless said otherwise):
#
#   100 Root
#   +-- 110 Hub (not a category)
#   |   +-- 120 Heart disease (category)
#   |   |   +-- 130 Coronary disease
#   |   |   |   +-- 140 Angina (token)
#   |   |   +-- 150 Arrhythmia (token)
#   |   +-- 160 Lung disease (category)
#   |       +-- 150 Arrhythmia (second parent, to test two categories)
#   +-- 170 Orphan group
#       +-- 180 Orphan disease (token, below no category)
#   901 CPT4 procedure (token, not SNOMED) -> SNOMED parent 130
NAMES = {
    100: ("Root", "SNOMED", "Condition"),
    110: ("Hub", "SNOMED", "Condition"),
    120: ("Heart disease", "SNOMED", "Condition"),
    130: ("Coronary disease", "SNOMED", "Condition"),
    140: ("Angina", "SNOMED", "Condition"),
    150: ("Arrhythmia", "SNOMED", "Condition"),
    160: ("Lung disease", "SNOMED", "Condition"),
    170: ("Orphan group", "SNOMED", "Condition"),
    180: ("Orphan disease", "SNOMED", "Condition"),
    901: ("Coronary angiography", "CPT4", "Procedure"),
    # drugs: ATC class > ingredient > clinical drug form > clinical drug (token)
    300: ("Antidiabetic drugs", "ATC", "Drug"),
    310: ("Metformin", "RxNorm", "Drug"),
    320: ("Metformin Oral Tablet", "RxNorm", "Drug"),
    330: ("Metformin 500 MG Oral Tablet", "RxNorm", "Drug"),
    340: ("Aspirin", "RxNorm Extension", "Drug"),
    350: ("Aspirin 81 MG Oral Tablet", "RxNorm Extension", "Drug"),
    360: ("Metformin / Aspirin Pack", "RxNorm", "Drug"),
    370: ("Unmapped vaccine", "CVX", "Drug"),
    # visits: a visit type (category) > place of service / clinic, which are in other vocabularies
    500: ("Inpatient Visit", "Visit", "Visit"),
    510: ("Inpatient Hospital", "CMS Place of Service", "Visit"),
    520: ("Outpatient Visit", "Visit", "Visit"),
    530: ("Clinic/Center", "NUCC", "Visit"),
    540: ("Radiology Clinic/Center", "NUCC", "Visit"),
    550: ("Home", "CMS Place of Service", "Visit"),
}
CLASSES = {310: "Ingredient", 340: "Ingredient", 320: "Clinical Drug Form", 330: "Clinical Drug",
           350: "Clinical Drug", 360: "Clinical Pack", 300: "ATC 4th"}
# parent -> children
EDGES = {100: [110, 170], 110: [120, 160], 120: [130, 150], 130: [140], 160: [150], 170: [180], 130: [140]}
CPT_PARENT = {901: [130], 320: [310], 330: [320], 350: [340], 360: [330, 350]}
EDGES[300] = [310]
EDGES.update({500: [510], 520: [530], 530: [540]})


def tables():
    concept = pl.DataFrame(
        {
            "concept_id": list(NAMES),
            "concept_name": [v[0] for v in NAMES.values()],
            "vocabulary_id": [v[1] for v in NAMES.values()],
            "domain_id": [v[2] for v in NAMES.values()],
            "concept_class_id": [CLASSES.get(c, "Clinical Finding") for c in NAMES],
        }
    )
    parents = {}
    for parent, children in EDGES.items():
        for child in children:
            parents.setdefault(child, set()).add(parent)
    for child, ps in CPT_PARENT.items():
        parents.setdefault(child, set()).update(ps)

    def ancestors(node):
        found = {}
        stack = [(p, 1) for p in parents.get(node, ())]
        while stack:
            p, d = stack.pop()
            if p not in found or d < found[p]:
                found[p] = d
                stack.extend((pp, d + 1) for pp in parents.get(p, ()))
        return found

    rows = [(node, node, 0) for node in NAMES]
    for node in NAMES:
        rows += [(a, node, d) for a, d in ancestors(node).items()]
    ancestor = pl.DataFrame(rows, schema=["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"], orient="row")
    return concept, ancestor


CATEGORIES = ((120, "Heart disease"), (160, "Lung disease"))


class BuildUniverseTest(unittest.TestCase):
    def universe(self, tokens, domain="Condition", categories=CATEGORIES):
        concept, ancestor = tables()
        result = qu.build_universe(concept, ancestor, tokens, domain, categories)
        return {row["concept_id"]: row for row in result.iter_rows(named=True)}

    def test_ancestors_stop_at_the_categories(self):
        universe = self.universe([140])
        # the token, its ancestors that are a category or below one
        self.assertEqual(set(universe), {140, 130, 120})
        self.assertNotIn(110, universe)  # the hub above the category
        self.assertNotIn(100, universe)

    def test_is_token_and_num_tokens(self):
        universe = self.universe([140, 150])
        self.assertTrue(universe[140]["is_token"])
        self.assertFalse(universe[130]["is_token"])
        self.assertEqual(universe[120]["num_tokens"], 2)  # Angina and Arrhythmia
        self.assertEqual(universe[130]["num_tokens"], 1)  # Angina only
        self.assertEqual(universe[140]["num_tokens"], 1)

    def test_a_concept_below_two_categories_lists_both(self):
        universe = self.universe([150])
        self.assertEqual(universe[150]["categories"], [120, 160])
        self.assertEqual(set(universe), {150, 120, 160})

    def test_the_depth_is_below_the_closest_category(self):
        universe = self.universe([140])
        self.assertEqual(universe[120]["depth"], 0)
        self.assertEqual(universe[130]["depth"], 1)
        self.assertEqual(universe[140]["depth"], 2)

    def test_a_token_below_no_category_is_only_itself(self):
        universe = self.universe([180])
        self.assertEqual(set(universe), {180})
        self.assertEqual(universe[180]["categories"], [])
        self.assertIsNone(universe[180]["depth"])

    def test_a_token_outside_snomed_takes_the_categories_of_its_snomed_ancestors(self):
        universe = self.universe([901], domain="Procedure")
        self.assertEqual(universe[901]["categories"], [120])
        self.assertTrue(universe[901]["is_token"])
        self.assertEqual({130, 120} - set(universe), set())  # its SNOMED ancestors are questions
        self.assertNotIn(110, universe)

    def test_only_the_tokens_of_the_domain_are_used(self):
        # 901 is a Procedure, so asking for conditions with it as a token gives nothing for it
        self.assertEqual(self.universe([901], domain="Condition"), {})

    def test_a_category_that_does_not_match_the_vocabulary_fails(self):
        concept, ancestor = tables()
        with self.assertRaisesRegex(ValueError, "vocabulary version"):
            qu.build_universe(concept, ancestor, [140], "Condition", ((120, "Cardiac disease"),))
        with self.assertRaisesRegex(ValueError, "vocabulary version"):
            qu.build_universe(concept, ancestor, [140], "Condition", ((999999, "Heart disease"),))


VISIT_TEST_CATEGORIES = ((500, "Inpatient Visit"), (520, "Outpatient Visit"))


class VisitUniverseTest(unittest.TestCase):
    def universe(self, tokens):
        concept, ancestor = tables()
        # the vocabularies are the defaults of the Visit domain
        result = qu.build_universe(concept, ancestor, tokens, "Visit", VISIT_TEST_CATEGORIES)
        return {row["concept_id"]: row for row in result.iter_rows(named=True)}

    def test_a_place_of_service_rolls_up_to_its_visit_type_across_vocabularies(self):
        universe = self.universe([510])
        # Inpatient Hospital (CMS Place of Service) is below Inpatient Visit (Visit)
        self.assertEqual(set(universe), {510, 500})
        self.assertEqual(universe[510]["categories"], [500])
        self.assertEqual(universe[500]["num_tokens"], 1)
        self.assertFalse(universe[500]["is_token"])

    def test_the_levels_between_a_token_and_its_visit_type_are_questions(self):
        universe = self.universe([540])
        # Radiology Clinic/Center > Clinic/Center (not a token) > Outpatient Visit
        self.assertEqual(set(universe), {540, 530, 520})
        self.assertEqual({k: universe[k]["depth"] for k in universe}, {540: 2, 530: 1, 520: 0})
        self.assertEqual(universe[520]["num_tokens"], 1)

    def test_a_visit_below_no_type_is_a_question_on_its_own(self):
        universe = self.universe([550])
        self.assertEqual(set(universe), {550})
        self.assertEqual(universe[550]["categories"], [])

    def test_the_tokens_under_a_type_are_counted(self):
        universe = self.universe([510, 540, 550])
        self.assertEqual(universe[500]["num_tokens"], 1)
        self.assertEqual(universe[520]["num_tokens"], 1)
        self.assertEqual(set(universe), {500, 510, 520, 530, 540, 550})

    def test_other_domains_ignore_the_visit_tokens(self):
        concept, ancestor = tables()
        result = qu.build_universe(concept, ancestor, [510, 540], "Condition", CATEGORIES)
        self.assertEqual(len(result), 0)

    def test_the_visit_categories_are_configured_for_the_domain(self):
        self.assertEqual(qu.CATEGORIES["Visit"], qu.VISIT_CATEGORIES)
        self.assertIn("CMS Place of Service", qu.VOCABULARIES["Visit"])
        self.assertIsNone(qu.ANCESTOR_CLASSES["Visit"])


class DrugUniverseTest(unittest.TestCase):
    def universe(self, tokens):
        concept, ancestor = tables()
        result = qu.build_universe(concept, ancestor, tokens, "Drug")
        return {row["concept_id"]: row for row in result.iter_rows(named=True)}

    def test_a_drug_is_mapped_to_its_ingredient_only(self):
        universe = self.universe([330])
        # the drug and its ingredient: not the form in between, not the ATC class above
        self.assertEqual(set(universe), {330, 310})
        self.assertFalse(universe[310]["is_token"])
        self.assertEqual(universe[310]["num_tokens"], 1)

    def test_a_pack_has_all_of_its_ingredients(self):
        universe = self.universe([360])
        self.assertEqual(set(universe), {360, 310, 340})
        self.assertEqual(universe[360]["categories"], [310, 340])

    def test_rxnorm_extension_ingredients_count_and_other_vocabularies_do_not(self):
        universe = self.universe([350, 370])
        self.assertIn(340, universe)
        self.assertEqual(universe[370]["categories"], [])  # a vaccine of another vocabulary: itself only

    def test_an_ingredient_that_is_a_token_is_a_question_once(self):
        universe = self.universe([310, 330])
        self.assertTrue(universe[310]["is_token"])
        self.assertEqual(universe[310]["num_tokens"], 2)


class HardcodedCategoriesTest(unittest.TestCase):
    def test_the_lists_have_no_duplicate_ids_and_cover_the_domains(self):
        for domain, categories in qu.CATEGORIES.items():
            if categories is None:  # the drugs: the ingredients of the concept table
                continue
            ids = [concept_id for concept_id, _ in categories]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate ids in the {domain} categories")
            self.assertTrue(categories)
        names = [name for _, name in qu.CATEGORIES["Procedure"]]
        self.assertIn("Endoscopy", names)
        self.assertNotIn("Surgical procedure", names)  # too generic, left out on purpose

    def test_hubs_are_not_categories(self):
        names = {name for categories in qu.CATEGORIES.values() if categories for _, name in categories}
        for hub in ("Disorder by body site", "Disorder of body system", "Procedure by method", "Procedure on body system"):
            self.assertNotIn(hub, names)


if __name__ == "__main__":
    unittest.main()
