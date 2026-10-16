import os
import pickle
import random
import tempfile
import time
import unittest

from cehrgpt.qa.concept_hierarchy import ConceptHierarchy, query_concepts
from cehrgpt.qa.query_generation import (
    SECONDS_PER_DAY,
    PatientFuture,
    PatientTimeline,
    QueryConfig,
    QueryCoverage,
    generate_queries,
)

# 30 Clinical finding > 10 Diabetes mellitus > (11 Type 2, 12 Type 1); 30 > 20 Asthma
PARENTS = {"11": {"10"}, "12": {"10"}, "10": {"30"}, "20": {"30"}}
DOMAINS = {"11": "Condition", "12": "Condition", "10": "Condition", "20": "Condition", "30": "Condition", "99": "Visit"}
TOKENS = {"11", "12", "20"}
INDEX = 1_600_000_000.0


class FakeOntology:
    """Stands in for cehrgpt.omop.ontology.Ontology, which needs an Athena download."""

    parents_map = PARENTS
    concept_domain_map = DOMAINS


def day(n):
    return INDEX + n * SECONDS_PER_DAY


def timeline(events):
    return PatientTimeline([c for c, _ in events], [t for _, t in events], INDEX)


class ConceptHierarchyTest(unittest.TestCase):
    def setUp(self):
        self.hierarchy = ConceptHierarchy(PARENTS, DOMAINS)

    def test_ancestors_are_transitive_and_include_the_concept(self):
        self.assertEqual(self.hierarchy.ancestors("11"), {"11", "10", "30"})
        self.assertEqual(self.hierarchy.ancestors("30"), {"30"})
        self.assertEqual(self.hierarchy.ancestors("unknown"), {"unknown"})

    def test_without_a_hierarchy_a_concept_covers_only_itself(self):
        flat = ConceptHierarchy()
        self.assertFalse(flat.has_hierarchy)
        self.assertEqual(flat.ancestors("11"), {"11"})
        self.assertTrue(self.hierarchy.has_hierarchy)

    def test_cycles_do_not_hang(self):
        cyclic = ConceptHierarchy({"a": {"b"}, "b": {"a"}})
        self.assertEqual(cyclic.ancestors("a"), {"a", "b"})

    def test_descendants_and_subtree_sizes(self):
        self.assertEqual(self.hierarchy.subtree_sizes(TOKENS), {"11": 1, "12": 1, "10": 2, "30": 3, "20": 1})
        self.assertEqual(sorted(self.hierarchy.descendants_among("10", TOKENS)), ["11", "12"])

    def test_query_concepts_include_the_ancestors(self):
        concepts = query_concepts(TOKENS, self.hierarchy)
        self.assertEqual(set(concepts), {"11", "12", "10", "20", "30"})
        self.assertEqual(concepts["10"], 2)

    def test_broad_concepts_can_be_dropped(self):
        self.assertEqual(set(query_concepts(TOKENS, self.hierarchy, max_covered_tokens=2)), {"11", "12", "10", "20"})

    def test_domains_filter(self):
        self.assertEqual(query_concepts(TOKENS, self.hierarchy, allowed_domains=["Visit"]), {})
        self.assertEqual(len(query_concepts(TOKENS, self.hierarchy, allowed_domains=["Condition"])), 5)

    def test_loaded_from_an_ontology_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            # Without the file there is no hierarchy
            self.assertFalse(ConceptHierarchy.from_tokenizer_folder(folder).has_hierarchy)
            with open(os.path.join(folder, "ontology.pickle"), "wb") as f:
                pickle.dump(FakeOntology(), f)
            hierarchy = ConceptHierarchy.from_tokenizer_folder(folder)
        self.assertEqual(hierarchy.ancestors("12"), {"12", "10", "30"})
        self.assertEqual(hierarchy.domain("99"), "Visit")


class PatientFutureTest(unittest.TestCase):
    def setUp(self):
        self.hierarchy = ConceptHierarchy(PARENTS, DOMAINS)
        self.concepts = set(query_concepts(TOKENS, self.hierarchy))
        self.future = PatientFuture(
            timeline(
                [
                    ("[VS]", day(-50)),
                    ("12", day(-50)),
                    ("D3", day(-40)),
                    ("11", day(40)),
                    ("20", day(200)),
                    ("year:2020", day(210)),
                    ("12", day(400)),
                ]
            ),
            self.hierarchy,
            TOKENS,
            self.concepts,
        )

    def test_descendant_events_count_for_the_ancestors(self):
        self.assertEqual(self.future.event_days["11"], [40])
        self.assertEqual(self.future.event_days["10"], [40, 400])
        self.assertEqual(self.future.event_days["30"], [40, 200, 400])

    def test_prior_concepts_include_ancestors_of_history(self):
        self.assertEqual(self.future.prior_concepts, {"12", "10", "30"})

    def test_follow_up(self):
        self.assertAlmostEqual(self.future.follow_up_days, 400)

    def test_window_is_open_at_the_start_and_closed_at_the_end(self):
        has = self.future.has_event
        self.assertFalse(has("11", 0, 39))
        self.assertTrue(has("11", 0, 40))
        self.assertFalse(has("11", 40, 90))  # day 40 belongs to (0, 40], not to (40, 90]
        self.assertTrue(has("11", 30, 90))
        self.assertTrue(has("10", 90, 730))  # the later descendant event at day 400
        self.assertFalse(has("12", 0, 365))  # the type 1 event is in history and at 400
        self.assertFalse(has("unknown", 0, 730))


class GenerateQueriesTest(unittest.TestCase):
    def setUp(self):
        self.hierarchy = ConceptHierarchy(PARENTS, DOMAINS)
        self.concepts = sorted(query_concepts(TOKENS, self.hierarchy))
        self.timeline = timeline(
            [("12", day(-50)), ("11", day(40)), ("20", day(200)), ("12", day(400))]
        )
        self.future = PatientFuture(self.timeline, self.hierarchy, TOKENS, set(self.concepts))

    def generate(self, seed=0, n=6, config=None, **kwargs):
        return generate_queries(self.future, self.concepts, random.Random(seed), n, config, **kwargs)

    def test_labels_agree_with_the_events(self):
        queries = []
        for seed in range(40):
            queries += self.generate(seed)
        self.assertTrue(queries)
        for q in queries:
            self.assertEqual(q.label, self.future.has_event(q.concept_id, q.start_days, q.end_days), q)
        self.assertEqual({q.label for q in queries}, {True, False})

    def test_an_ancestor_is_positive_through_its_descendants(self):
        config = QueryConfig(positive_fraction=1.0, windows=((0, 90),))
        concepts = set()
        for seed in range(30):
            concepts |= {q.concept_id for q in self.generate(seed, config=config)}
        self.assertEqual(concepts, {"11", "10", "30"})  # type 2 at day 40 and its ancestors

    def test_negatives_need_the_record_to_cover_the_window(self):
        short = PatientFuture(
            timeline([("12", day(-5)), ("20", day(10))]), self.hierarchy, TOKENS, set(self.concepts)
        )
        config = QueryConfig(positive_fraction=0.0)
        for seed in range(60):
            for q in generate_queries(short, self.concepts, random.Random(seed), 4, config):
                self.assertFalse(q.label)
                self.assertLessEqual(q.end_days, 10)

    def test_reproducible_and_distinct(self):
        self.assertEqual(self.generate(7), self.generate(7))
        queries = self.generate(3, n=8)
        keys = [(q.concept_id, q.start_days, q.end_days) for q in queries]
        self.assertEqual(len(keys), len(set(keys)))

    def test_positive_fraction(self):
        for fraction in (0.0, 1.0):
            config = QueryConfig(positive_fraction=fraction)
            labels = {q.label for seed in range(30) for q in self.generate(seed, config=config)}
            self.assertEqual(labels, {fraction == 1.0})

    def test_new_onset_only(self):
        config = QueryConfig(new_onset_only=True)
        for seed in range(60):
            for q in self.generate(seed, config=config):
                # type 1 diabetes was recorded before the index date, and so were its ancestors
                self.assertNotIn(q.concept_id, {"12", "10", "30"})

    def test_out_of_scope_fraction(self):
        for fraction, expect_motor in ((0.0, True), (1.0, False)):
            config = QueryConfig(positive_fraction=1.0, out_of_scope_fraction=fraction, windows=((0, 365),))
            concepts = {q.concept_id for seed in range(40) for q in self.generate(seed, config=config, motor_codes={"20"})}
            self.assertEqual("20" in concepts, expect_motor)
            if not expect_motor:
                self.assertNotIn("20", concepts)

    def test_coverage_reaches_the_whole_pool(self):
        pool = [str(1000 + i) for i in range(60)]
        patient = PatientFuture(timeline([("11", day(5)), ("11", day(400))]), self.hierarchy, TOKENS, set(self.concepts) | set(pool))
        config = QueryConfig(positive_fraction=0.0)
        asked = {}
        rng = random.Random(0)
        coverage = QueryCoverage(8)
        for _ in range(80):
            for q in generate_queries(patient, pool, rng, 3, config, coverage=coverage):
                asked[q.concept_id] = asked.get(q.concept_id, 0) + 1
        self.assertEqual(len(asked), 60)
        self.assertLessEqual(max(asked.values()), 2 * min(asked.values()) + 3)

    def test_a_large_pool_is_not_scanned_per_query(self):
        pool = [str(i) for i in range(200_000)]
        patient = PatientFuture(timeline([("11", day(5)), ("11", day(400))]), self.hierarchy, TOKENS, set(self.concepts))
        config = QueryConfig(positive_fraction=0.0)
        rng = random.Random(0)
        start = time.time()
        for _ in range(300):
            generate_queries(patient, pool, rng, 4, config, coverage=QueryCoverage(8))
        self.assertLess(time.time() - start, 5.0)

    def test_invalid_config(self):
        for kwargs in (
            {"windows": ()},
            {"windows": ((10, 10),)},
            {"windows": ((-1, 5),)},
            {"positive_fraction": 1.2},
            {"out_of_scope_fraction": -0.5},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    QueryConfig(**kwargs)
        with self.assertRaises(ValueError):
            QueryCoverage(0)
        with self.assertRaises(ValueError):
            PatientTimeline(["1"], [1.0, 2.0], 0.0)


if __name__ == "__main__":
    unittest.main()
