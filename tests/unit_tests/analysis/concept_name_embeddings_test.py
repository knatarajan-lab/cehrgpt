import hashlib
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import polars as pl

from cehrgpt.analysis import concept_name_embeddings as cne

# ---------------------------------------------------------------------------------------
# A small Athena style vocabulary
# ---------------------------------------------------------------------------------------
# SNOMED:  1 Disease
#            10 Cardiovascular, 11 Respiratory, 12 Neoplastic            (depth 1)
#              20 Hypertension (10), 21 Heart failure (10), 22 Asthma (11),
#              23 Lung cancer (11 and 12), 24 Breast cancer (12)         (depth 2)
#                30 Essential hypertension (20), 31 Acute asthma (22)    (depth 3)
#              41 Odd finding: parent 20 (two levels under 10) and 11 directly
SNOMED_PARENTS = {
    10: [1], 11: [1], 12: [1],
    20: [10], 21: [10], 22: [11], 23: [11, 12], 24: [12],
    30: [20], 31: [22], 41: [20, 11],
}
SNOMED_NAMES = {
    1: "Disease", 10: "Disorder of cardiovascular system", 11: "Disorder of respiratory system",
    12: "Neoplastic disease", 20: "Hypertension", 21: "Heart failure", 22: "Asthma",
    23: "Lung cancer", 24: "Breast cancer", 30: "Essential hypertension", 31: "Acute asthma",
    41: "Odd finding",
}
# ATC: A,B,C,J,N (1st) > A10,B01,C09,J01,N02 (2nd) > the 5th level substances
ATC = {
    200: ("A", "Alimentary tract and metabolism", "ATC 1st", None),
    201: ("B", "Blood and blood forming organs", "ATC 1st", None),
    202: ("C", "Cardiovascular system", "ATC 1st", None),
    203: ("J", "Antiinfectives for systemic use", "ATC 1st", None),
    204: ("N", "Nervous system", "ATC 1st", None),
    210: ("A10", "Drugs used in diabetes", "ATC 2nd", 200),
    211: ("B01", "Antithrombotic agents", "ATC 2nd", 201),
    212: ("C09", "Agents acting on the renin-angiotensin system", "ATC 2nd", 202),
    213: ("J01", "Antibacterials for systemic use", "ATC 2nd", 203),
    214: ("N02", "Analgesics", "ATC 2nd", 204),
    220: ("A10BA02", "metformin", "ATC 5th", 210),
    221: ("B01AC06", "acetylsalicylic acid", "ATC 5th", 211),
    222: ("N02BA01", "acetylsalicylic acid", "ATC 5th", 214),
    223: ("C09AA03", "lisinopril", "ATC 5th", 212),
    224: ("J01CA04", "amoxicillin", "ATC 5th", 213),
}
RXNORM = {
    1000: ("metformin", "Ingredient", "RxNorm"),
    1001: ("aspirin", "Ingredient", "RxNorm"),
    1002: ("lisinopril", "Ingredient", "RxNorm"),
    1003: ("amoxicillin", "Ingredient", "RxNorm"),
    1004: ("unmapped ingredient", "Ingredient", "RxNorm"),
    1005: ("metformin 500 MG Oral Tablet", "Clinical Drug", "RxNorm"),
    1006: ("extension ingredient", "Ingredient", "RxNorm Extension"),
}


def concept_rows():
    rows = []

    def add(cid, name, domain, vocabulary, cls, standard="S", code="", invalid=None):
        rows.append(dict(
            concept_id=cid, concept_name=name, domain_id=domain, vocabulary_id=vocabulary,
            concept_class_id=cls, standard_concept=standard, concept_code=code,
            invalid_reason=invalid,
        ))

    for cid, name in SNOMED_NAMES.items():
        add(cid, name, "Condition", "SNOMED", "Clinical Finding")
    add(60, "Appendectomy", "Procedure", "SNOMED", "Procedure")
    add(98, "Old finding", "Condition", "SNOMED", "Clinical Finding", invalid="D")
    add(99, "Non standard finding", "Condition", "SNOMED", "Clinical Finding", standard=None)
    add(97, "ICD10 finding", "Condition", "ICD10", "ICD10 Code")
    for cid, (code, name, cls, _) in ATC.items():
        add(cid, name, "Drug", "ATC", cls, standard="C", code=code)
    for cid, (name, cls, vocabulary) in RXNORM.items():
        add(cid, name, "Drug", vocabulary, cls)
    return rows


def closure(parents, nodes):
    """concept_ancestor rows: every ancestor with its minimum number of levels."""
    rows = []
    for node in nodes:
        distance, queue = {node: 0}, [node]
        while queue:
            current = queue.pop(0)
            for parent in parents.get(current, []):
                if parent not in distance:
                    distance[parent] = distance[current] + 1
                    queue.append(parent)
        rows += [
            dict(ancestor_concept_id=a, descendant_concept_id=node,
                 min_levels_of_separation=d, max_levels_of_separation=d)
            for a, d in distance.items()
        ]
    return rows


def ancestor_rows():
    parents = dict(SNOMED_PARENTS)
    parents.update({cid: [v[3]] for cid, v in ATC.items() if v[3]})
    # the clinical drug sits below its ingredient
    parents[1005] = [1000]
    nodes = set(SNOMED_NAMES) | set(ATC) | set(RXNORM) | {60}
    return closure(parents, nodes)


def relationship_rows():
    def row(a, b, relationship="ATC - RxNorm", invalid=None):
        return dict(concept_id_1=a, concept_id_2=b, relationship_id=relationship, invalid_reason=invalid)

    return [
        row(220, 1000),                              # metformin, ATC to RxNorm
        row(1000, 220, "RxNorm - ATC"),              # the same link in the other direction
        row(221, 1001), row(222, 1001),              # aspirin: two ATC classes (B and N)
        row(1002, 223, "RxNorm - ATC"),              # lisinopril, only the RxNorm to ATC direction
        row(224, 1003, invalid="D"),                 # amoxicillin: the only link is invalid
        row(1000, 20, "Maps to"),                    # not an ATC link
        row(221, 1006),                              # an RxNorm Extension ingredient
    ]


def write_tables(folder, fmt="parquet"):
    tables = {
        "concept": pl.DataFrame(concept_rows()),
        "concept_ancestor": pl.DataFrame(ancestor_rows()),
        "concept_relationship": pl.DataFrame(relationship_rows()),
    }
    paths = {}
    for name, frame in tables.items():
        if fmt == "parquet":
            path = os.path.join(folder, f"{name}.parquet")
            frame.write_parquet(path)
        elif fmt == "folder":
            path = os.path.join(folder, name)
            os.makedirs(path, exist_ok=True)
            frame.write_parquet(os.path.join(path, "part-0.parquet"))
        else:  # Athena: tab separated, upper case header, no quoting
            path = os.path.join(folder, name.upper() + ".csv")
            frame.rename({c: c.upper() for c in frame.columns}).write_csv(
                path, separator="\t", quote_style="never"
            )
        paths[name] = path
    return paths


class FakeEncoder:
    """Hash based vectors; counts how many texts it was asked to embed."""

    def __init__(self, model_name="fake", dim=16):
        self.model_name, self.dim, self.calls = model_name, dim, 0

    def describe(self):
        return {"model_name": self.model_name, "pooling": "fake"}

    def encode(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
            out[i] = np.random.RandomState(seed).randn(self.dim)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------
class LoadingTest(TempDirTest):
    def test_the_same_result_from_parquet_folders_and_athena_csv(self):
        results = []
        for fmt in ("parquet", "folder", "csv"):
            folder = os.path.join(self.tmp, fmt)
            os.makedirs(folder)
            paths = write_tables(folder, fmt)
            results.append(cne.load_concepts(paths["concept"]).sort("concept_id"))
        for other in results[1:]:
            self.assertTrue(results[0].equals(other))

    def test_snomed_standard_valid_concepts_only(self):
        paths = write_tables(self.tmp)
        ids = set(cne.load_concepts(paths["concept"])["concept_id"].to_list())
        self.assertEqual(ids, set(SNOMED_NAMES) | {60})  # no ICD10, non standard or invalid
        self.assertIn(99, set(cne.load_concepts(paths["concept"], include_non_standard=True)["concept_id"]))
        self.assertIn(98, set(cne.load_concepts(paths["concept"], include_invalid=True)["concept_id"]))

    def test_domains_and_classes(self):
        paths = write_tables(self.tmp)
        self.assertEqual(cne.load_concepts(paths["concept"], domains=["Procedure"])["concept_id"].to_list(), [60])
        rx = cne.load_concepts(paths["concept"], "RxNorm", concept_classes=["Ingredient"])
        self.assertEqual(set(rx["concept_id"].to_list()), {1000, 1001, 1002, 1003, 1004})

    def test_missing_columns_and_empty_names(self):
        path = os.path.join(self.tmp, "c.parquet")
        pl.DataFrame({"concept_id": [1], "concept_name": ["x"]}).write_parquet(path)
        with self.assertRaisesRegex(ValueError, "vocabulary_id"):
            cne.load_concepts(path)

    def test_text_template(self):
        frame = pl.DataFrame({"concept_name": ["Asthma"], "domain_id": ["Condition"]})
        self.assertEqual(cne.render_texts(frame, "{domain_id}: {concept_name}"), ["Condition: Asthma"])

    def test_find_concept_id(self):
        paths = write_tables(self.tmp)
        self.assertEqual(cne.find_concept_id(paths["concept"], "Disease"), 1)
        with self.assertRaisesRegex(ValueError, "No standard SNOMED concept"):
            cne.find_concept_id(paths["concept"], "Not a concept")


# ---------------------------------------------------------------------------------------
class DiseaseGroupTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.paths = write_tables(self.tmp)
        self.ancestors = cne.read_ancestors(self.paths["concept_ancestor"])
        self.ids = sorted(SNOMED_NAMES)

    def test_anchors_by_depth(self):
        self.assertEqual(cne.discover_anchors(self.ancestors, 1, 1), [10, 11, 12])
        # 41 is directly under 11, so it is two levels below the root as well
        self.assertEqual(cne.discover_anchors(self.ancestors, 1, 2), [20, 21, 22, 23, 24, 41])

    def groups(self, **kwargs):
        anchors = {a: SNOMED_NAMES[a] for a in cne.discover_anchors(self.ancestors, 1, 1)}
        return cne.assign_disease_groups(self.ancestors, self.ids, anchors, **kwargs)

    def test_a_concept_takes_the_group_of_its_anchor_ancestor(self):
        names = self.groups().names
        self.assertEqual(names[20], "Disorder of cardiovascular system")
        self.assertEqual(names[30], "Disorder of cardiovascular system")  # two levels below the anchor
        self.assertEqual(names[31], "Disorder of respiratory system")
        self.assertEqual(names[10], "Disorder of cardiovascular system")  # an anchor is in its own group
        self.assertEqual(names[1], cne.OTHER_GROUP)  # the root is above every anchor

    def test_the_closest_anchor_wins(self):
        # 41 is two levels under the cardiovascular anchor but one under the respiratory one
        self.assertEqual(self.groups().names[41], "Disorder of respiratory system")

    def test_equal_distance_is_multiple_groups_or_the_first(self):
        # lung cancer is directly under the respiratory and the neoplastic anchors
        self.assertEqual(self.groups().names[23], cne.MULTIPLE_GROUP)
        self.assertEqual(self.groups(tie_break="first").names[23], "Disorder of respiratory system")
        with self.assertRaises(ValueError):
            self.groups(tie_break="nope")

    def test_group_cap_and_size_floor(self):
        grouping = self.groups(max_groups=1)
        real = [g for g in grouping.sizes if g not in (cne.OTHER_GROUP, cne.MULTIPLE_GROUP)]
        self.assertEqual(len(real), 1)
        small = self.groups(min_group_size=100)
        self.assertEqual(set(small.sizes), {cne.OTHER_GROUP, cne.MULTIPLE_GROUP})
        self.assertEqual(sum(self.groups().sizes.values()), len(self.ids))

    def test_direct_parents(self):
        parents = cne.direct_parents(self.ancestors, self.ids)
        self.assertEqual(sorted(parents[23]), [11, 12])
        self.assertEqual(parents[30], [20])
        self.assertNotIn(1, parents)


# ---------------------------------------------------------------------------------------
class AtcGroupTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.paths = write_tables(self.tmp)
        self.ancestors = cne.read_ancestors(self.paths["concept_ancestor"])
        self.pairs = cne.load_atc_pairs(self.paths["concept"], self.paths["concept_relationship"])

    def pairs_as_set(self, pairs=None):
        pairs = self.pairs if pairs is None else pairs
        return set(zip(pairs["drug_concept_id"].to_list(), pairs["atc_concept_id"].to_list()))

    def test_links_in_both_directions_valid_only(self):
        self.assertEqual(
            self.pairs_as_set(),
            {(1000, 220), (1001, 221), (1001, 222), (1002, 223)},
        )  # metformin once despite two rows; amoxicillin's invalid link and the "Maps to" row are out

    def test_relationship_names_and_extra_vocabularies(self):
        only = cne.load_atc_pairs(self.paths["concept"], self.paths["concept_relationship"], relationship_ids=["RxNorm - ATC"])
        self.assertEqual(self.pairs_as_set(only), {(1000, 220), (1002, 223)})
        extended = cne.load_atc_pairs(self.paths["concept"], self.paths["concept_relationship"], ("RxNorm", "RxNorm Extension"))
        self.assertIn((1006, 221), self.pairs_as_set(extended))

    def test_level_labels_use_the_atc_code(self):
        labels = cne.atc_level_labels(self.paths["concept"], 1)
        self.assertEqual(labels[200], "A: Alimentary tract and metabolism")
        self.assertEqual(set(labels), {200, 201, 202, 203, 204})
        with self.assertRaises(ValueError):
            cne.atc_level_labels(self.paths["concept"], 6)

    def assign(self, level=1, drugs=(1000, 1001, 1002, 1003, 1004, 1005), **kwargs):
        return cne.assign_atc_groups(
            self.pairs, self.ancestors, cne.atc_level_labels(self.paths["concept"], level),
            list(drugs), kwargs.pop("ingredient_ids", [1000, 1001, 1002, 1003, 1004]), **kwargs,
        ).names

    def test_first_level(self):
        names = self.assign()
        self.assertEqual(names[1000], "A: Alimentary tract and metabolism")
        self.assertEqual(names[1002], "C: Cardiovascular system")
        self.assertEqual(names[1001], cne.MULTIPLE_GROUP)  # blood and nervous system
        self.assertEqual(names[1003], cne.OTHER_GROUP)  # its only link is invalid
        self.assertEqual(names[1004], cne.OTHER_GROUP)  # no link at all

    def test_a_drug_without_a_link_takes_the_group_of_its_ingredient(self):
        self.assertEqual(self.assign()[1005], "A: Alimentary tract and metabolism")
        without = self.assign(ingredient_ids=[])
        self.assertEqual(without[1005], cne.OTHER_GROUP)

    def test_tie_break_first_takes_the_lowest_class(self):
        self.assertEqual(self.assign(tie_break="first")[1001], "B: Blood and blood forming organs")

    def test_other_levels(self):
        self.assertEqual(self.assign(level=2)[1000], "A10: Drugs used in diabetes")
        self.assertEqual(self.assign(level=5)[1000], "A10BA02: metformin")
        # at level 5 aspirin is in two different classes, at level 2 as well
        self.assertEqual(self.assign(level=2)[1001], cne.MULTIPLE_GROUP)

    def test_climbing_needs_the_ancestor_table(self):
        with self.assertRaisesRegex(ValueError, "concept_ancestor"):
            cne.assign_atc_groups(self.pairs, None, cne.atc_level_labels(self.paths["concept"], 1), [1000])
        # the fifth level is the level of the links themselves, so no climbing is needed
        result = cne.assign_atc_groups(self.pairs, None, cne.atc_level_labels(self.paths["concept"], 5), [1000, 1002])
        self.assertEqual(result.names[1002], "C09AA03: lisinopril")

    def test_ingredients_of(self):
        self.assertEqual(set(cne.ingredients_of(self.paths["concept"])), {1000, 1001, 1002, 1003, 1004})


# ---------------------------------------------------------------------------------------
class AnalysisTest(unittest.TestCase):
    def blobs(self, n_groups=4, per_group=60, dim=16, noise=0.1, seed=0):
        rng = np.random.RandomState(seed)
        centres = np.linalg.qr(rng.randn(dim, dim))[0][:n_groups]
        vectors = np.concatenate([c + noise * rng.randn(per_group, dim) for c in centres])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        groups = [f"g{i}" for i in range(n_groups) for _ in range(per_group)]
        return vectors.astype(np.float32), groups

    def test_top_k_matches_a_brute_force_search(self):
        rng = np.random.RandomState(1)
        corpus = rng.randn(200, 8).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        queries = np.arange(0, 200, 7)
        found = cne.top_k_cosine(corpus[queries], corpus, 5, exclude=queries, chunk=13)
        scores = corpus[queries] @ corpus.T
        scores[np.arange(len(queries)), queries] = -np.inf
        np.testing.assert_array_equal(found, np.argsort(-scores, axis=1)[:, :5])
        self.assertNotIn(int(queries[0]), found[0])

    def test_top_k_with_fewer_rows_than_k(self):
        corpus = np.eye(3, dtype=np.float32)
        self.assertEqual(cne.top_k_cosine(corpus, corpus, 10, exclude=np.arange(3)).shape, (3, 2))

    def test_separated_groups_cluster_perfectly(self):
        vectors, groups = self.blobs()
        metrics, composition = cne.cluster_agreement(vectors, groups, 4)
        self.assertGreater(metrics["adjusted_rand_index"], 0.99)
        self.assertGreater(metrics["homogeneity"], 0.99)
        self.assertTrue(all(c["top_group_share"] > 0.99 for c in composition))
        self.assertEqual(sum(c["size"] for c in composition), len(groups))

    def test_random_vectors_do_not_agree_with_the_groups(self):
        rng = np.random.RandomState(0)
        vectors = rng.randn(240, 16).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        _, groups = self.blobs()
        metrics, _ = cne.cluster_agreement(vectors, groups, 4)
        self.assertLess(metrics["adjusted_rand_index"], 0.1)

    def test_neighbour_purity_against_chance(self):
        vectors, groups = self.blobs()
        result = cne.neighbor_group_purity(vectors, groups, k=5, n_queries=100)
        self.assertGreater(result["purity"], 0.97)
        self.assertAlmostEqual(result["chance"], 0.25, places=6)
        self.assertEqual(set(result["purity_by_group"]), {"g0", "g1", "g2", "g3"})
        rng = np.random.RandomState(2)
        noise = rng.randn(*vectors.shape).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        random = cne.neighbor_group_purity(noise, groups, k=5, n_queries=240)
        self.assertLess(random["purity"], 0.4)

    def test_parent_retrieval(self):
        rng = np.random.RandomState(0)
        parents_vec = rng.randn(30, 32)
        children_vec = parents_vec + 0.05 * rng.randn(30, 32)
        other = rng.randn(100, 32)
        vectors = np.concatenate([parents_vec, children_vec, other]).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = list(range(len(vectors)))
        parents = {30 + i: [i] for i in range(30)}
        result = cne.parent_retrieval(vectors, ids, parents, ks=(1, 5))
        self.assertEqual(result["n_queries"], 30)
        self.assertEqual(result["accuracy_at_1"], 1.0)
        self.assertEqual(cne.parent_retrieval(vectors, ids, {})["n_queries"], 0)
        # a child whose parent is a random vector is not found
        shuffled = vectors.copy()
        shuffled[:30] = rng.randn(30, 32) / 6
        shuffled[:30] /= np.linalg.norm(shuffled[:30], axis=1, keepdims=True)
        self.assertLess(cne.parent_retrieval(shuffled, ids, parents, ks=(1,))["accuracy_at_1"], 0.5)

    def test_projection_and_plot(self):
        vectors, groups = self.blobs(per_group=20)
        for method in ("pca", "tsne", "auto"):
            coordinates, used = cne.project_2d(vectors, method)
            self.assertEqual(coordinates.shape, (len(vectors), 2))
        with tempfile.TemporaryDirectory() as folder:
            groups = groups[:-5] + [cne.OTHER_GROUP] * 3 + [cne.MULTIPLE_GROUP] * 2
            path = os.path.join(folder, "p.png")
            cne.plot_projection(coordinates, groups, path, "title")
            self.assertGreater(os.path.getsize(path), 5000)
        with self.assertRaises(ValueError):
            cne.project_2d(vectors, "nope")


# ---------------------------------------------------------------------------------------
def make_tiny_model(folder):
    """A small random BERT with a toy vocabulary, so the encoder can be tested offline."""
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "heart", "failure", "asthma", "acute",
             "essential", "hypertension", "metformin", "cancer", "lung", "of", "the", "disease", "oral", "tablet"]
    vocab = os.path.join(folder, "vocab.txt")
    with open(vocab, "w") as f:
        f.write("\n".join(words))
    BertTokenizerFast(vocab_file=vocab).save_pretrained(folder)
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64)).save_pretrained(folder)


class EncoderTest(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp()
        make_tiny_model(cls.folder)
        cls.encoder = cne.HuggingFaceEncoder(cls.folder, batch_size=3, device="cpu")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_vectors_are_unit_length(self):
        vectors = self.encoder.encode(["heart failure", "asthma", "acute lung cancer disease"])
        self.assertEqual(vectors.shape, (3, 32))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_padding_does_not_change_a_vector(self):
        alone = self.encoder.encode(["asthma"])[0]
        long_text = "acute essential hypertension of the lung cancer disease oral tablet"
        together = self.encoder.encode(["asthma", long_text])[0]
        np.testing.assert_allclose(alone, together, atol=1e-5)

    def test_order_is_preserved_through_the_length_sorting(self):
        texts = ["acute essential hypertension of the lung", "asthma", "heart failure of the lung cancer", "metformin"]
        full = self.encoder.encode(texts)
        for i, text in enumerate(texts):
            np.testing.assert_allclose(full[i], self.encoder.encode([text])[0], atol=1e-5)
        np.testing.assert_allclose(self.encoder.encode(texts[::-1])[::-1], full, atol=1e-5)

    def test_empty_input_and_description(self):
        self.assertEqual(self.encoder.encode([]).shape[0], 0)
        description = self.encoder.describe()
        self.assertEqual(description["pooling"], "masked mean over tokens")
        self.assertEqual(description["normalize"], "l2")

    def test_mean_pool_ignores_padding(self):
        import torch

        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
        pooled = cne.mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))


# ---------------------------------------------------------------------------------------
class CacheTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.paths = write_tables(self.tmp)
        self.concepts = cne.load_concepts(self.paths["concept"])
        self.texts = cne.render_texts(self.concepts, "{concept_name}")
        self.out = os.path.join(self.tmp, "out")

    def embed(self, encoder, texts=None, **kwargs):
        return cne.embed_with_cache(self.concepts, texts or self.texts, encoder, self.out, chunk_size=4, **kwargs)

    def test_a_second_run_reuses_the_cache(self):
        encoder = FakeEncoder()
        first = self.embed(encoder)
        self.assertEqual(encoder.calls, len(self.texts))
        again = self.embed(encoder)
        self.assertEqual(encoder.calls, len(self.texts))
        np.testing.assert_array_equal(first, again)
        for name in ("embeddings.npy", "concepts.parquet", "embedder.json"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)))

    def test_a_different_model_or_text_recomputes(self):
        self.embed(FakeEncoder("a"))
        encoder = FakeEncoder("b")
        self.embed(encoder)
        self.assertEqual(encoder.calls, len(self.texts))
        changed = FakeEncoder("b")
        self.embed(changed, ["Condition: " + t for t in self.texts])
        self.assertEqual(changed.calls, len(self.texts))

    def test_an_interrupted_run_resumes_the_missing_chunk(self):
        encoder = FakeEncoder()
        full = self.embed(encoder)
        os.remove(os.path.join(self.out, "embedding_parts", "part_00001.npy"))
        resumed = FakeEncoder()
        np.testing.assert_array_equal(self.embed(resumed), full)
        self.assertEqual(resumed.calls, 4)  # one chunk of four texts

    def test_overwrite(self):
        self.embed(FakeEncoder())
        encoder = FakeEncoder()
        self.embed(encoder, overwrite=True)
        self.assertEqual(encoder.calls, len(self.texts))


# ---------------------------------------------------------------------------------------
class EndToEndTest(TempDirTest):
    def args(self, *extra):
        out = os.path.join(self.tmp, "out")
        return cne.create_arg_parser().parse_args([
            "--concept_path", self.paths["concept"],
            "--concept_ancestor_path", self.paths["concept_ancestor"],
            "--concept_relationship_path", self.paths["concept_relationship"],
            "--output_dir", out, "--min_group_size", "1", "--knn", "3", *extra,
        ])

    def setUp(self):
        super().setUp()
        self.paths = write_tables(self.tmp)

    def read(self, name):
        with open(os.path.join(self.tmp, "out", name)) as f:
            return json.load(f)

    def test_snomed_run(self):
        metrics = cne.run(self.args("--group_depth", "1", "--domains", "Condition"), FakeEncoder())
        for name in ("embeddings.npy", "metrics.json", "groups.csv", "projection.csv", "projection.png", "cluster_composition.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, "out", name)), name)
        self.assertEqual(metrics["vocabulary_id"], "SNOMED")
        self.assertEqual(metrics["num_concepts"], len(SNOMED_NAMES))
        self.assertIn("parent_retrieval", metrics)
        self.assertIn("neighbour_group_purity", metrics)
        groups = pl.read_csv(os.path.join(self.tmp, "out", "groups.csv"))
        self.assertIn("Disorder of cardiovascular system", groups["group"].to_list())
        projection = pl.read_csv(os.path.join(self.tmp, "out", "projection.csv"))
        self.assertEqual(len(projection), len(SNOMED_NAMES))
        self.assertEqual(self.read("metrics.json")["embedding_dim"], 16)

    def test_rxnorm_run_colours_by_atc(self):
        metrics = cne.run(
            self.args("--vocabulary_id", "RxNorm", "--concept_classes", "Ingredient", "--atc_level", "1"),
            FakeEncoder(),
        )
        self.assertEqual(metrics["vocabulary_id"], "RxNorm")
        self.assertEqual(metrics["num_concepts"], 5)
        self.assertNotIn("parent_retrieval", metrics)
        projection = pl.read_csv(os.path.join(self.tmp, "out", "projection.csv"))
        by_name = dict(zip(projection["concept_name"].to_list(), projection["group"].to_list()))
        self.assertEqual(by_name["metformin"], "A: Alimentary tract and metabolism")
        self.assertEqual(by_name["aspirin"], cne.MULTIPLE_GROUP)
        self.assertEqual(by_name["unmapped ingredient"], cne.OTHER_GROUP)

    def test_runs_can_share_one_embedding_cache(self):
        encoder = FakeEncoder()
        cache = os.path.join(self.tmp, "cache")
        first = self.args("--group_depth", "1", "--domains", "Condition", "--cache_dir", cache)
        cne.run(first, encoder)
        embedded = encoder.calls
        self.assertGreater(embedded, 0)
        for name in ("embeddings.npy", "concepts.parquet", "embedder.json"):
            self.assertTrue(os.path.exists(os.path.join(cache, name)), name)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out", "embeddings.npy")))
        # another grouping of the same concepts, in another folder: nothing is embedded again
        second = self.args("--group_depth", "2", "--domains", "Condition", "--cache_dir", cache)
        second.output_dir = os.path.join(self.tmp, "out_depth2")
        cne.run(second, encoder)
        self.assertEqual(encoder.calls, embedded)
        self.assertTrue(os.path.exists(os.path.join(second.output_dir, "metrics.json")))

    def test_colouring_by_a_column_needs_no_hierarchy(self):
        args = self.args("--color_by", "domain_id", "--domains", "Condition", "Procedure")
        args.concept_ancestor_path = None
        metrics = cne.run(args, FakeEncoder())
        self.assertNotIn("parent_retrieval", metrics)
        self.assertEqual(metrics["projection"], "tsne")

    def test_helpful_errors(self):
        args = self.args("--color_by", "atc_group")
        args.concept_relationship_path = None
        with self.assertRaisesRegex(ValueError, "concept_relationship_path"):
            cne.run(args, FakeEncoder())
        args = self.args("--group_depth", "9")
        with self.assertRaisesRegex(ValueError, "level"):
            cne.run(args, FakeEncoder())
        args = self.args("--domains", "Device")
        with self.assertRaisesRegex(ValueError, "No concepts"):
            cne.run(args, FakeEncoder())


# ---------------------------------------------------------------------------------------
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "run_concept_name_embeddings.sh")


class ShellScriptTest(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.model = tempfile.mkdtemp()
        make_tiny_model(cls.model)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.model, ignore_errors=True)

    def run_script(self, *args, check=True):
        import subprocess
        import sys

        result = subprocess.run(
            ["sh", SCRIPT, "--python=" + sys.executable, *args],
            capture_output=True, text=True,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_a_full_run_on_athena_style_csv_files(self):
        vocab = os.path.join(self.tmp, "athena")
        os.makedirs(vocab)
        write_tables(vocab, "csv")
        out = os.path.join(self.tmp, "out")
        result = self.run_script(
            f"--vocab_dir={vocab}", f"--output_dir={out}", f"--model={self.model}",
            "--snomed_group_depths=1 2", "--atc_levels=1", "--batch_size=4",
            "--extra_args=--min_group_size 1 --knn 3 --max_plot_points 60",
        )
        for run in ("snomed_depth1", "snomed_depth2", "rxnorm_atc1"):
            for name in ("metrics.json", "projection.png", "projection.csv", "groups.csv"):
                self.assertTrue(os.path.exists(os.path.join(out, run, name)), f"{run}/{name}")
        # the two SNOMED runs share one embedding cache, so only the first one embeds
        first = open(os.path.join(out, "logs", "snomed_depth1.log")).read()
        second = open(os.path.join(out, "logs", "snomed_depth2.log")).read()
        self.assertIn("Embedding concepts", first)
        self.assertNotIn("Embedding concepts", second)
        self.assertTrue(os.path.exists(os.path.join(out, "cache", "snomed", "embeddings.npy")))
        self.assertTrue(os.path.exists(os.path.join(out, "cache", "rxnorm", "embeddings.npy")))
        # the numbers are summarised in the run log and on the screen
        self.assertIn("nearest 3 neighbours in the same group", result.stdout)
        self.assertIn("k-means", open(os.path.join(out, "run.log")).read())
        with open(os.path.join(out, "snomed_depth1", "metrics.json")) as f:
            self.assertEqual(json.load(f)["num_concepts"], len(SNOMED_NAMES))
        with open(os.path.join(out, "rxnorm_atc1", "metrics.json")) as f:
            self.assertEqual(json.load(f)["num_concepts"], 5)

    def test_parquet_folders_and_a_single_analysis(self):
        vocab = os.path.join(self.tmp, "folders")
        os.makedirs(vocab)
        write_tables(vocab, "folder")
        out = os.path.join(self.tmp, "out")
        self.run_script(
            f"--vocab_dir={vocab}", f"--output_dir={out}", f"--model={self.model}",
            "--analyses=rxnorm", "--atc_levels=2", "--batch_size=4",
            "--extra_args=--min_group_size 1 --knn 3",
        )
        self.assertTrue(os.path.exists(os.path.join(out, "rxnorm_atc2", "metrics.json")))
        self.assertFalse(os.path.exists(os.path.join(out, "snomed_depth2")))

    def test_dry_run_prints_the_commands_and_writes_nothing(self):
        vocab = os.path.join(self.tmp, "athena")
        os.makedirs(vocab)
        write_tables(vocab, "csv")
        out = os.path.join(self.tmp, "never")
        result = self.run_script(
            f"--vocab_dir={vocab}", f"--output_dir={out}", "--dry_run", "--gpu_id=0",
            "--rxnorm_concept_classes=Ingredient,Clinical Drug", "--snomed_domains=Condition Procedure",
            "--snomed_group_root_id=1",
        )
        self.assertFalse(os.path.exists(out))
        for expected in ("CONCEPT.csv", "CONCEPT_ANCESTOR.csv", "CONCEPT_RELATIONSHIP.csv",
                         "--cache_dir", "--color_by disease_group", "--color_by atc_group",
                         "--group_root_id 1", "--domains Condition Procedure",
                         "--concept_classes Ingredient Clinical Drug"):
            self.assertIn(expected, result.stdout)
        self.assertEqual(result.stdout.count("-m cehrgpt.analysis.concept_name_embeddings"), 3)  # 1 snomed + 2 atc levels

    def test_all_domains_and_all_classes_drop_the_filters(self):
        vocab = os.path.join(self.tmp, "athena")
        os.makedirs(vocab)
        write_tables(vocab, "csv")
        result = self.run_script(
            f"--vocab_dir={vocab}", f"--output_dir={self.tmp}/o", "--dry_run",
            "--snomed_domains=all", "--rxnorm_concept_classes=all",
        )
        self.assertNotIn("--domains", result.stdout)
        self.assertNotIn("--concept_classes", result.stdout)

    def test_explicit_paths_override_the_vocab_dir(self):
        vocab = os.path.join(self.tmp, "athena")
        os.makedirs(vocab)
        paths = write_tables(vocab, "csv")
        result = self.run_script(
            f"--vocab_dir={vocab}", f"--concept_path={paths['concept']}.other", f"--output_dir={self.tmp}/o",
            "--dry_run", check=False,
        )
        self.assertNotEqual(result.returncode, 0)  # the explicit path does not exist
        self.assertIn("concept table was not found", result.stdout)

    def test_clear_errors(self):
        vocab = os.path.join(self.tmp, "athena")
        os.makedirs(vocab)
        write_tables(vocab, "csv")
        cases = [
            (["--vocab_dir=" + vocab], "--output_dir is required"),
            (["--output_dir=" + self.tmp + "/o"], "concept table was not found"),
            (["--vocab_dir=" + self.tmp, "--output_dir=" + self.tmp + "/o"], "concept table was not found"),
            (["--vocab_dir=" + vocab, "--output_dir=" + self.tmp + "/o", "--analyses=bogus"], "unknown analysis"),
            (["--vocab_dir=" + vocab, "--output_dir=" + self.tmp + "/o", "--nonsense"], "Unknown option"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                result = self.run_script(*args, check=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stdout)
        os.remove(os.path.join(vocab, "CONCEPT_RELATIONSHIP.csv"))
        result = self.run_script("--vocab_dir=" + vocab, "--output_dir=" + self.tmp + "/o", "--analyses=rxnorm", check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("concept_relationship", result.stdout)
        # snomed does not need it
        self.run_script("--vocab_dir=" + vocab, "--output_dir=" + self.tmp + "/o", "--analyses=snomed", "--dry_run")

    def test_help(self):
        result = self.run_script("--help")
        self.assertIn("--vocab_dir=DIR", result.stdout)
        self.assertIn("--atc_levels=LIST", result.stdout)

    def test_a_failing_run_stops_with_its_log(self):
        vocab = os.path.join(self.tmp, "athena")
        os.makedirs(vocab)
        write_tables(vocab, "csv")
        result = self.run_script(
            f"--vocab_dir={vocab}", f"--output_dir={self.tmp}/o", "--model=/no/such/model",
            "--analyses=rxnorm", check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("failed", result.stdout)


if __name__ == "__main__":
    unittest.main()
