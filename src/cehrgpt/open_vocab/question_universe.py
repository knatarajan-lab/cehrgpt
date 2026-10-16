"""
The questions the open vocabulary head is trained on: the concepts of the tokenizer and their
ancestors in the OHDSI hierarchy, up to a fixed list of top level categories.

A question is a concept of the tokenizer, or an ancestor of one that is a top level category or
below one. The categories are listed in this file; nothing above them is a question, which keeps
out groups that are too generic to ask about ("Disorder by body site", "Procedure by method"). A
concept below no category is still a question, but only itself. Only SNOMED ancestors are used:
concept_ancestor also lists MedDRA and other vocabularies above a SNOMED concept, and the CPT4,
ICD10PCS and ICD9Proc procedures have SNOMED ancestors.

Visits are a domain of their own: the visit types (inpatient, outpatient, emergency room, ...) are the
categories, and the places of service and specialties below them (Inpatient Hospital, Ambulatory
Surgical Center, Radiology Clinic/Center, ...) are questions that roll up to the type through the
hierarchy, so an inpatient hospital stay is also an inpatient visit. These concepts live in the Visit,
CMS Place of Service, NUCC and Medicare Specialty vocabularies.

Drugs (RxNorm, RxNorm Extension) are mapped to their ingredient: the questions are the drug concepts
of the tokenizer and the ingredients above them, and nothing in between (clinical drug components and
forms, which mostly cover the same patients as one child) and nothing above (no ATC classes).

    python -m cehrgpt.open_vocab.question_universe --vocab_path OHDSI_DIR \
        --tokenizers TOKENIZER_DIR [TOKENIZER_DIR ...] --output_dir OUT

The lists are a first draft taken from the groups at depth 1 to 3 below Disease, Clinical finding
and Procedure that cover enough tokens of the tokenizers. The id and the name of every category are
checked against the concept table, so an id of another vocabulary version fails loudly.
"""

import argparse
import glob
import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import polars as pl

LOG = logging.getLogger(__name__)

SNOMED = "SNOMED"
RXNORM = ("RxNorm", "RxNorm Extension")
INGREDIENT = "Ingredient"

# The vocabularies whose ancestors are questions, and for the drugs the only class an ancestor may have
VISIT_VOCABULARIES = ("Visit", "CMS Place of Service", "NUCC", "Medicare Specialty")
VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "Condition": (SNOMED,),
    "Procedure": (SNOMED,),
    "Drug": RXNORM,
    "Visit": VISIT_VOCABULARIES,
}
ANCESTOR_CLASSES: Dict[str, Optional[Tuple[str, ...]]] = {
    "Condition": None,
    "Procedure": None,
    "Drug": (INGREDIENT,),
    "Visit": None,
}

# Conditions below Disease: the kind of disease
DISEASE_CATEGORIES = (
    (432795, 'Traumatic AND/OR non-traumatic injury'),
    (4027384, 'Inflammatory disorder'),
    (4266186, 'Neoplasm and/or hamartoma'),
    (433128, 'Complication'),
    (432250, 'Disorder due to infection'),
    (440508, 'Congenital disease'),
    (435244, 'Developmental disorder'),
    (4179873, 'Degenerative disorder'),
    (432586, 'Mental disorder'),
    (444363, 'Drug-related disorder'),
    (436670, 'Metabolic disease'),
    (441406, 'Disorder of fetus or newborn'),
    (439658, 'Disorder of pregnancy'),
    (440371, 'Disorder of immune function'),
    (443723, 'Disorder of cellular component of blood'),
    (4185470, 'Disorder of hematopoietic morphology'),
    (442562, 'Poisoning'),
    (37204336, 'Genetic disease'),
    (4180630, 'Disorder of labor / delivery'),
    (4090739, 'Nutritional disorder'),
    (4179872, 'Disorder of hemostatic system'),
    (435524, 'Sleep disorder'),
    (443404, 'Disorder of puerperium'),
)

# Conditions below Disease: the body system (the first level below the generic 'by body site' groups)
BODY_SYSTEM_CATEGORIES = (
    (4244662, 'Disorder of musculoskeletal system'),
    (4201745, 'Disorder of digestive system'),
    (134057, 'Disorder of cardiovascular system'),
    (376337, 'Disorder of nervous system'),
    (4171379, 'Disorder of the genitourinary system'),
    (4134440, 'Visual system disorder'),
    (4028387, 'Disorder of integument'),
    (320136, 'Disorder of respiratory system'),
    (31821, 'Disorder of endocrine system'),
    (4176644, 'Disorder of auditory system'),
    (4339468, 'Ear, nose and throat disorder'),
    (4042502, 'Disease of mouth'),
    (317248, 'Disorder of hematopoietic structure'),
)

# Conditions that are findings, below Clinical finding: symptoms and signs. Some are also above diseases
FINDING_CATEGORIES = (
    (4011630, 'Neurological finding'),
    (4170962, 'Pain / sensation finding'),
    (4088927, 'Pregnancy, childbirth and puerperium finding'),
    (432453, 'General clinical state finding'),
    (4179304, 'Finding of movement'),
    (437382, 'Urine finding'),
    (437312, 'Bleeding'),
    (4275722, 'Swelling'),
    (135930, 'Musculoskeletal finding'),
    (4024705, 'Fetal finding'),
    (4041284, 'Functional finding'),
    (4096864, 'Abdominal organ finding'),
    (4293175, 'Mental state, behavior and/or psychosocial function finding'),
)

# Procedures: the body system or organ, and endoscopy. The other kinds of procedure (surgical, imaging,
# evaluation, ...) are too generic to ask about
PROCEDURE_CATEGORIES = (
    (4042641, 'Procedure on cardiovascular system'),
    (4042673, 'Procedure on heart'),
    (4040551, 'Procedure on blood vessel'),
    (4042516, 'Procedure on musculoskeletal system'),
    (4043018, 'Procedure on digestive system'),
    (4040392, 'Procedure on genitourinary system'),
    (4043019, 'Procedure on nervous system'),
    (4043017, 'Procedure on respiratory system'),
    (4155790, 'Procedure on visual system'),
    (4043020, 'Procedure on endocrine system'),
    (4040401, 'Procedure on skin'),
    (4040393, 'Procedure on ear and related structures'),
    (4134148, 'Procedure on female genital organ'),
    (4134149, 'Procedure on male genital organ'),
    (4044900, 'Procedure on bone (organ)'),
    (4042330, 'Procedure on joint'),
    (4189705, 'Breast procedure'),
    (4313889, 'Endoscopy'),
)

# Visits: the type of visit. The places of service and the specialties below them are questions too
VISIT_CATEGORIES = (
    (9201, 'Inpatient Visit'),
    (9202, 'Outpatient Visit'),
    (9203, 'Emergency Room Visit'),
    (262, 'Emergency Room and Inpatient Visit'),
    (581476, 'Home Visit'),
)

CATEGORIES: Dict[str, Optional[Tuple[Tuple[int, str], ...]]] = {
    "Condition": DISEASE_CATEGORIES + BODY_SYSTEM_CATEGORIES + FINDING_CATEGORIES,
    "Procedure": PROCEDURE_CATEGORIES,
    # every RxNorm ingredient is a category, see ingredient_categories
    "Drug": None,
    "Visit": VISIT_CATEGORIES,
}


def ingredient_categories(concept: pl.DataFrame) -> Tuple[Tuple[int, str], ...]:
    """Every RxNorm and RxNorm Extension ingredient: the categories of the drugs."""
    found = concept.filter(
        (pl.col("concept_class_id") == INGREDIENT) & pl.col("vocabulary_id").is_in(list(RXNORM))
    ).sort("concept_id")
    return tuple(zip(found["concept_id"].to_list(), found["concept_name"].to_list()))


def check_categories(concept: pl.DataFrame, categories: Sequence[Tuple[int, str]]) -> None:
    """Every category must exist in this vocabulary version under the name given here."""
    names = dict(zip(concept["concept_id"].to_list(), concept["concept_name"].to_list()))
    for concept_id, name in categories:
        if names.get(concept_id) != name:
            raise ValueError(
                f"Category {concept_id} is {names.get(concept_id)!r} in the concept table, "
                f"expected {name!r}; the vocabulary version may differ"
            )


def build_universe(
    concept: pl.DataFrame,
    ancestor: pl.DataFrame,
    token_ids: Iterable[int],
    domain: str,
    categories: Optional[Sequence[Tuple[int, str]]] = None,
    vocabularies: Optional[Sequence[str]] = None,
    ancestor_classes: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    The questions of one domain: one row per concept.

    Columns: concept_id, concept_name, vocabulary_id, is_token (the concept is in the tokenizer),
    categories (the top level categories at or above the concept, a list of concept ids; empty for
    a token below no category), depth (levels below its closest category, None without one or for a
    token outside SNOMED),
    num_tokens (tokens at or below the concept, itself included).
    """
    if categories is None:
        categories = CATEGORIES[domain] if CATEGORIES[domain] is not None else ingredient_categories(concept)
    categories = tuple(categories)
    vocabularies = tuple(vocabularies if vocabularies is not None else VOCABULARIES.get(domain, (SNOMED,)))
    if ancestor_classes is None:
        ancestor_classes = ANCESTOR_CLASSES.get(domain)
    check_categories(concept, categories)
    category_ids = [c for c, _ in categories]

    token_ids = set(token_ids)
    tokens = concept.filter((pl.col("domain_id") == domain) & pl.col("concept_id").is_in(list(token_ids)))
    token_list = tokens["concept_id"].to_list()
    snomed = concept.filter(pl.col("vocabulary_id").is_in(list(vocabularies)))["concept_id"].to_list()

    # node -> the categories at or above it, with the distance; a category is at distance 0
    below = ancestor.filter(
        pl.col("ancestor_concept_id").is_in(category_ids) & pl.col("descendant_concept_id").is_in(snomed)
    ).select(
        pl.col("ancestor_concept_id").alias("category"),
        pl.col("descendant_concept_id").alias("node"),
        pl.col("min_levels_of_separation").alias("distance"),
    )
    # every (token, node) pair where node is the token or an ancestor of it
    pairs = ancestor.filter(
        pl.col("descendant_concept_id").is_in(token_list) & pl.col("ancestor_concept_id").is_in(snomed)
    ).select(
        pl.col("descendant_concept_id").alias("token"),
        pl.col("ancestor_concept_id").alias("node"),
    )
    own = pl.DataFrame({"token": token_list, "node": token_list}, schema={"token": pl.Int64, "node": pl.Int64})
    members = pl.concat([pairs, own]).unique()
    # an ancestor is a question only if it is a category or below one; a token always is
    in_a_category = set(below["node"].to_list())
    if ancestor_classes is not None:
        # only ancestors of these classes are questions (the ingredient of a drug, not its forms)
        allowed = concept.filter(pl.col("concept_class_id").is_in(list(ancestor_classes)))["concept_id"].to_list()
        in_a_category &= set(allowed)
    members = members.filter(pl.col("node").is_in(list(in_a_category)) | (pl.col("node") == pl.col("token")))

    sizes = members.group_by("node").agg(pl.col("token").n_unique().alias("num_tokens"))
    per_node = below.group_by("node").agg(
        pl.col("category").sort().alias("categories"),
        pl.col("distance").min().alias("depth"),
    )
    # a token outside SNOMED (CPT4, ICD10PCS, ...) is below no SNOMED category itself, but its SNOMED
    # ancestors are, so a token takes the categories of itself and of its ancestors
    from_ancestors = (
        members.join(per_node, on="node")
        .select("token", "categories")
        .explode("categories")
        .group_by("token")
        .agg(pl.col("categories").unique().sort().alias("token_categories"))
        .rename({"token": "concept_id"})
    )
    universe = (
        sizes.rename({"node": "concept_id"})
        .join(per_node.rename({"node": "concept_id"}), on="concept_id", how="left")
        .join(from_ancestors, on="concept_id", how="left")
        .with_columns(
            pl.coalesce("token_categories", "categories").fill_null([]).cast(pl.List(pl.Int64)).alias("categories"),
            pl.col("concept_id").is_in(token_list).alias("is_token"),
        )
        .join(concept.select("concept_id", "concept_name", "vocabulary_id"), on="concept_id", how="left")
        .select("concept_id", "concept_name", "vocabulary_id", "is_token", "categories", "depth", "num_tokens")
        .sort("num_tokens", descending=True)
    )
    return universe


def _scan(vocab_path: str, name: str) -> pl.LazyFrame:
    return pl.scan_parquet(glob.glob(os.path.join(vocab_path, name, "*.parquet")))


def read_token_ids(tokenizer_dirs: Sequence[str]) -> Set[int]:
    from tokenizers import Tokenizer

    ids: Set[int] = set()
    for folder in tokenizer_dirs:
        vocab = Tokenizer.from_file(os.path.join(folder, "cehrgpt_tokenizer.json")).get_vocab()
        ids |= {int(token) for token in vocab if token.isdigit()}
    return ids


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, pl.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab_path", required=True, help="OHDSI folder with concept/ and concept_ancestor/ (parquet)")
    parser.add_argument("--tokenizers", nargs="+", required=True, help="Tokenizer folders; their concepts are united")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domains", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    concept = _scan(args.vocab_path, "concept").select(
        pl.col("concept_id").cast(pl.Int64), "concept_name", "domain_id", "vocabulary_id", "concept_class_id"
    ).collect()
    ancestor = _scan(args.vocab_path, "concept_ancestor").select(
        pl.col("ancestor_concept_id").cast(pl.Int64), pl.col("descendant_concept_id").cast(pl.Int64),
        pl.col("min_levels_of_separation").cast(pl.Int64),
    ).collect()
    token_ids = read_token_ids(args.tokenizers)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {}
    for domain in args.domains:
        universe = build_universe(concept, ancestor, token_ids, domain)
        result[domain] = universe
        universe.write_parquet(os.path.join(args.output_dir, f"{domain.lower()}_universe.parquet"))
        universe.with_columns(pl.col("categories").list.eval(pl.element().cast(pl.String)).list.join(";")).write_csv(
            os.path.join(args.output_dir, f"{domain.lower()}_universe.csv")
        )
        uncovered = universe.filter(pl.col("is_token") & (pl.col("categories").list.len() == 0))
        LOG.info(
            "%s: %s questions, %s tokens, %s ancestors only; %s tokens are below no category",
            domain, len(universe), universe["is_token"].sum(), (~universe["is_token"]).sum(), len(uncovered),
        )
    return result


if __name__ == "__main__":
    main()
