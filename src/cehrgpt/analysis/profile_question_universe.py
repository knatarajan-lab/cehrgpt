"""Profile the concepts of tokenizers against the OHDSI hierarchy, to choose how far up to go.

python profile_universe.py --vocab_path OHDSI_DIR --tokenizers OLD NEW
OHDSI_DIR has concept/ and concept_ancestor/ (parquet folders). Prints, per tokenizer, the
concepts by vocabulary and class, then for the Condition and Procedure domains how many groups
and tokens there are at every depth below the root (the Disease / Procedure concept).
"""
import argparse, os, glob
import numpy as np
import polars as pl
from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer


def scan(vocab_path, name):
    files = glob.glob(os.path.join(vocab_path, name, "*.parquet")) or glob.glob(os.path.join(vocab_path, name + "*"))
    return pl.scan_parquet(files) if files[0].endswith("parquet") else pl.scan_csv(files, separator="\t")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_path", required=True)
    p.add_argument("--tokenizers", nargs="+", required=True)
    p.add_argument("--condition_root", default="Disease")
    p.add_argument("--procedure_root", default="Procedure")
    p.add_argument("--condition_root_code", default=None, help="SNOMED concept_code of the condition root (64572001 is Disease); used instead of the name")
    p.add_argument("--procedure_root_code", default=None, help="SNOMED concept_code of the procedure root; used instead of the name")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--examples", nargs="*", default=[], help="Names (substrings) of concepts whose ancestor chain is printed, e.g. 'Coronary artery bypass'")
    a = p.parse_args()
    concept = scan(a.vocab_path, "concept").select(
        pl.col("concept_id").cast(pl.Int64), "concept_name", "domain_id", "vocabulary_id", "concept_class_id", "concept_code"
    ).collect()
    anc = scan(a.vocab_path, "concept_ancestor").select(
        pl.col("ancestor_concept_id").cast(pl.Int64), pl.col("descendant_concept_id").cast(pl.Int64),
        pl.col("min_levels_of_separation").cast(pl.Int64),
    ).collect()
    for path in a.tokenizers:
        tokens = [int(t) for t in CehrGptTokenizer.from_pretrained(path).get_vocab() if t.isdigit()]
        mine = concept.filter(pl.col("concept_id").is_in(tokens))
        print(f"\n===== {path}: {len(tokens)} numeric tokens, {len(mine)} in the concept table")
        print(mine.group_by("domain_id", "vocabulary_id", "concept_class_id").len().sort("len", descending=True).head(25))
        for domain, root_name, vocab, root_code in (
            ("Condition", a.condition_root, "SNOMED", a.condition_root_code),
            ("Procedure", a.procedure_root, "SNOMED", a.procedure_root_code),
        ):
            if root_code:
                roots = concept.filter((pl.col("concept_code").cast(pl.String) == root_code) & (pl.col("vocabulary_id") == vocab)).sort("concept_id")
            else:
                roots = concept.filter((pl.col("concept_name") == root_name) & (pl.col("vocabulary_id") == vocab)).sort("concept_id")
            if roots.is_empty():
                print(f"\n-- {domain}: no {vocab} concept named {root_name!r}"); continue
            root = roots["concept_id"][0]
            ids = mine.filter(pl.col("domain_id") == domain)["concept_id"].to_list()
            below = anc.filter(pl.col("ancestor_concept_id") == root).select(
                pl.col("descendant_concept_id").alias("node"), pl.col("min_levels_of_separation").alias("depth"))
            under = anc.filter(pl.col("ancestor_concept_id").is_in(below["node"].to_list()) & pl.col("descendant_concept_id").is_in(ids))
            print(f"\n-- {domain}: root {root_name!r} ({root}); {len(ids)} tokens, {under['descendant_concept_id'].n_unique()} below the root")
            print(" depth  groups  tokens_covered  tokens_per_group min/median/max   largest group")
            for d in range(1, a.max_depth + 1):
                nodes = below.filter(pl.col("depth") == d)["node"].to_list()
                size = (under.filter(pl.col("ancestor_concept_id").is_in(nodes)).group_by("ancestor_concept_id")
                        .agg(pl.col("descendant_concept_id").n_unique().alias("n")))
                if size.is_empty(): print(f" {d:>5}  0"); continue
                covered = under.filter(pl.col("ancestor_concept_id").is_in(nodes))["descendant_concept_id"].n_unique()
                top = size.sort("n", descending=True).head(1).join(concept, left_on="ancestor_concept_id", right_on="concept_id")
                n = size["n"].to_numpy()
                print(f" {d:>5}  {len(nodes):>6}  {covered:>14}  {n.min():>6}/{int(np.median(n))}/{n.max()}   {top['concept_name'][0][:40]} ({top['n'][0]})")
        for example in a.examples:
            found = mine.filter(pl.col("concept_name").str.to_lowercase().str.contains(example.lower(), literal=True)).head(3)
            if found.is_empty():
                print(f"\n-- example {example!r}: no token of this tokenizer has this name"); continue
            for cid, cname in zip(found["concept_id"], found["concept_name"]):
                print(f"\n-- ancestors of {cname!r} ({cid}); tokens = tokens of this tokenizer at or below the ancestor")
                chain = anc.filter(pl.col("descendant_concept_id") == cid).join(
                    concept.select(pl.col("concept_id").alias("ancestor_concept_id"), "concept_name", "vocabulary_id"), on="ancestor_concept_id")
                below_count = (anc.filter(pl.col("ancestor_concept_id").is_in(chain["ancestor_concept_id"].to_list())
                                          & pl.col("descendant_concept_id").is_in(tokens))
                               .group_by("ancestor_concept_id").agg(pl.col("descendant_concept_id").n_unique().alias("tokens")))
                chain = chain.join(below_count, on="ancestor_concept_id", how="left").sort("min_levels_of_separation")
                for row in chain.iter_rows(named=True):
                    print(f"  {row['min_levels_of_separation']:>3} up  {row['concept_name'][:60]:<60} {row['vocabulary_id']:<14} tokens={row['tokens']}")
        drugs = mine.filter(pl.col("domain_id") == "Drug")
        ing = anc.join(concept.filter(pl.col("concept_class_id") == "Ingredient").select(pl.col("concept_id").alias("ancestor_concept_id")), on="ancestor_concept_id")
        has = ing.filter(pl.col("descendant_concept_id").is_in(drugs["concept_id"].to_list()))["descendant_concept_id"].n_unique()
        print(f"\n-- Drug: {len(drugs)} tokens, {has} with an Ingredient ancestor or are an ingredient")
        print(drugs.group_by("vocabulary_id", "concept_class_id").len().sort("len", descending=True).head(10))


if __name__ == "__main__":
    main()
