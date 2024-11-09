import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.functions import udf


@udf(t.ArrayType(t.StringType()))
def filter_concepts(sequence):
    return [concept for concept in sequence if concept.isnumeric()]


@udf(t.FloatType())
def kl_divergence_udf(prob1, prob2):
    return prob1 * np.log(prob1 / (prob2 + 1e-10))


def calculate_dist(dataframe: DataFrame) -> DataFrame:
    concept_dataframe = dataframe.withColumn(
        "concept_ids", filter_concepts("concept_ids")
    ).select(f.explode("concept_ids").alias("concept_id"))
    row_count = concept_dataframe.count()
    return (
        concept_dataframe.groupby("concept_id")
        .count()
        .withColumn("prob", f.col("count") / f.lit(row_count))
        .drop("count")
    )


def main(args):
    spark = SparkSession.builder.appName("Calculate KL divergence").getOrCreate()

    reference_sequence = spark.read.parquet(args.reference_sequence_path)
    comparison_sequence = spark.read.parquet(args.comparison_sequence_path)

    reference_dist = calculate_dist(reference_sequence)
    comparison_dist = calculate_dist(comparison_sequence)

    join_conditions = reference_dist.concept_id == comparison_dist.concept_id
    columns = [
        reference_dist.prob.alias("reference_prob"),
        f.coalesce(comparison_dist.prob, f.lit(1e-10)).alias("prob"),
    ]

    joined_results = (
        reference_dist.join(comparison_dist, join_conditions, "left_outer")
        .select(columns)
        .withColumn(
            "kl",
            f.col("reference_prob") * f.log(f.col("reference_prob") / f.col("prob")),
        )
    )
    joined_results.select(f.bround(f.sum("kl"), 4)).show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for calculating the KL divergent between concept distributions of the two patient sequences"
    )
    parser.add_argument(
        "--reference_sequence_path",
        dest="reference_sequence_path",
        action="store",
        help="The path for reference data sequence",
        required=True,
    )
    parser.add_argument(
        "--comparison_sequence_path",
        dest="comparison_sequence_path",
        action="store",
        help="The path for comparison data sequence",
        required=True,
    )
    main(parser.parse_args())
