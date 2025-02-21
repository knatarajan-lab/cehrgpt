import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from cehrgpt.rl_runners.grpo.compute_patient_sequence_concept_prevalence import (
    create_arg_parser,
)

conditional_prob_distribution_name = "concept_conditional_prob_distribution"


# Copied over from cehrgpt.tools.generate_causal_patient_split_by_age because spark requires all the functions
# to be self-contained unless they exist in the spark venv.
@f.udf(t.StringType())
def create_age_group_udf(age):
    group_number = age // 10
    return f"age:{group_number * 10}-{(group_number + 1) * 10}"


def main(args):
    spark = SparkSession.builder.appName(
        "Compute the conditional concept probability distribution"
    ).getOrCreate()
    patient_sequence = spark.read.parquet(args.patient_events_dir)
    if args.use_sample:
        patient_sequence = patient_sequence.sample(args.sample_frac)

    patient_sequence = (
        patient_sequence.select(
            f.concat(f.array(f.lit("[START]")), f.col("concept_ids")).alias(
                "concept_ids"
            ),
            f.concat(f.array(f.col("ages")[0]), f.col("ages")).alias("ages"),
            f.concat(
                f.array(f.col("visit_concept_ids")[0]), f.col("visit_concept_ids")
            ).alias("visit_concept_ids"),
        )
        .withColumn(
            "shifted_concept_ids",
            f.concat(f.slice("concept_ids", 2, 100_000_000), f.array(f.lit("[END]"))),
        )
        .withColumn(
            "concept_pairs",
            f.arrays_zip(
                "ages", "visit_concept_ids", "concept_ids", "shifted_concept_ids"
            ),
        )
    )

    concept_pair = (
        patient_sequence.select(f.explode("concept_pairs").alias("concept_pair"))
        .withColumn("age_group", create_age_group_udf(f.col("concept_pair.ages")))
        .withColumn(
            "visit_concept_id",
            f.col("concept_pair.visit_concept_ids"),
        )
        .withColumn("concept_id_1", f.col("concept_pair.concept_ids"))
        .withColumn("concept_id_2", f.col("concept_pair.shifted_concept_ids"))
        .drop("concept_pair")
    )

    group_total = (
        concept_pair.groupby("visit_concept_id", "age_group")
        .count()
        .withColumnRenamed("count", "total")
    )
    concept_pair_prob = (
        concept_pair.groupby(
            "concept_id_1", "concept_id_2", "visit_concept_id", "age_group"
        )
        .count()
        .join(group_total, ["visit_concept_id", "age_group"])
        .withColumn("prob", f.col("count") / f.col("total"))
    )

    concept_pair_prob.write.mode("overwrite").parquet(
        os.path.join(args.output_dir, conditional_prob_distribution_name)
    )


if __name__ == "__main__":
    main(
        create_arg_parser(
            "Arguments for generating conditional concept probability distribution"
        ).parse_args()
    )
