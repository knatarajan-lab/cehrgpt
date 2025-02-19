import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from .compute_patient_sequence_co_occurrence import (
    create_age_group_udf,
    create_arg_parser,
)

concept_prevalence_stats_name = "patient_concept_stats"


def main(args):
    spark = SparkSession.builder.appName(
        "Compute the patient marginal concept probability"
    ).getOrCreate()
    patient_sequence = spark.read.parquet(args.patient_events_dir)
    if args.use_sample:
        patient_sequence = patient_sequence.sample(args.sample_frac)
    patient_sequence = (
        patient_sequence.withColumn(
            "age_group", create_age_group_udf(f.col("concept_ids")[1])
        )
        .withColumn("gender", f.col("concept_ids")[2])
        .withColumn("race", f.col("concept_ids")[3])
        .withColumn(
            "concept_id", f.explode(f.slice("concept_ids", 5, f.size("concept_ids")))
        )
    )

    demographic_concept_stats = patient_sequence.groupBy(
        f.col("age_group"), f.col("gender"), f.col("race"), f.col("concept_id")
    ).count()

    demographic_concept_total = demographic_concept_stats.groupby(
        "age_group", "gender", "race"
    ).agg(f.sum("count").alias("total"))

    demographic_concept_stats = (
        demographic_concept_stats.join(
            demographic_concept_total, ["age_group", "gender", "race"]
        )
        .withColumn("prob", f.col("count") / f.col("total"))
        .drop("total")
    )

    demographic_concept_stats.write.mode("overwrite").parquet(
        os.path.join(args.output_dir, concept_prevalence_stats_name)
    )


if __name__ == "__main__":
    main(
        create_arg_parser(
            "Arguments for generating the patient sequence length stats"
        ).parse_args()
    )
