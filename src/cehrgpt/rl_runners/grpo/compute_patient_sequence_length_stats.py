import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from .compute_patient_sequence_co_occurrence import (
    create_age_group_udf,
    create_arg_parser,
)


def main(args):
    spark = SparkSession.builder.appName(
        "Compute the patient sequence length stats"
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
        .withColumn("log_patient_length", f.log(f.size("concept_ids")))
    )

    patient_seq_length_stats = patient_sequence.groupBy(
        f.col("age_group"),
        f.col("gender"),
        f.col("race"),
    ).agg(
        f.mean("log_patient_length").alias("log_mean"),
        f.stddev("log_patient_length").alias("log_std"),
    )
    patient_seq_length_stats.write.mode("overwrite").parquet(
        os.path.join(args.output_dir, "patient_seq_length_stats")
    )


if __name__ == "__main__":
    main(
        create_arg_parser(
            "Arguments for generating the patient sequence length stats"
        ).parse_args()
    )
