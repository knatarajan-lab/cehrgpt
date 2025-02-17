from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from cehrgpt.tools.generate_causal_patient_split_by_age import age_group_func


@udf(StringType())
def create_age_group_udf(age_str):
    return age_group_func(age_str)


def main(args):
    spark = SparkSession.builder.appName(
        "Compute the temporal co-occurrence matrix"
    ).getOrCreate()

    time_window = args.time_window if args.time_window is not None else 1_000_000
    output_dir = (
        f"{args.output_dir}_{args.time_window}"
        if args.time_window is None
        else f"{args.output_dir}_lifetime"
    )
    patient_events = spark.read.parquet(args.patient_events_dir)
    patient_events = patient_events.where(
        f.col("num_of_concepts").between(args.min_num_concepts, args.max_num_concepts)
    )
    if args.use_sample:
        patient_events = patient_events.sample(args.sample_frac)

    patient_events = (
        patient_events.withColumn("year", f.element_at("concept_ids", 1))
        .withColumn("age_group", create_age_group_udf(f.element_at("concept_ids", 2)))
        .withColumn("gender", f.element_at("concept_ids", 3))
        .withColumn("race", f.element_at("concept_ids", 4))
    )

    co_occurrence_raw_count_by_demographic_group = (
        patient_events.alias("past")
        .join(
            patient_events.alias("future"),
            f.col("past.person_id") == f.col("future.person_id"),
        )
        .where(f.col("future.datetime") > f.col("past.datetime"))
        .where(f.datediff(f.col("future.date"), f.col("past.date")) <= time_window)
        .groupBy(
            f.col("past.year").alias("year"),
            f.col("past.age_group").alias("age_group"),
            f.col("past.gender").alias("gender"),
            f.col("past.race").alias("race"),
            f.col("past.standard_concept_id").alias("concept_id_1"),
            f.col("future.standard_concept_id").alias("concept_id_2"),
        )
        .count()
    )
    co_occurrence_raw_count_by_demographic_group = (
        co_occurrence_raw_count_by_demographic_group.crossJoin(
            co_occurrence_raw_count_by_demographic_group.select(
                f.sum("count").alias("total")
            )
        )
        .withColumn("prob", f.col("count") / f.col("total"))
        .drop("total")
    )

    co_occurrence_raw_count_by_demographic_group.write.mode("overwrite").parquet(
        output_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for generating the co-occurrence matrix"
    )

    parser.add_argument(
        "--patient_events_dir",
        dest="patient_events_dir",
        action="store",
        help="The path for patient events data",
        required=True,
    )
    parser.add_argument(
        "--time_window", dest="time_window", action="store", type=int, required=False
    )
    parser.add_argument(
        "--max_num_concepts",
        dest="max_num_concepts",
        action="store",
        type=int,
        default=1024,
        required=False,
    )
    parser.add_argument(
        "--min_num_concepts",
        dest="min_num_concepts",
        action="store",
        type=int,
        default=20,
        required=False,
    )
    parser.add_argument("--use_sample", dest="use_sample", action="store_true")
    parser.add_argument(
        "--sample_frac",
        dest="sample_frac",
        action="store",
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--output_dir", dest="output_dir", action="store", required=True
    )
    main(parser.parse_args())
