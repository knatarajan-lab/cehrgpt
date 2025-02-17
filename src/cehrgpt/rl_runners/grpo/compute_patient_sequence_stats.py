import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


# Copied over from cehrgpt.tools.generate_causal_patient_split_by_age because spark requires all the functions
# to be self-contained unless they exist in the spark venv.
@udf(StringType())
def create_age_group_udf(age_str):
    age = int(age_str.split(":")[1])
    group_number = age // 10
    return f"age:{group_number * 10}-{(group_number + 1) * 10}"


def main(args):
    spark = SparkSession.builder.appName(
        "Compute the temporal co-occurrence matrix"
    ).getOrCreate()
    time_window = args.time_window if args.time_window else 1_000_000
    output_dir = (
        os.path.join(args.output_dir, f"co_occurrence_{args.time_window}")
        if args.time_window
        else os.path.join(args.output_dir, f"co_occurrence_lifetime")
    )
    patient_events = spark.read.parquet(
        os.path.join(args.patient_events_dir, "all_patient_events")
    )
    start_age_events = spark.read.parquet(
        os.path.join(
            args.patient_events_dir, "demographic_events", "sequence_age_tokens"
        )
    ).select(
        "person_id",
        create_age_group_udf(f.col("standard_concept_id")).alias("age_group"),
    )
    race_events = spark.read.parquet(
        os.path.join(
            args.patient_events_dir, "demographic_events", "sequence_race_tokens"
        )
    ).select("person_id", f.col("standard_concept_id").alias("race"))
    gender_events = spark.read.parquet(
        os.path.join(
            args.patient_events_dir, "demographic_events", "sequence_gender_tokens"
        )
    ).select("person_id", f.col("standard_concept_id").alias("gender"))

    patient_events = (
        patient_events.join(start_age_events, "person_id")
        .join(race_events, "person_id")
        .join(gender_events, "person_id")
    )

    qualified_persons = (
        patient_events.groupBy("person_id")
        .count()
        .where(f.col("count").between(args.min_num_concepts, args.max_num_concepts))
        .select("person_id")
    )
    patient_events = patient_events.join(qualified_persons, "person_id")
    if args.use_sample:
        patient_events = patient_events.sample(args.sample_frac)

    co_occurrence_raw_count_by_demographic_group = (
        patient_events.alias("past")
        .join(
            patient_events.alias("future"),
            f.col("past.person_id") == f.col("future.person_id"),
        )
        .where(f.col("future.datetime") > f.col("past.datetime"))
        .where(f.datediff(f.col("future.date"), f.col("past.date")) <= time_window)
        .groupBy(
            f.col("past.age_group").alias("age_group"),
            f.col("past.gender").alias("gender"),
            f.col("past.race").alias("race"),
            f.col("past.standard_concept_id").alias("concept_id_1"),
            f.col("future.standard_concept_id").alias("concept_id_2"),
        )
        .count()
    )
    demographic_group_total = co_occurrence_raw_count_by_demographic_group.groupBy(
        "age_group", "gender", "race"
    ).agg(f.sum("count").alias("total"))
    co_occurrence_raw_count_by_demographic_group = (
        co_occurrence_raw_count_by_demographic_group.join(
            demographic_group_total,
            ["age_group", "gender", "race"],
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
        "--time_window",
        dest="time_window",
        action="store",
        type=int,
        default=None,
        required=False,
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
