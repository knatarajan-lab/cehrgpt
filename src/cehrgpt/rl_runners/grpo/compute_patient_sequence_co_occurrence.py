import os
import re

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType

temporal_co_occurrence_stats_name = "temporal_cooccurrence_matrices"
# Compile the regex pattern once for better performance
_CO_OCCURRENCE_PATTERN = re.compile(r"co_occurrence_\d+_(?:\d+|lifetime)/?$")


def is_co_occurrence_folder(folder_name: str) -> bool:
    """
    Check if a folder name matches the co-occurrence pattern.

    Args:
        folder_name: String to check against the co-occurrence pattern

    Returns:
        bool: True if the name matches the pattern, False otherwise
    """
    return _CO_OCCURRENCE_PATTERN.match(os.path.basename(folder_name)) is not None


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
    time_window_start = args.time_window_start if args.time_window_start else 0
    time_window = args.time_window if args.time_window else 1_000_000_000
    output_dir = os.path.join(args.output_dir, temporal_co_occurrence_stats_name)
    output_dir = (
        os.path.join(
            output_dir,
            f"co_occurrence_{time_window_start}_{time_window_start + args.time_window}",
        )
        if args.time_window
        else os.path.join(output_dir, f"co_occurrence_{time_window_start}_lifetime")
    )
    patient_events = spark.read.parquet(
        os.path.join(args.patient_events_dir, "clinical_events")
    ).select("person_id", "standard_concept_id", "date", "datetime", "priority")
    visit_type_tokens = (
        spark.read.parquet(
            os.path.join(args.patient_events_dir, "att_events", "artificial_tokens")
        )
        .where(
            f.col("standard_concept_id").isin(["[VS]", "[VE]"])
            | f.col("standard_concept_id").cast(IntegerType()).isNotNull()
        )
        .select("person_id", "standard_concept_id", "date", "datetime", "priority")
    )
    patient_events = patient_events.unionByName(visit_type_tokens)
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

    # Assuming 'artificial_tokens' is your DataFrame
    patient_events = patient_events.withColumn(
        "datetime",
        f.from_unixtime(
            f.unix_timestamp("datetime") + (f.col("priority") / 1000),
            "yyyy-MM-dd HH:mm:ss.SSS",
        ),
    ).drop("priority")

    co_occurrence_raw_count_by_demographic_group = (
        patient_events.alias("past")
        .join(
            patient_events.alias("future"),
            f.col("past.person_id") == f.col("future.person_id"),
        )
        .where(f.col("future.datetime") > f.col("past.datetime"))
        .where(
            f.datediff(f.col("future.date"), f.col("past.date")) >= time_window_start
        )
        .where(
            f.datediff(f.col("future.date"), f.col("past.date"))
            < time_window_start + time_window
        )
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


def create_arg_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--patient_events_dir",
        dest="patient_events_dir",
        action="store",
        help="The path for patient events data",
        required=True,
    )
    parser.add_argument(
        "--time_window_start",
        dest="time_window_start",
        action="store",
        type=int,
        default=0,
        required=False,
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
    return parser


if __name__ == "__main__":
    main(
        create_arg_parser(
            "Arguments for generating the co-occurrence matrix"
        ).parse_args()
    )
