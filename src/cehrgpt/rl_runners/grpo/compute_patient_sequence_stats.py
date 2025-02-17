from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window

from cehrgpt.tools.generate_causal_patient_split_by_age import age_group_func


@udf(ArrayType(StringType()))
def unique_concepts(sequence):
    return list(set([concept for concept in sequence if concept.isnumeric()]))


@udf(StringType())
def create_age_group_udf(age_str):
    return age_group_func(age_str)


def compute_marginal(dataframe, num_of_partitions):
    all_concepts_dataframe = (
        dataframe.withColumn("unique_concepts", unique_concepts("concept_ids"))
        .select(f.explode("unique_concepts").alias("concept_id"))
        .drop("unique_concepts")
    )
    marginal_dist = all_concepts_dataframe.groupBy("concept_id").count()
    data_size = all_concepts_dataframe.count()
    marginal_dist = marginal_dist.withColumn(
        "prob", f.col("count") / f.lit(data_size)
    ).withColumn("concept_order", f.row_number().over(Window.orderBy(f.desc("prob"))))
    num_of_concepts = marginal_dist.count()
    partition_size = num_of_concepts // num_of_partitions
    marginal_dist = (
        marginal_dist.withColumn(
            "concept_partition",
            f.floor(f.col("concept_order") / f.lit(partition_size)).cast("int")
            + f.lit(1),
        )
        .withColumn(
            "concept_partition",
            f.when(
                f.col("concept_partition") > num_of_partitions, num_of_partitions
            ).otherwise(f.col("concept_partition")),
        )
        .drop("concept_order")
    )
    return marginal_dist


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
        f.col("num_of_concepts").between(
            args.min_num_of_concepts, args.max_num_of_concepts
        )
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
