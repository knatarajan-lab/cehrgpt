import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def main(args):
    spark = SparkSession.builder.appName("Create the reward dataset").getOrCreate()
    real_data = spark.read.parquet(args.real_data_dir)
    real_data = (
        real_data.sample(args.real_sample_frac)
        .select("concept_ids")
        .withColumn("label", f.lit(1))
    )
    synthetic_data = spark.read.parquet(args.synthetic_data_dir)
    synthetic_data = (
        synthetic_data.sample(args.synthetic_sample_frac)
        .select("concept_ids")
        .withColumn("label", f.lit(0))
    )
    real_data.unionByName(synthetic_data).write.mode("overwrite").parquet(
        os.path.join(args.output_dir, "reward_dataset")
    )


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Create reward dataset")
    parser.add_argument(
        "--real_data_dir",
    )
    parser.add_argument(
        "--synthetic_data_dir",
    )
    parser.add_argument(
        "--real_sample_frac",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--synthetic_sample_frac",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--output_dir",
    )
    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
