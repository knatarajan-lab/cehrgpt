import argparse
import logging
import os

from cehrbert_data.utils.logging_utils import add_console_logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

add_console_logging()
logger = logging.getLogger(__name__)


def main(
    real_sequence_folder: str,
    synthetic_sequence_folder: str,
    output_folder: str,
    num_patients: int,
):
    spark = SparkSession.builder.appName(
        "Create the DPO training dataset"
    ).getOrCreate()

    logger.info(
        f"real_sequence_folder: {real_sequence_folder}\n"
        f"synthetic_sequence_folder: {synthetic_sequence_folder}\n"
        f"output_folder: {output_folder}\n"
    )

    if not os.path.exists(real_sequence_folder):
        raise RuntimeError(
            f"Real patient sequences must exist in {real_sequence_folder}"
        )

    if not os.path.exists(synthetic_sequence_folder):
        raise RuntimeError(
            f"Synthetic patient sequences must exist in {synthetic_sequence_folder}"
        )

    real_sequences = spark.read.parquet(real_sequence_folder)
    synthetic_sequences = spark.read.parquet(synthetic_sequence_folder)

    real_sequences_sample = (
        real_sequences.sample(float(num_patients) / real_sequences.count(), seed=42)
        .withColumn("id", f.monotonically_increasing_id())
        .select(
            f.col("id").alias("chosen_id"),
            f.col("concept_ids").alias("chosen_concept_id"),
            f.col("concept_values").alias("chosen_concept_value"),
            f.col("concept_value_masks").alias("chosen_concept_value_mask"),
        )
    )
    synthetic_sequences_sample = (
        synthetic_sequences.sample(
            float(num_patients) / synthetic_sequences.count(), seed=42
        )
        .withColumn("id", f.monotonically_increasing_id())
        .select(
            f.col("id").alias("rejected_id"),
            f.col("concept_ids").alias("rejected_concept_id"),
            f.col("concept_values").alias("rejected_concept_value"),
            f.col("concept_value_masks").alias("rejected_concept_value_mask"),
        )
    )

    dataset = real_sequences_sample.join(
        synthetic_sequences_sample,
        real_sequences_sample.chosen_id == synthetic_sequences.rejected_id,
    ).drop("chosen_id", "rejected_id")

    dataset.write.mode("overwrite").parquet(output_folder)


def create_app_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Arguments for generate training data for Bert"
    )
    parser.add_argument(
        "--real_sequence_folder",
        dest="real_sequence_folder",
        action="store",
        help="The path for your input_folder where the Real OMOP folder is",
        required=True,
    )
    parser.add_argument(
        "--synthetic_sequence_folder",
        dest="synthetic_sequence_folder",
        action="store",
        help="The path for your input_folder where the Synthetic OMOP folder is",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--num_patients",
        dest="num_patients",
        action="store",
        type=int,
        required=True,
    )
    return parser


if __name__ == "__main__":
    ARGS = create_app_arg_parser().parse_args()
    main(
        ARGS.real_sequence_folder,
        ARGS.synthetic_sequence_folder,
        ARGS.output_folder,
        ARGS.num_patients,
    )
