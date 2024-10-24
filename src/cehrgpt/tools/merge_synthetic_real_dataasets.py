import argparse
import logging
import os
import shutil
from enum import Enum
from typing import List

from cehrbert_data.utils.logging_utils import add_console_logging
from cehrbert_data.utils.spark_utils import validate_table_names
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

add_console_logging()
logger = logging.getLogger(__name__)

COHORT_FOLDER_NAME = "cohorts"


class MergeType(Enum):
    TRAIN_AND_TEST = "train_and_test"
    TEST_ONLY = "test_only"


def main(
    real_omop_folder: str,
    synthetic_omop_folder: str,
    domain_table_list: List[str],
    output_folder: str,
    merge_type: str,
):
    spark = SparkSession.builder.appName(
        "Merge Synthetic OMOP and Real OMOP datasets"
    ).getOrCreate()

    logger.info(
        f"real_omop_folder: {real_omop_folder}\n"
        f"synthetic_omop_folder: {synthetic_omop_folder}\n"
        f"output_folder: {output_folder}\n"
        f"domain_table_list: {domain_table_list}\n"
        f"merge_type: {merge_type}\n"
    )

    patient_splits_folder = os.path.join(real_omop_folder, "patient_splits")
    if not os.path.exists(patient_splits_folder):
        raise RuntimeError(f"patient_splits must exist in {real_omop_folder}")

    patient_splits = spark.read.parquet(patient_splits_folder)
    patient_splits = patient_splits.select("person_id", "split")

    # Generate the real patient splits
    real_person = spark.read.parquet(os.path.join(real_omop_folder, "person"))
    real_person = (
        real_person.select("person_id")
        .join(patient_splits, "person_id")
        .withColumn("is_real", F.lit(1))
    )
    max_real_person_id = real_person.select(F.max("person_id")).collect()[0][0]
    if merge_type == MergeType.TEST_ONLY.value:
        real_person = real_person.where("split='test'")
    synthetic_person = spark.read.parquet(os.path.join(synthetic_omop_folder, "person"))
    synthetic_person = (
        synthetic_person.select("person_id")
        .withColumn("split", F.lit("train"))
        .withColumn("is_real", F.lit(0))
    )
    merge_patient_splits = real_person.unionByName(synthetic_person)
    merge_patient_splits = merge_patient_splits.withColumn(
        "new_person_id",
        F.when(F.col("is_real") == 1, F.col("person_id")).otherwise(
            F.col("person_id") + F.lit(max_real_person_id)
        ),
    )
    merge_patient_splits.cache()

    # re-assign visit_occurrence_id
    real_visit_occurrence = spark.read.parquet(
        os.path.join(real_omop_folder, "visit_occurrence")
    )
    max_real_visit_occurrence_id = real_visit_occurrence.select(
        F.max("visit_occurrence_id")
    ).collect()[0][0]

    synthetic_visit_occurrence_mapping = (
        spark.read.parquet(os.path.join(synthetic_omop_folder, "visit_occurrence"))
        .select("visit_occurrence_id")
        .withColumn(
            "new_visit_occurrence_id",
            F.col("visit_occurrence_id") + F.lit(max_real_visit_occurrence_id),
        )
    )

    for domain_table in domain_table_list:
        real_domain_table = spark.read.parquet(
            os.path.join(real_omop_folder, domain_table)
        )
        synthetic_domain_table = spark.read.parquet(
            os.path.join(synthetic_omop_folder, domain_table)
        )
        # The synthetic and real datasets should have the same schema, this is the pre-caution to make sure the columns
        # exist in both datasets
        real_columns = real_domain_table.schema.fieldNames()
        synthetic_columns = synthetic_domain_table.schema.fieldNames()
        common_columns = [f for f in synthetic_columns if f in real_columns]

        real_domain_table = real_domain_table.join(
            merge_patient_splits,
            (real_domain_table["person_id"] == merge_patient_splits["person_id"])
            & (merge_patient_splits["is_real"] == 1),
        ).select(
            [real_domain_table[f] for f in common_columns]
            + [
                merge_patient_splits["is_real"],
                merge_patient_splits["split"],
                merge_patient_splits["new_person_id"],
            ]
        )

        synthetic_domain_table = synthetic_domain_table.join(
            merge_patient_splits,
            (synthetic_domain_table["person_id"] == merge_patient_splits["person_id"])
            & (merge_patient_splits["is_real"] == 0),
        ).select(
            [synthetic_domain_table[f] for f in common_columns]
            + [
                merge_patient_splits["is_real"],
                merge_patient_splits["split"],
                merge_patient_splits["new_person_id"],
            ]
        )
        # Re-map visit_occurrence_id
        if "visit_occurrence_id" in [
            _.lower() for _ in synthetic_domain_table.schema.fieldNames()
        ]:
            synthetic_domain_table = (
                synthetic_domain_table.join(
                    synthetic_visit_occurrence_mapping, "visit_occurrence_id"
                )
                .drop("visit_occurrence_id")
                .withColumnRenamed("new_visit_occurrence_id", "visit_occurrence_id")
            )

        merge_domain_table = real_domain_table.unionByName(synthetic_domain_table)
        merge_domain_table = (
            merge_domain_table.withColumnRenamed("person_id", "original_person_id")
            .withColumnRenamed("new_person_id", "person_id")
            .drop("new_person_id")
        )
        merge_domain_table.write.mode("overwrite").parquet(
            os.path.join(output_folder, domain_table)
        )

    # Rename the columns for the patient splits dataframe
    merge_patient_splits.withColumnRenamed(
        "person_id", "original_person_id"
    ).withColumnRenamed("new_person_id", "person_id").write.mode("overwrite").parquet(
        os.path.join(output_folder, "patient_splits")
    )

    # Copy concept tables
    for concept_table in ["concept", "concept_relationship", "concept_ancestor"]:
        shutil.copytree(
            os.path.join(real_omop_folder, concept_table),
            os.path.join(output_folder, concept_table),
        )


def create_app_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Arguments for generate training data for Bert"
    )
    parser.add_argument(
        "--real_omop_folder",
        dest="real_omop_folder",
        action="store",
        help="The path for your input_folder where the Real OMOP folder is",
        required=True,
    )
    parser.add_argument(
        "--synthetic_omop_folder",
        dest="synthetic_omop_folder",
        action="store",
        help="The path for your input_folder where the Synthetic OMOP folder is",
        required=True,
    )
    parser.add_argument(
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to download",
        type=validate_table_names,
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
        "--merge_type",
        dest="merge_type",
        action="store",
        choices=[e.value for e in MergeType],
    )
    return parser


if __name__ == "__main__":
    ARGS = create_app_arg_parser().parse_args()
    main(
        ARGS.real_omop_folder,
        ARGS.synthetic_omop_folder,
        ARGS.domain_table_list,
        ARGS.output_folder,
        ARGS.merge_type,
    )
