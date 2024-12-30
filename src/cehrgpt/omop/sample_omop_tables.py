import argparse
import os

from pyspark.sql import SparkSession

# Define timestamp column for filtering based on the folder name
omop_tables = [
    "person",
    "death",
    "visit_occurrence",
    "condition_occurrence",
    "procedure_occurrence",
    "drug_exposure",
    "measurement",
    "observation",
    "observation_period",
    "condition_era",
    "drug_era",
]


# Main function to process the folders and upload tables
def main(args):
    spark = (
        SparkSession.builder.appName("Sample OMOP Tables")
        .config("spark.sql.legacy.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .getOrCreate()
    )
    patient_sample = spark.read.parquet(args.person_sample)
    for omop_table in omop_tables:
        omop_table_input = os.path.join(args.omop_folder, omop_table)
        omop_table_output = os.path.join(args.output_folder, omop_table)
        # If the input does not exist, let's skip
        if not os.path.exists(omop_table_input):
            continue
        # If the output already exists, and overwrite is not set to True, let's skip it
        if os.path.exists(omop_table_output) and not args.overwrite:
            continue
        omop_dataframe = spark.read.parquet(omop_table_input)
        sub_omop_dataframe = omop_dataframe.join(
            patient_sample.select("person_id"), "person_id"
        )
        sub_omop_dataframe.write.mode("overwrite" if args.overwrite else None).parquet(
            omop_table_output
        )


# Argument parsing moved under __name__ == "__main__"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for uploading OMOP tables")
    parser.add_argument(
        "--person_sample",
        required=True,
    )
    parser.add_argument(
        "--omop_folder",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    # Call the main function with parsed arguments
    main(parser.parse_args())
