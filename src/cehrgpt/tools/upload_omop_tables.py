import argparse
import configparser
from pathlib import Path

import pyspark.sql.functions as f
from pyspark.sql import SparkSession

# Define timestamp column for filtering based on the folder name
omop_timestamp_dict = {
    "person": "birth_datetime",
    "condition_occurrence": "condition_start_date",
    "measurement": "measurement_date",
    "drug_exposure": "drug_exposure_start_date",
    "procedure_occurrence": "procedure_date",
    "observation": "observation_date",
    "visit_occurrence": "visit_start_date",
}


# Function to initialize and return the SparkSession
def get_spark_session():
    spark = (
        SparkSession.builder.appName("OMOP Upload")
        .config("spark.sql.legacy.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .getOrCreate()
    )
    return spark


# Function to upload OMOP tables to a database
def upload_omop_tables(spark, domain_table_folder, db_properties):
    # Load parquet file from the specified folder
    df = spark.read.format("parquet").load(str(domain_table_folder) + "/")

    # Filter dates outside of the acceptable range
    if domain_table_folder.name in omop_timestamp_dict:
        timestamp_column = omop_timestamp_dict[domain_table_folder.name]
        df = df.filter(f.col(timestamp_column) > f.lit("1900-01-01").cast("date"))
        df = df.filter(f.col(timestamp_column) < f.lit("9999-01-01").cast("date"))

    # Cast appropriate columns to integer and date types
    for column in df.columns:
        if "concept_id" in column:
            df = df.withColumn(column, f.col(column).cast("integer"))
        if "date" in column:
            df = df.withColumn(column, f.col(column).cast("date"))

    # Write to the database with specified options
    df.repartition(10).write.format("jdbc").options(
        url=db_properties["base_url"],
        dbtable=domain_table_folder.name,
        user=db_properties["user"],
        password=db_properties["password"],
        batchsize=200000,
        queryTimeout=500,
    ).mode("overwrite").save()


# Main function to process the folders and upload tables
def main(credential_path, input_folder):
    # Load database properties from the credentials file
    config = configparser.ConfigParser()
    config.read(credential_path)
    db_properties = config.defaults()

    # Initialize SparkSession
    spark = get_spark_session()

    # Process each folder within the input folder
    input_folder = Path(input_folder)
    uploaded_tables = []
    for folder in input_folder.glob("*"):
        try:
            if folder.is_dir() and folder.name in omop_timestamp_dict:
                upload_omop_tables(spark, folder, db_properties)
                uploaded_tables.append(folder.name)
                print(f"Table: {folder.name} uploaded successfully")
        except Exception as e:
            print(f"Error uploading table {folder.name}: {e}")

    print(f"Uploaded tables: {uploaded_tables}")


# Argument parsing moved under __name__ == "__main__"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for uploading OMOP tables")

    parser.add_argument(
        "-c",
        "--credential_path",
        required=True,
        help="The path for your database credentials",
    )

    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        help="Path to the input folder containing the OMOP tables",
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.credential_path, args.input_folder)
