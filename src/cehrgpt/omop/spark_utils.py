# Adapted from cehrbert_data.utils.spark_utils and vocab_utils (MIT License, same copyright holder as cehrgpt).
"""The Spark helpers of the scripts that build OMOP tables."""

import argparse
import os
from os import path

import pyspark.sql.functions as F

PERSON = "person"
VISIT_OCCURRENCE = "visit_occurrence"
CONDITION_OCCURRENCE = "condition_occurrence"
PROCEDURE_OCCURRENCE = "procedure_occurrence"
DRUG_EXPOSURE = "drug_exposure"
DEVICE_EXPOSURE = "device_exposure"
OBSERVATION = "observation"
MEASUREMENT = "measurement"
CATEGORICAL_MEASUREMENT = "categorical_measurement"
OBSERVATION_PERIOD = "observation_period"
DEATH = "death"
CDM_TABLES = [
    PERSON,
    VISIT_OCCURRENCE,
    CONDITION_OCCURRENCE,
    PROCEDURE_OCCURRENCE,
    DRUG_EXPOSURE,
    DEVICE_EXPOSURE,
    OBSERVATION,
    MEASUREMENT,
    CATEGORICAL_MEASUREMENT,
    OBSERVATION_PERIOD,
    DEATH,
]


def roll_up_to_drug_ingredients(drug_exposure, concept, concept_ancestor):
    # lowercase the schema fields
    drug_exposure = drug_exposure.select([F.col(f_n).alias(f_n.lower()) for f_n in drug_exposure.schema.fieldNames()])

    drug_ingredient = (
        drug_exposure.select("drug_concept_id")
        .distinct()
        .join(concept_ancestor, F.col("drug_concept_id") == F.col("descendant_concept_id"))
        .join(concept, F.col("ancestor_concept_id") == F.col("concept_id"))
        .where(concept["concept_class_id"] == "Ingredient")
        .select(F.col("drug_concept_id"), F.col("concept_id").alias("ingredient_concept_id"))
    )

    drug_ingredient_fields = [
        F.coalesce(F.col("ingredient_concept_id"), F.col("drug_concept_id")).alias("drug_concept_id")
    ]
    drug_ingredient_fields.extend(
        [F.col(field_name) for field_name in drug_exposure.schema.fieldNames() if field_name != "drug_concept_id"]
    )

    drug_exposure = drug_exposure.join(drug_ingredient, "drug_concept_id", "left_outer").select(drug_ingredient_fields)

    return drug_exposure


def preprocess_domain_table(
        spark,
        input_folder,
        domain_table_name,
        with_drug_rollup=True,
):
    domain_table = spark.read.parquet(os.path.join(input_folder, domain_table_name))
    if "concept" in domain_table_name.lower():
        return domain_table

    # lowercase the schema fields
    domain_table = domain_table.select([F.col(f_n).alias(f_n.lower()) for f_n in domain_table.schema.fieldNames()])

    for f_n in domain_table.schema.fieldNames():
        if "date" in f_n and "datetime" not in f_n:
            # convert date columns to the date type
            domain_table = domain_table.withColumn(f_n, F.to_date(f_n))
        elif "datetime" in f_n:
            # convert date columns to the datetime type
            domain_table = domain_table.withColumn(f_n, F.to_timestamp(f_n))

    if domain_table_name == "visit_occurrence":
        # This is CDM 5.2, we need to rename this column to be CDM 5.3 compatible
        if "discharge_to_concept_id" in domain_table.schema.fieldNames():
            domain_table = domain_table.withColumnRenamed("discharge_to_concept_id", "discharged_to_concept_id")

    if with_drug_rollup:
        if (
                domain_table_name == "drug_exposure"
                and path.exists(os.path.join(input_folder, "concept"))
                and path.exists(os.path.join(input_folder, "concept_ancestor"))
        ):
            concept = spark.read.parquet(os.path.join(input_folder, "concept"))
            concept_ancestor = spark.read.parquet(os.path.join(input_folder, "concept_ancestor"))
            domain_table = roll_up_to_drug_ingredients(domain_table, concept, concept_ancestor)

    return domain_table


def validate_table_names(domain_names):
    for domain_name in domain_names.split(" "):
        if domain_name not in CDM_TABLES:
            raise argparse.ArgumentTypeError(f"{domain_name} is an invalid CDM table name")
    return domain_names
