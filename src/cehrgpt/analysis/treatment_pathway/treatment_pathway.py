#!/usr/bin/env python
"""
Treatment Pathways Study Script.

Analyzes treatment pathways using OMOP CDM data - generalized version of the original script
"""

import argparse
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze treatment pathways using OMOP CDM data"
    )

    parser.add_argument(
        "--omop-folder",
        required=True,
        help="Path to OMOP CDM data folder containing parquet files",
    )

    parser.add_argument(
        "--output-folder", required=True, help="Output folder for results"
    )

    parser.add_argument(
        "--drug-concepts",
        required=True,
        help="Comma-separated list of drug ancestor concept IDs (e.g., '21600381,21601461,21601560')",
    )

    parser.add_argument(
        "--target-conditions",
        required=True,
        help="Comma-separated list of target condition ancestor concept IDs (e.g., '316866')",
    )

    parser.add_argument(
        "--exclusion-conditions",
        help="Comma-separated list of exclusion condition ancestor concept IDs (e.g., '444094'). If not provided, no exclusion conditions will be applied.",
    )

    parser.add_argument(
        "--app-name",
        default="Treatment_Pathways",
        help="Spark application name (default: Treatment_Pathways)",
    )

    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )

    parser.add_argument(
        "--study-name",
        default="HTN",
        help="Study name prefix for cohort tables (default: HTN)",
    )

    parser.add_argument(
        "--save-cohort",
        action="store_true",
        default=False,
        help="Save cohort tables as parquet files",
    )

    return parser.parse_args()


def parse_concept_ids(concept_string):
    """Parse comma-separated concept IDs into a list of integers."""
    if not concept_string:
        return []
    return [int(x.strip()) for x in concept_string.split(",")]


def create_drug_concept_mapping(spark, drug_concepts):
    """Create drug concept mapping for medications - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    drug_concept = spark.sql(
        f"""
    SELECT DISTINCT
        ancestor_concept_id,
        descendant_concept_id
    FROM
    (
        SELECT
            ancestor_concept_id,
            descendant_concept_id
        FROM concept_ancestor AS ca
        WHERE ca.ancestor_concept_id IN ({drug_concept_ids})
    ) a
    """
    )
    drug_concept.cache()
    drug_concept.createOrReplaceTempView("drug_concept")


def create_htn_index_cohort(spark, drug_concepts, study_name):
    """Create HTN index cohort - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    htn_index_cohort = spark.sql(
        f"""
    SELECT person_id, INDEX_DATE, COHORT_END_DATE, observation_period_start_date, observation_period_end_date
    FROM (
      SELECT ot.PERSON_ID, ot.INDEX_DATE, MIN(e.END_DATE) as COHORT_END_DATE, ot.OBSERVATION_PERIOD_START_DATE, ot.OBSERVATION_PERIOD_END_DATE,
             ROW_NUMBER() OVER (PARTITION BY ot.PERSON_ID ORDER BY ot.INDEX_DATE) as RowNumber
      FROM (
        SELECT dt.PERSON_ID, dt.DRUG_EXPOSURE_START_DATE as index_date, op.OBSERVATION_PERIOD_START_DATE, op.OBSERVATION_PERIOD_END_DATE
        FROM (
          SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID, de.DRUG_EXPOSURE_START_DATE
          FROM (
            SELECT d.PERSON_ID, d.DRUG_CONCEPT_ID, d.DRUG_EXPOSURE_START_DATE,
                   COALESCE(d.DRUG_EXPOSURE_END_DATE, DATE_ADD(d.DRUG_EXPOSURE_START_DATE, d.DAYS_SUPPLY), DATE_ADD(d.DRUG_EXPOSURE_START_DATE, 1)) as DRUG_EXPOSURE_END_DATE,
                   ROW_NUMBER() OVER (PARTITION BY d.PERSON_ID ORDER BY d.DRUG_EXPOSURE_START_DATE) as RowNumber
            FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) d
            JOIN drug_concept ca
              ON d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
          ) de
          JOIN PERSON p ON p.PERSON_ID = de.PERSON_ID
          WHERE de.RowNumber = 1
        ) dt
        JOIN observation_period op
          ON op.PERSON_ID = dt.PERSON_ID AND (dt.DRUG_EXPOSURE_START_DATE BETWEEN op.OBSERVATION_PERIOD_START_DATE AND op.OBSERVATION_PERIOD_END_DATE)
        WHERE DATE_ADD(op.OBSERVATION_PERIOD_START_DATE, 365) <= dt.DRUG_EXPOSURE_START_DATE
          AND DATE_ADD(dt.DRUG_EXPOSURE_START_DATE, 1095) <= op.OBSERVATION_PERIOD_END_DATE
      ) ot
      JOIN (
        SELECT PERSON_ID, DATE_ADD(EVENT_DATE, -31) as END_DATE
        FROM (
          SELECT PERSON_ID, EVENT_DATE, EVENT_TYPE, START_ORDINAL,
                 ROW_NUMBER() OVER (PARTITION BY PERSON_ID ORDER BY EVENT_DATE, EVENT_TYPE) AS EVENT_ORDINAL,
                 MAX(START_ORDINAL) OVER (PARTITION BY PERSON_ID ORDER BY EVENT_DATE, EVENT_TYPE ROWS UNBOUNDED PRECEDING) as STARTS
          FROM (
            SELECT PERSON_ID, DRUG_EXPOSURE_START_DATE AS EVENT_DATE, 1 as EVENT_TYPE,
                   ROW_NUMBER() OVER (PARTITION BY PERSON_ID ORDER BY DRUG_EXPOSURE_START_DATE) as START_ORDINAL
            FROM (
              SELECT d.PERSON_ID, d.DRUG_CONCEPT_ID, d.DRUG_EXPOSURE_START_DATE,
                     COALESCE(d.DRUG_EXPOSURE_END_DATE, DATE_ADD(d.DRUG_EXPOSURE_START_DATE, d.DAYS_SUPPLY), DATE_ADD(d.DRUG_EXPOSURE_START_DATE, 1)) as DRUG_EXPOSURE_END_DATE,
                     ROW_NUMBER() OVER (PARTITION BY d.PERSON_ID ORDER BY d.DRUG_EXPOSURE_START_DATE) as RowNumber
              FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) d
              JOIN drug_concept ca
                ON d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
            ) cteExposureData
            UNION ALL
            SELECT PERSON_ID, DATE_ADD(DRUG_EXPOSURE_END_DATE, 31), 0 as EVENT_TYPE, NULL
            FROM (
              SELECT d.PERSON_ID, d.DRUG_CONCEPT_ID, d.DRUG_EXPOSURE_START_DATE,
                     COALESCE(d.DRUG_EXPOSURE_END_DATE, DATE_ADD(d.DRUG_EXPOSURE_START_DATE, d.DAYS_SUPPLY), DATE_ADD(d.DRUG_EXPOSURE_START_DATE, 1)) as DRUG_EXPOSURE_END_DATE,
                     ROW_NUMBER() OVER (PARTITION BY d.PERSON_ID ORDER BY d.DRUG_EXPOSURE_START_DATE) as RowNumber
              FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) d
              JOIN drug_concept ca
                ON d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
            ) cteExposureData
          ) RAWDATA
        ) E
        WHERE 2 * E.STARTS - E.EVENT_ORDINAL = 0
      ) e ON e.PERSON_ID = ot.PERSON_ID AND e.END_DATE >= ot.INDEX_DATE
      GROUP BY ot.PERSON_ID, ot.INDEX_DATE, ot.OBSERVATION_PERIOD_START_DATE, ot.OBSERVATION_PERIOD_END_DATE
    ) r
    WHERE r.RowNumber = 1
    """
    )

    htn_index_cohort.cache()
    htn_index_cohort.createOrReplaceTempView(f"{study_name}_index_cohort")


def create_htn_e0(spark, exclusion_conditions, study_name):
    """Create HTN_E0 - exact query from original."""
    if not exclusion_conditions:
        # If no exclusion conditions, return all patients from index cohort
        HTN_E0 = spark.sql(
            f"""
        SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
        FROM {study_name}_index_cohort ip
        """
        )
    else:
        exclusion_concept_ids = ",".join(map(str, exclusion_conditions))
        HTN_E0 = spark.sql(
            f"""
        SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
        FROM {study_name}_index_cohort ip
        LEFT JOIN (
          SELECT co.PERSON_ID, co.CONDITION_CONCEPT_ID
          FROM condition_occurrence co
          JOIN {study_name}_index_cohort ip ON co.PERSON_ID = ip.PERSON_ID
          JOIN drug_concept ca ON co.CONDITION_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({exclusion_concept_ids})
          WHERE co.CONDITION_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        ) dt ON dt.PERSON_ID = ip.PERSON_ID
        GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
        HAVING COUNT(dt.CONDITION_CONCEPT_ID) <= 0
        """
        )

    HTN_E0.cache()
    HTN_E0.createOrReplaceTempView(f"{study_name}_E0")


def create_htn_t0(spark, drug_concepts, study_name):
    """Create HTN_T0 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T0 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND DATE_ADD(ip.INDEX_DATE, -1)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) <= 0
    """
    )

    HTN_T0.createOrReplaceTempView(f"{study_name}_T0")


def create_htn_t1(spark, target_conditions, study_name):
    """Create HTN_T1 - exact query from original."""
    target_concept_ids = ",".join(map(str, target_conditions))

    HTN_T1 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT ce.PERSON_ID, ce.CONDITION_CONCEPT_ID
      FROM CONDITION_ERA ce
      JOIN {study_name}_index_cohort ip ON ce.PERSON_ID = ip.PERSON_ID
      JOIN concept_ancestor ca ON ce.CONDITION_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({target_concept_ids})
      WHERE ce.CONDITION_ERA_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
    ) ct ON ct.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(ct.CONDITION_CONCEPT_ID) >= 1
    """
    )

    HTN_T1.createOrReplaceTempView(f"{study_name}_T1")


def create_htn_t2(spark, drug_concepts, study_name):
    """Create HTN_T2 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T2 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 121) AND DATE_ADD(ip.INDEX_DATE, 240)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T2.createOrReplaceTempView(f"{study_name}_T2")


def create_htn_t3(spark, drug_concepts, study_name):
    """Create HTN_T3 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T3 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 241) AND DATE_ADD(ip.INDEX_DATE, 360)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T3.createOrReplaceTempView(f"{study_name}_T3")


def create_htn_t4(spark, drug_concepts, study_name):
    """Create HTN_T4 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T4 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 361) AND DATE_ADD(ip.INDEX_DATE, 480)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T4.createOrReplaceTempView(f"{study_name}_T4")


def create_htn_t5(spark, drug_concepts, study_name):
    """Create HTN_T5 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T5 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 481) AND DATE_ADD(ip.INDEX_DATE, 600)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T5.createOrReplaceTempView(f"{study_name}_T5")


def create_htn_t6(spark, drug_concepts, study_name):
    """Create HTN_T6 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T6 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 601) AND DATE_ADD(ip.INDEX_DATE, 720)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T6.createOrReplaceTempView(f"{study_name}_T6")


def create_htn_t7(spark, drug_concepts, study_name):
    """Create HTN_T7 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T7 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 721) AND DATE_ADD(ip.INDEX_DATE, 840)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T7.createOrReplaceTempView(f"{study_name}_T7")


def create_htn_t8(spark, drug_concepts, study_name):
    """Create HTN_T8 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T8 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 841) AND DATE_ADD(ip.INDEX_DATE, 960)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T8.createOrReplaceTempView(f"{study_name}_T8")


def create_htn_t9(spark, drug_concepts, study_name):
    """Create HTN_T9 - exact query from original."""
    drug_concept_ids = ",".join(map(str, drug_concepts))

    HTN_T9 = spark.sql(
        f"""
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM {study_name}_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN {study_name}_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN ({drug_concept_ids})
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 961) AND DATE_ADD(ip.INDEX_DATE, 1080)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T9.createOrReplaceTempView(f"{study_name}_T9")


def create_htn_match_cohort(spark, study_name):
    """Create HTN_MatchCohort - exact query from original."""
    HTN_MatchCohort = spark.sql(
        f"""
    SELECT c.person_id, c.index_date, c.cohort_end_date, c.observation_period_start_date, c.observation_period_end_date
    FROM {study_name}_index_cohort C
    INNER JOIN (
      SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID
      FROM (
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_E0
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T0
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T1
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T2
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T3
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T4
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T5
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T6
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T7
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T8
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM {study_name}_T9
      ) TopGroup
    ) I
    ON C.PERSON_ID = I.PERSON_ID
    AND c.index_date = i.index_date
    """
    )

    return HTN_MatchCohort


def main():
    args = parse_arguments()
    # Parse concept IDs
    drug_concepts = parse_concept_ids(args.drug_concepts)
    target_conditions = parse_concept_ids(args.target_conditions)
    exclusion_conditions = (
        parse_concept_ids(args.exclusion_conditions)
        if args.exclusion_conditions
        else []
    )

    print(f"Study name: {args.study_name}")
    print(f"Drug concepts: {drug_concepts}")
    print(f"Target conditions: {target_conditions}")
    print(f"Exclusion conditions: {exclusion_conditions}")

    # Initialize Spark
    spark = SparkSession.builder.appName(args.app_name).getOrCreate()

    try:
        # Load the source OMOP tables
        person = spark.read.parquet(os.path.join(args.omop_folder, "person"))
        visit_occurrence = spark.read.parquet(
            os.path.join(args.omop_folder, "visit_occurrence")
        )
        condition_occurrence = spark.read.parquet(
            os.path.join(args.omop_folder, "condition_occurrence")
        )
        procedure_occurrence = spark.read.parquet(
            os.path.join(args.omop_folder, "procedure_occurrence")
        )
        drug_exposure = spark.read.parquet(
            os.path.join(args.omop_folder, "drug_exposure")
        )
        observation_period = spark.read.parquet(
            os.path.join(args.omop_folder, "observation_period")
        )
        condition_era = spark.read.parquet(
            os.path.join(args.omop_folder, "condition_era")
        )

        print(f"person: {person.select('person_id').distinct().count()}")
        print(f"visit_occurrence: {visit_occurrence.count()}")
        print(f"condition_occurrence: {condition_occurrence.count()}")
        print(f"procedure_occurrence: {procedure_occurrence.count()}")
        print(f"drug_exposure: {drug_exposure.count()}")
        print(f"observation_period: {observation_period.count()}")

        concept = spark.read.parquet(os.path.join(args.omop_folder, "concept"))
        concept_ancestor = spark.read.parquet(
            os.path.join(args.omop_folder, "concept_ancestor")
        )

        # Create temporary views
        person.createOrReplaceTempView("person")
        visit_occurrence.createOrReplaceTempView("visit_occurrence")
        condition_occurrence.createOrReplaceTempView("condition_occurrence")
        procedure_occurrence.createOrReplaceTempView("procedure_occurrence")
        drug_exposure.createOrReplaceTempView("drug_exposure")
        observation_period.createOrReplaceTempView("observation_period")
        condition_era.createOrReplaceTempView("condition_era")

        concept_ancestor.createOrReplaceTempView("concept_ancestor")
        concept.createOrReplaceTempView("concept")

        # Create drug concept mapping
        create_drug_concept_mapping(spark, drug_concepts)

        # Create HTN index cohort
        create_htn_index_cohort(spark, drug_concepts, args.study_name)

        # Create all cohorts in sequence - keeping exact original function calls
        create_htn_e0(spark, exclusion_conditions, args.study_name)
        create_htn_t0(spark, drug_concepts, args.study_name)
        create_htn_t1(spark, target_conditions, args.study_name)
        create_htn_t2(spark, drug_concepts, args.study_name)
        create_htn_t3(spark, drug_concepts, args.study_name)
        create_htn_t4(spark, drug_concepts, args.study_name)
        create_htn_t5(spark, drug_concepts, args.study_name)
        create_htn_t6(spark, drug_concepts, args.study_name)
        create_htn_t7(spark, drug_concepts, args.study_name)
        create_htn_t8(spark, drug_concepts, args.study_name)
        create_htn_t9(spark, drug_concepts, args.study_name)

        # Create final cohort
        htn_match_cohort = create_htn_match_cohort(spark, args.study_name)

        if args.save_cohort:
            # Save results
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)

            output_path = os.path.join(args.output_folder, "htn_match_cohort")
            htn_match_cohort.write.mode("overwrite").parquet(output_path)

            # Read back and count
            htn_match_cohort = spark.read.parquet(output_path)
        final_count = htn_match_cohort.count()
        print(f"Final cohort count: {final_count}")

        print("Analysis completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
