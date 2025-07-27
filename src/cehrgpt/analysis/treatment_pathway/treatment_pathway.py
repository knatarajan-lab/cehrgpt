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
from pyspark.sql import types as t
from pyspark.sql.window import Window


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

    parser.add_argument(
        "--small-cell-suppression-threshold",
        type=int,
        default=0,
        help="Small cell suppression threshold (default: 0)",
    )

    parser.add_argument(
        "--generate-drug-sequence",
        action="store_true",
        required=True,
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


def create_qualifying_cohort(
    study_name, drug_concepts, target_conditions, exclusion_conditions, spark
):
    # Create drug concept mapping
    create_drug_concept_mapping(spark, drug_concepts)
    # Create HTN index cohort
    create_htn_index_cohort(spark, drug_concepts, study_name)
    # Create all cohorts in sequence - keeping exact original function calls
    create_htn_e0(spark, exclusion_conditions, study_name)
    create_htn_t0(spark, drug_concepts, study_name)
    create_htn_t1(spark, target_conditions, study_name)
    create_htn_t2(spark, drug_concepts, study_name)
    create_htn_t3(spark, drug_concepts, study_name)
    create_htn_t4(spark, drug_concepts, study_name)
    create_htn_t5(spark, drug_concepts, study_name)
    create_htn_t6(spark, drug_concepts, study_name)
    create_htn_t7(spark, drug_concepts, study_name)
    create_htn_t8(spark, drug_concepts, study_name)
    create_htn_t9(spark, drug_concepts, study_name)
    # Create final cohort
    htn_match_cohort = create_htn_match_cohort(spark, study_name)
    return htn_match_cohort


def create_drug_sequences(
    qualifying_cohort, drug_era, concept, concept_ancestor, drug_concept_ids
):
    """Create drug sequences for the matching cohort."""

    # Get all drugs that the matching cohort had taken
    drug_sequences = (
        drug_era.join(
            qualifying_cohort, drug_era.person_id == qualifying_cohort.person_id
        )
        .join(
            concept_ancestor,
            (drug_era.drug_concept_id == concept_ancestor.descendant_concept_id)
            & (concept_ancestor.ancestor_concept_id.isin(drug_concept_ids)),
        )
        .join(concept, drug_era.drug_concept_id == concept.concept_id)
        .select(
            "person_id",
            "drug_concept_id",
            f.year(qualifying_cohort.index_date).alias("index_year"),
            "drug_era_start_date",
            concept.concept_name,
        )
        .groupBy("person_id", "drug_concept_id", "index_year", "concept_name")
        .agg(f.min("drug_era_start_date").alias("drug_start_date"))
        .withColumn(
            "drug_seq",
            f.row_number().over(
                Window.partitionBy("person_id").orderBy(
                    "drug_start_date", "drug_concept_id"
                )
            ),
        )
    )
    return drug_sequences


def summarize_treatment_sequences(drug_sequences):
    """Summarize the unique treatment sequences observed."""
    print("Summarizing treatment sequences...")

    # Create pivot data for each sequence position (d1, d2, ..., d20)
    sequence_data = {}

    for i in range(1, 21):  # d1 through d20
        seq_data = drug_sequences.filter(f.col("drug_seq") == i).select(
            "person_id",
            f.col("drug_concept_id").alias(f"d{i}_concept_id"),
            f.col("concept_name").alias(f"d{i}_concept_name"),
            "index_year",
        )
        sequence_data[f"d{i}"] = seq_data

    # Start with persons and their first drug (d1)
    summary_base = (
        drug_sequences.select("person_id", "index_year")
        .distinct()
        .join(sequence_data["d1"], ["person_id", "index_year"], "left")
    )

    # Left join all subsequent drugs
    for i in range(2, 21):
        summary_base = summary_base.join(
            sequence_data[f"d{i}"], ["person_id", "index_year"], "left"
        )

    # Group and count sequences
    group_cols = ["index_year"]
    for i in range(1, 21):
        group_cols.extend([f"d{i}_concept_id", f"d{i}_concept_name"])

    drug_seq_summary = (
        summary_base.groupBy(*group_cols)
        .agg(f.countDistinct("person_id").alias("num_persons"))
        .filter(f.col("num_persons") > 0)  # Remove empty sequences
    )
    return drug_seq_summary


def apply_small_cell_suppression(drug_seq_summary, threshold=0):
    """Apply small cell count suppression logic."""
    print(f"Applying small cell suppression (threshold: {threshold})")

    # Apply basic suppression - replace small counts with "Other"
    for i in range(1, 21):
        drug_seq_summary = drug_seq_summary.withColumn(
            f"d{i}_concept_id",
            f.when(
                (f.col(f"d{i}_concept_id").isNotNull())
                & (f.col("num_persons") < threshold),
                -1,
            ).otherwise(f.col(f"d{i}_concept_id")),
        ).withColumn(
            f"d{i}_concept_name",
            f.when(f.col(f"d{i}_concept_id") == -1, "Other").otherwise(
                f.col(f"d{i}_concept_name")
            ),
        )

    # Re-aggregate after suppression
    group_cols = ["index_year"]
    for i in range(1, 21):
        group_cols.extend([f"d{i}_concept_id", f"d{i}_concept_name"])

    final_summary = drug_seq_summary.groupBy(*group_cols).agg(
        f.sum("num_persons").alias("num_persons")
    )
    return final_summary


def create_summary_tables(
    spark,
    person,
    drug_exposure,
    observation_period,
    concept_ancestor,
    match_cohort,
    final_drug_seq_summary,
    tx_list,
):
    """Create final summary tables for export."""
    print("Creating summary tables...")
    # Summary counts
    summary_data = []

    # Total persons
    total_persons = person.count()
    summary_data.append(("Number of persons", total_persons))

    # Persons with at least one drug exposure
    persons_with_drugs = (
        drug_exposure.join(
            concept_ancestor,
            (drug_exposure.drug_concept_id == concept_ancestor.descendant_concept_id)
            & (concept_ancestor.ancestor_concept_id.isin(tx_list)),
        )
        .select("person_id")
        .distinct()
        .count()
    )
    summary_data.append(
        ("Number of persons with at least one drug exposure", persons_with_drugs)
    )

    # Persons with sufficient observation time
    sufficient_observation = (
        drug_exposure.join(
            concept_ancestor,
            (drug_exposure.drug_concept_id == concept_ancestor.descendant_concept_id)
            & (concept_ancestor.ancestor_concept_id.isin(tx_list)),
        )
        .groupBy("person_id")
        .agg(f.min("drug_exposure_start_date").alias("first_drug"))
        .join(observation_period, "person_id")
        .filter(
            (
                f.date_add(observation_period.observation_period_start_date, 365)
                <= f.col("first_drug")
            )
            & (
                f.date_add(f.col("first_drug"), 1095)
                <= observation_period.observation_period_end_date
            )
        )
        .select("person_id")
        .distinct()
        .count()
    )
    summary_data.append(
        ("Number of persons with sufficient observation time", sufficient_observation)
    )

    # Final qualifying cohort
    final_cohort_count = match_cohort.count()
    summary_data.append(
        ("Number of persons in final qualifying cohort", final_cohort_count)
    )

    # Create summary DataFrame
    summary_schema = t.StructType(
        [
            t.StructField("count_type", t.StringType(), True),
            t.StructField("num_persons", t.IntegerType(), True),
        ]
    )

    summary_df = spark.createDataFrame(summary_data, summary_schema)

    # Person counts by year
    person_counts = final_drug_seq_summary.groupBy("index_year").agg(
        f.sum("num_persons").alias("num_persons")
    )

    # Add overall count (year 9999)
    overall_count = (
        final_drug_seq_summary.agg(f.sum("num_persons").alias("num_persons"))
        .withColumn("index_year", f.lit(9999))
        .select("index_year", "num_persons")
    )

    person_counts_final = person_counts.union(overall_count)

    # Add overall sequences (year 9999)
    overall_sequences = (
        final_drug_seq_summary.drop("index_year")
        .groupBy(
            *[
                col
                for col in final_drug_seq_summary.columns
                if col != "index_year" and col != "num_persons"
            ]
        )
        .agg(f.sum("num_persons").alias("num_persons"))
        .withColumn("index_year", f.lit(9999))
    )

    seq_counts_final = final_drug_seq_summary.union(overall_sequences)
    # Store results
    return summary_df, person_counts_final, seq_counts_final


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
        drug_era = spark.read.parquet(os.path.join(args.omop_folder, "drug_era"))

        print(f"person: {person.select('person_id').distinct().count()}")
        print(f"visit_occurrence: {visit_occurrence.count()}")
        print(f"condition_occurrence: {condition_occurrence.count()}")
        print(f"procedure_occurrence: {procedure_occurrence.count()}")
        print(f"drug_exposure: {drug_exposure.count()}")
        print(f"observation_period: {observation_period.count()}")
        print(f"condition_era: {condition_era.count()}")
        print(f"drug_era: {drug_era.count()}")

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
        drug_era.createOrReplaceTempView("drug_era")

        concept_ancestor.createOrReplaceTempView("concept_ancestor")
        concept.createOrReplaceTempView("concept")

        qualifying_cohort_cohort = create_qualifying_cohort(
            args.study_name,
            drug_concepts,
            target_conditions,
            exclusion_conditions,
            spark,
        )

        # Save results
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        output_path = os.path.join(
            args.output_folder, f"{args.study_name}_match_cohort"
        )
        qualifying_cohort_cohort.write.mode("overwrite").parquet(output_path)
        # Read back and count
        qualifying_cohort_cohort = spark.read.parquet(output_path)
        final_count = qualifying_cohort_cohort.count()
        print(f"Final cohort count: {final_count}")

        if args.generate_drug_sequence:
            drug_sequences = create_drug_sequences(
                qualifying_cohort_cohort,
                drug_era,
                concept,
                concept_ancestor,
                drug_concepts,
            )
            drug_sequences.write.mode("overwrite").parquet(
                os.path.join(output_path, f"{args.study_name}_drug_sequences")
            )
            drug_sequences = spark.read.parquet(
                os.path.join(output_path, f"{args.study_name}_drug_sequences")
            )
            print(
                f"Created drug sequences for {drug_sequences.select('person_id').distinct().count()} patients"
            )
            drug_seq_summary = summarize_treatment_sequences(drug_sequences)
            drug_seq_summary.write.mode("overwrite").parquet(
                os.path.join(output_path, f"{args.study_name}_drug_sequence_summary")
            )
            print(
                f"After suppression: {drug_seq_summary.count()} unique treatment sequences"
            )

            if args.small_cell_suppression_threshold > 0:
                drug_seq_summary = apply_small_cell_suppression(
                    drug_seq_summary, threshold=args.small_cell_suppression_threshold
                )
                drug_seq_summary.write.mode("overwrite").parquet(
                    os.path.join(
                        output_path,
                        f"{args.study_name}_drug_sequence_summary_small_cell_suppression",
                    )
                )
                drug_seq_summary = spark.read.parquet(
                    os.path.join(
                        output_path,
                        f"{args.study_name}_drug_sequence_summary_small_cell_suppression",
                    )
                )
                print(
                    f"After suppression: {drug_seq_summary.count()} unique treatment sequences"
                )

            summary_df, person_counts_final, seq_counts_final = create_summary_tables(
                spark,
                person,
                drug_exposure,
                observation_period,
                concept_ancestor,
                qualifying_cohort_cohort,
                drug_seq_summary,
                drug_concepts,
            )
            summary_df.write.mode("overwrite").parquet(
                os.path.join(output_path, f"{args.study_name}_summary")
            )
            person_counts_final.write.mode("overwrite").parquet(
                os.path.join(output_path, f"{args.study_name}_person_count_final")
            )
            seq_counts_final.write.mode("overwrite").parquet(
                os.path.join(output_path, f"{args.study_name}_sequence_count_final")
            )

        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
