#!/usr/bin/env python
"""
Treatment Pathways Study Script.

Analyzes hypertension treatment pathways using OMOP CDM data
"""

import argparse
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as f


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze hypertension treatment pathways using OMOP CDM data"
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
        "--app-name",
        default="HTN_Treatment_Pathways",
        help="Spark application name (default: HTN_Treatment_Pathways)",
    )

    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )

    return parser.parse_args()


def create_drug_concept_mapping(spark):
    """Create drug concept mapping for HTN medications - exact query from original."""
    drug_concept = spark.sql(
        """
    SELECT DISTINCT
        ancestor_concept_id,
        descendant_concept_id
    FROM
    (
        SELECT
            ancestor_concept_id,
            descendant_concept_id
        FROM concept_ancestor AS ca
        WHERE ca.ancestor_concept_id IN (21600381,21601461,21601560,21601664,21601744,21601782)
    ) a
    """
    )
    drug_concept.cache()
    drug_concept.createOrReplaceTempView("drug_concept")


def create_htn_index_cohort(spark):
    """Create HTN index cohort - exact query from original."""
    htn_index_cohort = spark.sql(
        """
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
              ON d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
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
                     ROW_NUMBER() OVER (PARTITION BY d.PERSON_ID ORDER BY DRUG_EXPOSURE_START_DATE) as RowNumber
              FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) d
              JOIN drug_concept ca
                ON d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
            ) cteExposureData
            UNION ALL
            SELECT PERSON_ID, DATE_ADD(DRUG_EXPOSURE_END_DATE, 31), 0 as EVENT_TYPE, NULL
            FROM (
              SELECT d.PERSON_ID, d.DRUG_CONCEPT_ID, d.DRUG_EXPOSURE_START_DATE,
                     COALESCE(d.DRUG_EXPOSURE_END_DATE, DATE_ADD(d.DRUG_EXPOSURE_START_DATE, d.DAYS_SUPPLY), DATE_ADD(d.DRUG_EXPOSURE_START_DATE, 1)) as DRUG_EXPOSURE_END_DATE,
                     ROW_NUMBER() OVER (PARTITION BY d.PERSON_ID ORDER BY DRUG_EXPOSURE_START_DATE) as RowNumber
              FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) d
              JOIN drug_concept ca
                ON d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
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
    htn_index_cohort.createOrReplaceTempView("htn_index_cohort")


def create_htn_e0(spark):
    """Create HTN_E0 - exact query from original."""
    HTN_E0 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT co.PERSON_ID, co.CONDITION_CONCEPT_ID
      FROM condition_occurrence co
      JOIN htn_index_cohort ip ON co.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON co.CONDITION_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (444094)
      WHERE co.CONDITION_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.CONDITION_CONCEPT_ID) <= 0
    """
    )

    HTN_E0.cache()
    HTN_E0.createOrReplaceTempView("HTN_E0")


def create_htn_t0(spark):
    """Create HTN_T0 - exact query from original."""
    HTN_T0 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (SELECT * FROM DRUG_EXPOSURE WHERE visit_occurrence_id IS NOT NULL) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND DATE_ADD(ip.INDEX_DATE, -1)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) <= 0
    """
    )

    HTN_T0.createOrReplaceTempView("HTN_T0")


def create_htn_t1(spark):
    """Create HTN_T1 - exact query from original."""
    HTN_T1 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT ce.PERSON_ID, ce.CONDITION_CONCEPT_ID
      FROM CONDITION_ERA ce
      JOIN htn_index_cohort ip ON ce.PERSON_ID = ip.PERSON_ID
      JOIN concept_ancestor ca ON ce.CONDITION_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (316866)
      WHERE ce.CONDITION_ERA_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
    ) ct ON ct.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(ct.CONDITION_CONCEPT_ID) >= 1
    """
    )

    HTN_T1.createOrReplaceTempView("HTN_T1")


def create_htn_t2(spark):
    """Create HTN_T2 - exact query from original."""
    HTN_T2 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 121) AND DATE_ADD(ip.INDEX_DATE, 240)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T2.createOrReplaceTempView("HTN_T2")


def create_htn_t3(spark):
    """Create HTN_T3 - exact query from original."""
    HTN_T3 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 241) AND DATE_ADD(ip.INDEX_DATE, 360)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T3.createOrReplaceTempView("HTN_T3")


def create_htn_t4(spark):
    """Create HTN_T4 - exact query from original."""
    HTN_T4 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 361) AND DATE_ADD(ip.INDEX_DATE, 480)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T4.createOrReplaceTempView("HTN_T4")


def create_htn_t5(spark):
    """Create HTN_T5 - exact query from original."""
    HTN_T5 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 481) AND DATE_ADD(ip.INDEX_DATE, 600)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T5.createOrReplaceTempView("HTN_T5")


def create_htn_t6(spark):
    """Create HTN_T6 - exact query from original."""
    HTN_T6 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 601) AND DATE_ADD(ip.INDEX_DATE, 720)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T6.createOrReplaceTempView("HTN_T6")


def create_htn_t7(spark):
    """Create HTN_T7 - exact query from original."""
    HTN_T7 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 721) AND DATE_ADD(ip.INDEX_DATE, 840)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T7.createOrReplaceTempView("HTN_T7")


def create_htn_t8(spark):
    """Create HTN_T8 - exact query from original."""
    HTN_T8 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 841) AND DATE_ADD(ip.INDEX_DATE, 960)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T8.createOrReplaceTempView("HTN_T8")


def create_htn_t9(spark):
    """Create HTN_T9 - exact query from original."""
    HTN_T9 = spark.sql(
        """
    SELECT ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    FROM htn_index_cohort ip
    LEFT JOIN (
      SELECT de.PERSON_ID, de.DRUG_CONCEPT_ID
      FROM (
        SELECT *
        FROM DRUG_EXPOSURE
        WHERE visit_occurrence_id IS NOT NULL
      ) de
      JOIN htn_index_cohort ip ON de.PERSON_ID = ip.PERSON_ID
      JOIN drug_concept ca ON de.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID AND ca.ANCESTOR_CONCEPT_ID IN (21600381, 21601461, 21601560, 21601664, 21601744, 21601782)
      WHERE de.DRUG_EXPOSURE_START_DATE BETWEEN ip.OBSERVATION_PERIOD_START_DATE AND ip.OBSERVATION_PERIOD_END_DATE
        AND de.DRUG_EXPOSURE_START_DATE BETWEEN DATE_ADD(ip.INDEX_DATE, 961) AND DATE_ADD(ip.INDEX_DATE, 1080)
    ) dt ON dt.PERSON_ID = ip.PERSON_ID
    GROUP BY ip.PERSON_ID, ip.INDEX_DATE, ip.COHORT_END_DATE
    HAVING COUNT(dt.DRUG_CONCEPT_ID) >= 1
    """
    )

    HTN_T9.createOrReplaceTempView("HTN_T9")


def create_htn_match_cohort(spark):
    """Create HTN_MatchCohort - exact query from original."""
    HTN_MatchCohort = spark.sql(
        """
    SELECT c.person_id, c.index_date, c.cohort_end_date, c.observation_period_start_date, c.observation_period_end_date
    FROM HTN_Index_Cohort C
    INNER JOIN (
      SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID
      FROM (
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_E0
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T0
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T1
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T2
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T3
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T4
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T5
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T6
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T7
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T8
        INTERSECT
        SELECT INDEX_DATE, COHORT_END_DATE, PERSON_ID FROM HTN_T9
      ) TopGroup
    ) I
    ON C.PERSON_ID = I.PERSON_ID
    AND c.index_date = i.index_date
    """
    )

    return HTN_MatchCohort


def main():
    args = parse_arguments()

    # Initialize Spark
    spark = SparkSession.builder.appName(f"HTN treatment pathway").getOrCreate()

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

        print(f"person: {visit_occurrence.select('person_id').distinct().count()}")
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
        create_drug_concept_mapping(spark)

        # Create HTN index cohort
        create_htn_index_cohort(spark)

        # Create all cohorts in sequence
        create_htn_e0(spark)
        create_htn_t0(spark)
        create_htn_t1(spark)
        create_htn_t2(spark)
        create_htn_t3(spark)
        create_htn_t4(spark)
        create_htn_t5(spark)
        create_htn_t6(spark)
        create_htn_t7(spark)
        create_htn_t8(spark)
        create_htn_t9(spark)

        # Create final cohort
        htn_match_cohort = create_htn_match_cohort(spark)

        # # Save results
        # if not os.path.exists(args.output_folder):
        #     os.makedirs(args.output_folder)
        #
        # output_path = os.path.join(args.output_folder, "htn_match_cohort")
        # htn_match_cohort.write.mode("overwrite").parquet(output_path)
        #
        # # Read back and count
        # saved_cohort = spark.read.parquet(output_path)
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
