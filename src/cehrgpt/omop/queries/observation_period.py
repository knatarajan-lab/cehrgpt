OBSERVATION_PERIOD_QUERY = """
select
    person_id,
    observation_period_start_date,
    case
        when observation_period_end_date >= add_months(current_date(), -12) then current_date()
        else observation_period_end_date
    end as observation_period_end_date,
    period_type_concept_id
from (
     SELECT person_id,
            MIN(observation_period_start_date) AS observation_period_start_date,
            MAX(observation_period_end_date)   AS observation_period_end_date,
            44814725                           as period_type_concept_id
     FROM (
              SELECT pt.person_id             AS person_id,
                     MIN(vt.visit_start_date) AS observation_period_start_date,
                     MAX(vt.visit_end_date)   AS observation_period_end_date
              FROM person as pt
                       JOIN visit_occurrence as vt ON pt.person_id = vt.person_id
              WHERE YEAR(vt.visit_start_date) >= 1985
                AND vt.visit_start_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT pt.person_id                 AS person_id,
                     MIN(co.condition_start_date) AS observation_period_start_date,
                     MAX(co.condition_start_date) AS observation_period_end_date
              FROM person as pt
                       JOIN condition_occurrence as co ON pt.person_id = co.person_id
              WHERE YEAR(co.condition_start_date) >= 1985
                AND co.condition_start_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT pt.person_id           AS person_id,
                     MIN(po.procedure_date) AS observation_period_start_date,
                     MAX(po.procedure_date) AS observation_period_end_date
              FROM person as pt
                       JOIN procedure_occurrence as po ON pt.person_id = po.person_id
              WHERE YEAR(po.procedure_date) >= 1985
                AND po.procedure_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT pt.person_id                     AS person_id,
                     MIN(de.drug_exposure_start_date) AS observation_period_start_date,
                     MAX(de.drug_exposure_start_date) AS observation_period_end_date
              FROM person as pt
                       JOIN drug_exposure as de ON pt.person_id = de.person_id
              WHERE YEAR(de.drug_exposure_start_date) >= 1985
                AND de.drug_exposure_start_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id
          ) as x
     WHERE x.observation_period_end_date IS NOT NULL
     GROUP BY person_id
 ) as z
"""

OBSERVATION_PERIOD_WITH_MEASUREMENT_QUERY = """
select
    person_id,
    observation_period_start_date,
    case
        when observation_period_end_date >= add_months(current_date(), -12) then current_date()
        else observation_period_end_date
    end as observation_period_end_date,
    period_type_concept_id
from (
     SELECT person_id,
            MIN(observation_period_start_date) AS observation_period_start_date,
            MAX(observation_period_end_date)   AS observation_period_end_date,
            44814725                           as period_type_concept_id
     FROM (
              SELECT pt.person_id             AS person_id,
                     MIN(vt.visit_start_date) AS observation_period_start_date,
                     MAX(vt.visit_end_date)   AS observation_period_end_date
              FROM person as pt
                       JOIN visit_occurrence as vt ON pt.person_id = vt.person_id
              WHERE YEAR(vt.visit_start_date) >= 1985
                AND vt.visit_start_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT pt.person_id                 AS person_id,
                     MIN(co.condition_start_date) AS observation_period_start_date,
                     MAX(co.condition_start_date) AS observation_period_end_date
              FROM person as pt
                       JOIN condition_occurrence as co ON pt.person_id = co.person_id
              WHERE YEAR(co.condition_start_date) >= 1985
                AND co.condition_start_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT pt.person_id           AS person_id,
                     MIN(po.procedure_date) AS observation_period_start_date,
                     MAX(po.procedure_date) AS observation_period_end_date
              FROM person as pt
                       JOIN procedure_occurrence as po ON pt.person_id = po.person_id
              WHERE YEAR(po.procedure_date) >= 1985
                AND po.procedure_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT pt.person_id                     AS person_id,
                     MIN(de.drug_exposure_start_date) AS observation_period_start_date,
                     MAX(de.drug_exposure_start_date) AS observation_period_end_date
              FROM person as pt
                       JOIN drug_exposure as de ON pt.person_id = de.person_id
              WHERE YEAR(de.drug_exposure_start_date) >= 1985
                AND de.drug_exposure_start_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id

              UNION

              SELECT
                pt.person_id AS person_id,
                MIN(m.measurement_date) AS observation_period_start_date,
                MAX(m.measurement_date) AS observation_period_end_date
              FROM person as pt
                JOIN measurement as m ON pt.person_id = m.person_id
              WHERE YEAR(m.measurement_date) >= 1985
                AND m.measurement_date <= current_date() --set lower and upper bound to ignore spurious dates
              GROUP BY pt.person_id
          ) as x
     WHERE x.observation_period_end_date IS NOT NULL
     GROUP BY person_id
 ) as z
"""
