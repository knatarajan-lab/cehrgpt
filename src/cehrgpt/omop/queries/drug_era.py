DRUG_ERA_QUERY = """
WITH temp_drug_ingredient AS
(
    SELECT
        d.*,
        lag(drug_exposure_end_date, 1) OVER(
            PARTITION BY person_id, ingredient_concept_id
            ORDER BY drug_exposure_start_date, drug_exposure_end_date DESC
        ) AS prev_end_date
    FROM
    (
        SELECT
            d.drug_exposure_id,
            d.person_id,
            c.concept_id as ingredient_concept_id,
            d.drug_type_concept_id,
            d.drug_exposure_start_date,
            CASE
                WHEN d.drug_exposure_end_date > d.drug_exposure_start_date THEN d.drug_exposure_end_date
                ELSE COALESCE(date_add(d.drug_exposure_start_date, d.days_supply), date_add(d.drug_exposure_start_date, 1))
            END AS drug_exposure_end_date
        FROM drug_exposure AS d
        JOIN concept_ancestor AS ca on ca.descendant_concept_id = d.drug_concept_id
        JOIN concept AS c on ca.ancestor_concept_id = c.concept_id
        WHERE d.drug_concept_id != 0
            AND c.vocabulary_id = 'RxNorm'
            AND c.concept_class_id = 'Ingredient'
            AND YEAR(d.drug_exposure_start_date) >= 1985 AND d.drug_exposure_start_date <= current_date()
    ) d
),
cte_drug_period AS
(
    SELECT
        *,
        SUM(start_new_era) OVER(PARTITION BY person_id, ingredient_concept_id
            ORDER BY drug_exposure_start_date, start_new_era DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS partition_num
    FROM
    (
        SELECT
            *,
            CASE
                WHEN prev_end_date IS NOT NULL THEN DATEDIFF(prev_end_date, drug_exposure_start_date)
                ELSE NULL
            END AS days_gap,
            CASE
                WHEN prev_end_date IS NOT NULL THEN
                    CASE
                        WHEN DATEDIFF(prev_end_date, drug_exposure_start_date) >= -30 THEN 0
                        ELSE 1
                    END
                ELSE 1
            END AS start_new_era
        FROM temp_drug_ingredient
    ) d
)

SELECT
    ROW_NUMBER() OVER(ORDER BY (SELECT NULL)) AS drug_era_id,
    d.person_id,
    d.drug_concept_id,
    d.drug_era_start_date,
    d.drug_era_end_date,
    d.drug_exposure_count,
    DATEDIFF(d.drug_era_start_date, LAG(d.drug_era_end_date, 1) OVER(PARTITION BY d.person_id, d.drug_concept_id ORDER BY d.drug_era_start_date)) AS gap_days
FROM
(
    SELECT
        person_id,
        ingredient_concept_id AS drug_concept_id,
        MIN(drug_exposure_start_date) AS drug_era_start_date,
        MAX(drug_exposure_end_date) AS drug_era_end_date,
        COUNT(*) AS drug_exposure_count
    FROM cte_drug_period
    GROUP BY person_id, ingredient_concept_id, partition_num
) d
"""
