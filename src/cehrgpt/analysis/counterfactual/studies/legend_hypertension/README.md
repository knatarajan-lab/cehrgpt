# LEGEND-HTN study definition: how it was read from the LEGEND repository

This folder holds the definition of the OHDSI LEGEND hypertension study
(<https://github.com/OHDSI/Legend>) as data, in the format described in `../README.md`. **No code parses the LEGEND repository.** When the
definition has to be read again (a new LEGEND version, another indication, a file that looks wrong),
an agent does it by following this document, then checks the result with the list at the end.

An earlier version of this repository had regular-expression code that did steps 2 and 3 of the
outcome section. It was removed on purpose. If a reference is useful, it is in the history:
`git show 97bd43c:src/cehrgpt/analysis/counterfactual/legend_outcomes.py`.

## 1. What is in this folder

| File | Content | Read by |
|---|---|---|
| `outcomes.json` | The 58 hypertension outcomes: concept sets and the criteria that select the events | `outcome_definitions.read_outcomes` |
| `negative_controls.csv` | The 76 negative control outcomes | `outcome_definitions.negative_controls` |
| `exposures.csv` | The 58 ingredients, 15 drug classes and 7 major classes of the study | `study_definition.load_exposures` |
| `study.json` | Indication, washout, exposure gap, monotherapy window, time at risk | `study_definition.load_study` |
| `source.json` | Which LEGEND commit all of the above comes from | tests, provenance |

The code turns these into what the pipeline uses (`outcome_definitions.resolve_outcomes` expands the
concept sets with the OMOP vocabulary; `study_definition.environment` gives the shell variables of the
scripts). Do not edit the data files by hand; read them again from LEGEND as described below.

## 2. Source and version

Work in a checkout of `https://github.com/OHDSI/Legend`. Record the version in `source.json`:

* `commit`: output of `git rev-parse HEAD`
* `commit_date`: output of `git log -1 --format=%cI`
* `repository`, `indication` (`Hypertension`), `files` (the LEGEND files listed in sections 3 to 6),
  `n_outcomes`, `n_negative_controls`
* `generated_by`: a short note that the files were read by an agent following this document

The files used are all under `inst/settings`, `inst/sql/sql_server` and `R` of the checkout.

## 3. Outcomes (`outcomes.json`)

### 3.1 Which outcomes, and their ids

Open `inst/settings/OutcomesOfInterest.csv`. Keep the rows with `indicationId == Hypertension`
(58 rows), in the order of the file. For each row:

* `name` is the name of the outcome **and** the name of its SQL file,
  `inst/sql/sql_server/<name>.sql`. Use the name exactly as written: one name,
  `Venous thromboembolic events `, ends in a space, and so does its file name.
* `cohortId` becomes `cohort_id` and `atlasId` becomes `atlas_id` (integers).

### 3.2 The concept sets of an outcome

Each outcome file is a generated ATLAS cohort definition. It starts with `CREATE TABLE #Codesets` and
has one `INSERT INTO #Codesets` per concept set, in this shape:

```sql
INSERT INTO #Codesets (codeset_id, concept_id)
SELECT 3 as codeset_id, c.concept_id FROM (select distinct I.concept_id FROM
(
  select concept_id from @vocabulary_database_schema.CONCEPT where concept_id in (4329847)and invalid_reason is null
UNION  select c.concept_id
  from @vocabulary_database_schema.CONCEPT c
  join @vocabulary_database_schema.CONCEPT_ANCESTOR ca on c.concept_id = ca.descendant_concept_id
  and ca.ancestor_concept_id in (4329847)
  and c.invalid_reason is null
) I
LEFT JOIN
(
  select concept_id from @vocabulary_database_schema.CONCEPT where concept_id in (314666)and invalid_reason is null
UNION  select c.concept_id
  ... and ca.ancestor_concept_id in (314666) ...
) E ON I.concept_id = E.concept_id
WHERE E.concept_id is null
) C;
```

Read it like this. The number after `SELECT` is the `codeset_id`. The part before `LEFT JOIN` (`I`) are
the included concepts; the part after it (`E`) the excluded ones. There is no `LEFT JOIN` when
nothing is excluded. In each part:

* the first `select`, `concept_id in (L)`, lists **every** concept of the set;
* the second `select` (after `UNION`), `ancestor_concept_id in (D)`, lists the concepts that count
  **with all their descendants**.

So `include_descendants = D` and `include = L minus D`. Do the same for the excluded part
(`exclude_descendants`, `exclude`). Lists are sorted, without duplicates, as integers.
(`invalid_reason is null` only drops invalid vocabulary concepts and needs no record.) The
example above gives
`{"include": [], "include_descendants": [4329847], "exclude": [], "exclude_descendants": [314666]}`.

Check that no file contains `concept_relationship`: LEGEND uses no "mapped" concepts. If one does,
stop and tell the user, the format has no place for it.

### 3.3 The criteria that select the events

After the concept sets the file has `with primary_events ... as ( -- Begin Primary Events ...
-- End Primary Events )`.

* **`primary`**: inside `-- Begin Primary Events` ... `-- End Primary Events`, list every domain
  block `-- Begin <Domain> Criteria` ... `-- End <Domain> Criteria`, in order, that is **not** inside a
  `-- Begin Correlated Criteria` block. `<Domain>` is `Condition Occurrence`, `Observation`,
  `Procedure Occurrence`, `Drug Exposure`, `Measurement`, `Visit Occurrence` or `Death`. Its
  `codeset_ids` are the numbers in `codesets.codeset_id = N` inside the block, in order of
  appearance without repeats (`[]` for `Death`, which has no concept set).
  Several blocks joined by `UNION ALL` are several entries.
* **`correlated`**: every `-- Begin Correlated Criteria` ... `-- End Correlated Criteria` block in
  the **whole file** (after the primary events for most outcomes, inside them for the
  "Hospitalization with ..." outcomes). List the domain blocks inside it, including nested ones, as
  for `primary`. Drop an entry that repeats an earlier one exactly.

Example, "Acute myocardial infarction": a condition (concept set 3) during a visit of concept set 2.

```json
{
 "name": "Acute myocardial infarction", "cohort_id": 2, "atlas_id": 7364,
 "codesets": {
  "2": {"codeset_id": 2, "include": [], "include_descendants": [262, 9201, 9203], "exclude": [], "exclude_descendants": []},
  "3": {"codeset_id": 3, "include": [], "include_descendants": [4329847], "exclude": [], "exclude_descendants": [314666]}
 },
 "primary": [{"domain": "Condition Occurrence", "codeset_ids": [3]}],
 "correlated": [{"domain": "Visit Occurrence", "codeset_ids": [2]}]
}
```

The file is a JSON list of such objects, in the order of `OutcomesOfInterest.csv`, written with an
indent of 1 and a final newline. The keys of `codesets` are the codeset ids as strings.

### 3.4 What is deliberately not recorded

The domain and concept sets are all that the pipeline can use. The following exist in the SQL and are
left out, which is why the code (`outcome_definitions.resolve_outcome`) marks an outcome as not supported when its
meaning depends on them:

* counts such as `HAVING COUNT(...) >= 2` ("followed by another record in the following year");
* value ranges of measurements (`value_as_number`, `range_high`);
* time windows between records, the gap that merges records into episodes, date offsets.

Do not try to encode them in `outcomes.json`.

### 3.5 Outcomes with a branch that needs a measured value (`partial`)

Two outcomes have a branch that cannot be evaluated on the token sequences because it depends on a
measured value, next to a branch that can. Add a `partial` entry to them in `outcomes.json`:

```json
"partial": {"kept_codeset_ids": [10], "left_out": "muscle disorder records with a creatine kinase above 5 x the upper limit of normal"}
```

| Outcome | Kept concept sets | Left out |
|---|---|---|
| Rhabdomyolysis | `[10]` (rhabdomyolysis records) | muscle disorder records with a creatine kinase above 5 x the upper limit of normal (concept set 11 with the measurement 12) |
| Hyperkalemia | `[0]` (hyperkalemia records) | potassium measurements above 5.6 mmol/L (the measurement criterion with concept set 1) |

This is a judgement, not something that can be read off mechanically: it applies when a primary
criterion of the measurement domain, or a correlated measurement, belongs to one branch of a `UNION ALL`
of primary criteria, and the other branch is a plain record of a concept set. Look for it when the
checks in section 7 report an unsupported outcome with a `Measurement` criterion, and tell the user
which outcomes got a `partial` entry and why.

## 4. Negative controls (`negative_controls.csv`)

Open `inst/settings/NegativeControls.csv` (columns `indicationId,conceptId,cohortId,name`). Keep the
rows with `indicationId == Hypertension` (76) and write `cohort_id,concept_id,name` (header in that
order), in file order, from `cohortId`, `conceptId` and `name`.

`inst/sql/sql_server/NegativeControls.sql` defines every negative control the same way: the first
condition record of the concept or any descendant. There is no SQL per control.

## 5. Exposures (`exposures.csv`)

Open `inst/settings/ExposuresOfInterest.csv`. Keep the rows with `indicationId == Hypertension` (80)
and write `type,name,cohort_id,concept_id,included_concept_ids` from `type`, `name`, `cohortId`,
`conceptId` and `includedConceptIds`, in file order:

* `type` is `Drug` (an ingredient: 58 rows), `Drug class` (15) or `Drug major class` (7);
* `concept_id` is empty for classes;
* `included_concept_ids` keeps LEGEND's `;`-separated ingredient ids of a class, and is empty for a
  drug.

## 6. Study constants (`study.json`)

Each value is a number that appears once in a file of LEGEND; read it from there, do not remember it.

| Field | Where | What to look for |
|---|---|---|
| `name` | | `LEGEND-<indication>` |
| `description` | | one sentence, free text |
| `exposures` | `exposures.csv` | `levels`: the values of `type` in the order `Drug class`, `Drug major class`, `Drug`; `default_level`: `Drug class`; `drug_level`: `Drug` |
| `default_pair` | the previous analysis script | `ACE inhibitors` vs `Thiazide or thiazide-like diuretics` (exact names from `exposures.csv`) |
| `reference_results` | the previous analysis script | the three results that script listed for that pair: AMI 0.99 (0.87–1.13), Heart failure 1.18 (1.07–1.30), Stroke 1.09 (0.99–1.20); they are copied, not read from LEGEND, so say where they came from if the user asks |
| `indication.name` | | the indication, `Hypertension` |
| `indication.concept_ids` | `inst/sql/sql_server/CreateExposureCohortsHypertension.sql`, last block | `WHERE condition_concept_id IN (SELECT descendant_concept_id ... WHERE ancestor_concept_id IN (316866))`: the ids in the `IN (...)` (descendants count too) |
| `indication.window_days` | same block | `condition_start_date >= DATEADD(DAY, -365, new_user.cohort_start_date)` together with `condition_start_date <= new_user.cohort_start_date`: 365 days up to and including the index date |
| `washout_days` | `R/CreateExposureCohorts.R` | `washout_period = 365` (it appears in two calls, with the same value; the SQL default `@washout_period` agrees) |
| `era_gap_days` | `R/CreateExposureCohorts.R` | `max_gap = 30` (same) |
| `monotherapy_window_days` | `CreateExposureCohortsHypertension.sql`, `#mono_new_user` | `concurrent.cohort_start_date <= DATEADD(DAY, 7, first_line.cohort_start_date)`: the 7 |
| `time_at_risk.risk_window_start_day` | `inst/settings/cmAnalysisListHypertension.json`, analysis `PS stratification, on-treatment` | `createStudyPopArgs.riskWindowStart` |
| `time_at_risk.min_days_at_risk` | same analysis | `createStudyPopArgs.minDaysAtRisk` |
| `time_at_risk.remove_subjects_with_prior_outcome` | same analysis | `createStudyPopArgs.removeSubjectsWithPriorOutcome` |
| `time_at_risk.intent_to_treat_min_days_at_risk` | analysis `PS stratification, intent-to-treat` | `createStudyPopArgs.minDaysAtRisk` |
| `source_files` | | the four LEGEND files above, as paths in the repository |

What the cohort rules of `CreateExposureCohortsHypertension.sql` mean, for orientation (the pipeline
implements them in `extract_drug_initiation_sequences.py`):

1. **Exposure eras**: drug records of an ingredient or class are merged into continuous eras when
   the gap between them is at most `max_gap` days.
2. **First use** of an ingredient or class is the start of its first era, and needs `washout_period`
   days of observation before it.
3. **First line**: no era of any exposure of the study starts before the first use.
4. **Monotherapy**: exactly one ingredient (`exposure_type = 'Drug'`) has an era starting from the
   index date up to the monotherapy window after it. A combination product counts for each ingredient.
5. **Indication**: a record of the indication concept in the year up to the index date.

## 7. Checks after reading

Run these, and say so in the answer if any differs from what is written here (and why):

1. `outcomes.json` has 58 outcomes with 58 different `cohort_id`; `negative_controls.csv` has 76 rows;
   `exposures.csv` has 58 `Drug`, 15 `Drug class` and 7 `Drug major class` rows; `source.json` agrees
   with the counts.
2. "Acute myocardial infarction": concept set 3 is `include_descendants [4329847]` with
   `exclude_descendants [314666]`; concept set 2 is `include_descendants [262, 9201, 9203]`; one
   primary `Condition Occurrence [3]`; one correlated `Visit Occurrence [2]`.
3. "Hemorrhagic stroke": concept set 4 is `include [376713, 432923, 439847]` and has **no**
   descendants.
4. "Hospitalization with heart failure": two primary `Visit Occurrence` entries, and a correlated
   `Condition Occurrence`.
5. "All-cause mortality": primary `Death` with no concept set. "Rhabdomyolysis" and "Hyperkalemia" carry a `partial` entry (section 3.5).
6. The ACE inhibitor class has the ten ingredients `1335471, 1340128, 1341927, 1363749, 1308216,
   1310756, 1373225, 1331235, 1334456, 1342439`, the class of thiazide or thiazide-like diuretics
   `1395058, 974166, 978555, 907013`, and the other 44 ingredients make up the exclusion list (ARBs,
   dihydropyridine and non-dihydropyridine calcium channel blockers, second-line agents).
7. `study.json` has 316866, 365, 365, 30, 7, 1, 1.
8. The tests pass: `pytest tests/unit_tests/analysis/counterfactual/outcome_definitions_test.py
   tests/unit_tests/analysis/counterfactual/study_definition_test.py`. They check the points above, that
   15 outcomes of the paper (Supplementary Table of the LEGEND-HTN paper, the outcomes "during an
   inpatient or ER visit") come out restricted to inpatient and ER visits, and that 121 of the 134
   outcomes and controls can be evaluated on the token sequences.

If the new counts or concept sets differ from the old files, show the user a diff of the data files
before replacing them: a changed outcome changes every result downstream.
