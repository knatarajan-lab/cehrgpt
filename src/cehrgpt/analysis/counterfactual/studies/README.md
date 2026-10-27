# Study definitions

The analysis (`scripts/run_causal_study.sh`) knows no study. Everything that makes a study what it
is, the drugs to compare, who counts as a new user, the follow-up rules and the outcomes, is read from
a **study definition**: a folder with the files below. The studies bundled with the package are the
sub-folders of this folder; any other folder works the same way (`STUDY=/path/to/folder`).

| Bundled study | Content |
|---|---|
| `legend_hypertension/` | The hypertension study of OHDSI LEGEND: 80 exposures (58 drugs, 15 drug classes, 7 major classes), 58 outcomes, 76 negative controls. Its `README.md` says how it was read from the LEGEND repository. |

## Files of a study folder

| File | Required | Content |
|---|---|---|
| `study.json` | yes | name, exposure levels, default pair, indication, cohort rules, time at risk, reference results |
| `exposures.csv` | yes | the drugs and the groups of drugs |
| `outcomes.json` | yes | the outcomes |
| `negative_controls.csv` | no | negative control outcomes (outcomes the exposures are not expected to affect) |
| `source.json` | no | where the study comes from (repository, commit, files), for provenance only |
| `README.md` | no | how the files were produced |

`python src/cehrgpt/analysis/counterfactual/study_definition.py --study <name or folder> --list`
checks that the folder can be read and shows its exposures.

### `study.json`

```json
{
 "name": "LEGEND-Hypertension",
 "description": "free text",
 "exposures": {"levels": ["Drug class", "Drug major class", "Drug"],
               "default_level": "Drug class", "drug_level": "Drug"},
 "default_pair": {"target": "ACE inhibitors", "comparator": "Thiazide or thiazide-like diuretics"},
 "indication": {"name": "Hypertension", "concept_ids": [316866], "window_days": 365},
 "washout_days": 365,
 "era_gap_days": 30,
 "monotherapy_window_days": 7,
 "time_at_risk": {"min_days_at_risk": 1},
 "reference_results": [{"target": "...", "comparator": "...", "label": "AMI",
                        "hr": 0.99, "ci_lower": 0.87, "ci_upper": 1.13}]
}
```

| Key | Meaning |
|---|---|
| `name` | Name of the study; its lower-case form labels the output folder |
| `exposures.levels` | The values of `type` in `exposures.csv`, in the order to list them |
| `exposures.default_level` | The level that `TARGET` and `COMPARATOR` are looked up in |
| `exposures.drug_level` | The level of the single drugs (ingredients). All drugs of this level are the *drugs of the study*: the target and the comparator use some, **every other one goes on the exclusion list** |
| `default_pair` | Target and comparator used when none is given (optional) |
| `indication` | A patient needs a record of one of `concept_ids` (descendants count) in the `window_days` days up to and including the index date. Optional: without it there is no indication filter |
| `washout_days` | Days of observed history needed before the index date |
| `era_gap_days` | Records of a drug at most this many days apart belong to one continuous exposure |
| `monotherapy_window_days` | Exactly one drug of the study may start from the index date up to this many days later; a combination product counts for each of its ingredients. `null`: the older same-visit checks |
| `time_at_risk.min_days_at_risk` | Patients with less time at risk are left out |
| `reference_results` | Published results for a pair (matched by the names of `target` and `comparator`); the script prints them at the end for that pair. Optional |

Other keys may be added (for example the source files); the code ignores what it does not use. A
value used as a default in the scripts may contain only letters, digits and `_ , . -`.

### `exposures.csv`

Columns `type,name,cohort_id,concept_id,included_concept_ids`, one row per exposure:

* `type`: one of `exposures.levels`;
* `name`: what `TARGET` and `COMPARATOR` are matched to (the exact name, not case sensitive, or a part
  that identifies exactly one name of the level);
* `cohort_id`: an id of the source study (not used by the analysis);
* `concept_id`: the OMOP concept of a single drug; empty for a group;
* `included_concept_ids`: for a group the concept ids of its drugs, separated by `;`; empty for a drug.
  Every id has to be a drug of `exposures.drug_level` if the group is to be used as target or comparator.

### `outcomes.json`

A list; each entry is an outcome reduced to what the pipeline can use:

```json
{
 "name": "Acute myocardial infarction", "cohort_id": 2, "atlas_id": 7364,
 "codesets": {"3": {"codeset_id": 3, "include": [], "include_descendants": [4329847],
                    "exclude": [], "exclude_descendants": [314666]},
              "2": {"codeset_id": 2, "include": [], "include_descendants": [262, 9201, 9203],
                    "exclude": [], "exclude_descendants": []}},
 "primary":    [{"domain": "Condition Occurrence", "codeset_ids": [3]}],
 "correlated": [{"domain": "Visit Occurrence", "codeset_ids": [2]}],
 "partial": {"kept_codeset_ids": [10], "left_out": "what the dropped branch needs"}
}
```

* `codesets`: concept sets by id. `include`: concepts that count as they are; `include_descendants`:
  concepts that count with all their descendants; `exclude` / `exclude_descendants` likewise for the
  concepts that are taken out again (descendants are looked up in the OMOP vocabulary at run time).
* `primary`: the criteria that select the events; `domain` is `Condition Occurrence`,
  `Observation`, `Procedure Occurrence`, `Drug Exposure`, `Measurement`, `Visit Occurrence` or `Death`
  (`Death` has no concept set).
* `correlated`: criteria that have to hold around the event. A `Visit Occurrence` entry (typically
  inpatient and ER visits) means the event only counts inside such a visit.
* `partial` (optional): only the listed concept sets are used because the other branch of the
  definition needs something the token sequences do not carry (a measured value). `left_out` says what
  was dropped and is recorded with the outcome.
* `cohort_id`, `atlas_id`: ids of the source study, not used by the analysis.

`outcome_definitions.py` decides what can be evaluated on the token sequences: records of a concept
set (condition, observation, procedure), optionally inside an inpatient or ER visit, and the death
record are supported; outcomes that need a confirming second record, a drug, a measured value or the
cause of a death are listed in Step 0 with the reason and left out.

### `negative_controls.csv`

Columns `cohort_id,concept_id,name`. A negative control is the first condition record of the concept
or any of its descendants.

## What the pipeline derives from a study

For a pair (`TARGET`, `COMPARATOR`) the study gives: the drugs of each; the exclusion list (all other
drugs of the study); the indication and its window; the washout, exposure gap and monotherapy window;
the minimum time at risk; the outcomes and negative controls; the reference results. The other
settings of a run (the design, the time-at-risk design, the preference-score trim, GPUs, sample sizes)
are options of the script.

## Adding a study

1. Create a folder with the files above. Write `README.md` in it to say how it was produced, so it can
   be produced again (see `legend_hypertension/README.md`).
2. Check it: `study_definition.py --study <folder> --list`, then
   `outcome_definitions.py --study <folder> --vocab_path <omop vocabulary> --output_dir <dir>`, which
   lists the outcomes that cannot be evaluated.
3. Run `STUDY=<folder> bash scripts/run_causal_study.sh`.
4. To bundle it, put the folder here and add it to the table above.
