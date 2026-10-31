# Study definitions

The analysis (`scripts/run_causal_study.sh`) knows no study. Everything that makes a study what it
is, the drugs to compare, who counts as a new user, the follow-up rules and the outcomes, is read from
a **study definition**: a folder with the files below. The studies bundled with the package are the
sub-folders of this folder; any other folder works the same way (`STUDY=/path/to/folder`).

| Bundled study | Content |
|---|---|
| `legend_hypertension/` | The hypertension study of OHDSI LEGEND: 80 exposures (58 drugs, 15 drug classes, 7 major classes), 45 outcomes (13 of its 58 cannot be shown in token sequences), 76 negative controls. Its `README.md` says how it was read from the LEGEND repository. |

## Files of a study folder

| File | Required | Content |
|---|---|---|
| `study.json` | yes | name, exposure levels, default pair, indication, cohort rules, time at risk, reference results |
| `exposures.csv` | yes | the drugs and the groups of drugs |
| `outcomes.csv` | yes | the outcomes and the negative controls (outcomes the exposures are not expected to affect) |
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
 "default_pair": {"target": "Thiazide or thiazide-like diuretics", "comparator": "ACE inhibitors"},
 "indication": {"name": "Hypertension", "concept_ids": [316866], "window_days": 365},
 "washout_days": 365,
 "era_gap_days": 30,
 "monotherapy_window_days": 7,
 "time_at_risk": {"min_days_at_risk": 1}
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
| `primary_outcomes` | Names of the outcomes the publication calls primary (optional). `compare_hazard_ratios.py` shows them first, then the other outcomes of `reference_results.csv` as safety outcomes, in the order of that file |

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

### `outcomes.csv`

One row is a concept set of an outcome. Columns:

```
name,kind,concept_ids,exact_concept_ids,exclude_concept_ids,exclude_exact_concept_ids,inpatient_or_er,note
Heart failure,primary,316139,,,,true,
Acute myocardial infarction,primary,4329847,,314666,,true,
Hemorrhagic stroke,primary,,376713;432923;439847,,,true,
All-cause mortality,primary,[DEATH],,,,false,
Rash,negative_control,140648,,,,,
```

| Column | Meaning |
|---|---|
| `name` | The outcome; the label in the results is its lower-case form with underscores. Rows with the same name are one outcome |
| `kind` | `primary` (default) or `negative_control`: an outcome the exposures are not expected to affect, so its hazard ratio should be 1; it is analysed like the others and `control_outcome_summary.csv` reports the bias |
| `concept_ids` | Concepts that count **with all their descendants** (looked up in the OMOP vocabulary at run time); several ids are separated by `;`. `[DEATH]` is the death of the patient |
| `exact_concept_ids` | Concepts that count only as themselves |
| `exclude_concept_ids` | Concepts taken out again, with their descendants |
| `exclude_exact_concept_ids` | Concepts taken out again, only as themselves |
| `inpatient_or_er` | `true` if the event only counts during an inpatient or ER visit (the visits with the concepts 262, 9201 or 9203); empty or `false` otherwise |
| `note` | Free text, for example what part of a source definition was left out. Not used by the analysis |

An event of an outcome is the first record, after the index date, of any concept of any of its rows. Each row
has its own exclusions, so an outcome of several rows is the union of its concept sets. The rows of an
outcome have the same `kind` and `inpatient_or_er`. Only `name` and `concept_ids` are needed; the other
columns may be left out of the file.

The token sequences hold a record as the token of its concept, whatever the domain (condition,
observation, procedure), so no domain is given. An outcome that depends on something they do not carry
(a measured value, a drug, a second confirming record, the cause of a death) has no row; if you still
want it, give the concept set you accept as its event and say so in `note`.

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
   reads `outcomes.csv`, lists the outcomes that need an inpatient or ER visit and reports mistakes in the file.
3. Run `STUDY=<folder> bash scripts/run_causal_study.sh`.
4. To bundle it, put the folder here and add it to the table above.

## `reference_results.csv` (optional)

Published results to compare a run with; one row per result.

```
target,comparator,outcome,time_at_risk,adjustment,source,hr,ci_lower,ci_upper,note
Thiazide or thiazide-like diuretics,ACE inhibitors,Stroke,intent_to_treat,stratified,CUMC,0.91,0.729,1.135,
```

| Column | Meaning |
|---|---|
| `target`, `comparator` | Exact names from `exposures.csv`. The results of the opposite comparison are inverted (1/hr, the interval inverted and swapped) |
| `outcome` | Name of the outcome |
| `time_at_risk` | `on_treatment` or `intent_to_treat`; the script prints the results of its own `TIME_AT_RISK` |
| `adjustment` | `stratified` or `matched` (propensity score) |
| `source` | The database. A result of the site that is replicated (for example `CUMC`) is used where there is one; `Meta-analysis` results are only counted at the end of a run |
| `hr`, `ci_lower`, `ci_upper` | Hazard ratio and 95% interval; `ci_lower` or `ci_upper` may be empty when the interval is cut off in the source |
| `note` | Free text |
