# ORAL Surveillance study definition

This folder emulates the ORAL Surveillance trial (NCT02092467; Ytterberg et al., *N Engl J Med*
2022;386:316-326) in the format described in `../README.md`. The trial randomized patients with
rheumatoid arthritis to tofacitinib 5 mg or 10 mg twice daily or to a TNF inhibitor (adalimumab or
etanercept), all on background methotrexate, and tested whether tofacitinib was non-inferior for
major adverse cardiovascular events (MACE) and for cancer excluding non-melanoma skin cancer. It was
not: the hazard ratios of the combined tofacitinib doses were 1.33 (95% CI 0.91–1.94) for MACE and
1.48 (1.04–2.09) for cancer, the upper limits above the margin of 1.8.

Unlike `legend_hypertension/`, these files were written by hand, not read from a source repository.

## 1. Files

| File | Content |
|---|---|
| `study.json` | Indication (rheumatoid arthritis), washout, exposure gap, monotherapy window, time at risk |
| `exposures.csv` | Tofacitinib, the five TNF inhibitors, and the other advanced RA therapies (exclusion list only) |
| `outcomes.csv` | The two primary outcomes, safety outcomes reported by the trial, 20 negative controls |
| `reference_results.csv` | The two primary results of the trial |
| `source.json` | The trial and the publication |

## 2. Exposures and cohorts

| Trial | Here |
|---|---|
| Tofacitinib 5 mg or 10 mg twice daily | `Tofacitinib` (one ingredient; the sequences do not carry the dose, so the arms are the combined doses) |
| Adalimumab 40 mg every 2 weeks, or etanercept 50 mg weekly | `TNF inhibitors (adalimumab or etanercept)`, the default comparator |
| | `TNF inhibitors (all)` adds infliximab, certolizumab pegol and golimumab, for a larger cohort: `COMPARATOR="TNF inhibitors (all)"` |
| Rheumatoid arthritis | Indication: a record of 80809 (Rheumatoid arthritis) or a descendant in the 365 days up to the index date |
| Age 50 or older | `min_age`: 50 on the index date (the `ages` value of the index drug) |
| At least one cardiovascular risk factor | `inclusion`: a record, any time before the index drug, of one of the concepts below |
| Randomized assignment | New users; no use of any drug of the study (all rows of level `Drug`) before the index date, exactly one starting in the first 7 days |

Every drug of the study that is not in the pair is on the exclusion list. With the default pair that
is the other TNF inhibitors, the other JAK inhibitors (baricitinib, upadacitinib), abatacept,
tocilizumab, sarilumab, rituximab and anakinra. So the cohorts are patients starting their first
advanced therapy. The trial allowed earlier biologics other than a failed TNF inhibitor, so it is a
broader population on this point.

`era_gap_days` is 60 rather than LEGEND's 30, so that the dosing interval of the injected drugs
(up to 8 weeks for infliximab in `TNF inhibitors (all)`) does not end an exposure. It matters only
for `TIME_AT_RISK=on_treatment`; the trial's analysis was close to intent-to-treat.

## 3. Cardiovascular risk factors, and what is left out

The `inclusion` rule of `study.json` asks for one of these records, descendants included, any time
before the index drug:

| Trial risk factor | Concepts |
|---|---|
| Current cigarette smoker | 437264 Tobacco dependence syndrome, 4209423 Nicotine dependence, 40766945 Current smoker |
| Hypertension | 316866 Hypertensive disorder |
| Hyperlipidemia | 432867 Hyperlipidemia, 4029305 Hypercholesterolemia |
| Diabetes mellitus | 201820 Diabetes mellitus |
| History of coronary artery disease (MI, revascularization, cardiac arrest, unstable angina, ACS) | 4185932 Ischemic heart disease, 314666 Old myocardial infarction, 4215140 Acute coronary syndrome, 321042 Cardiac arrest, 4336464 Coronary artery bypass graft, 4216130 Percutaneous coronary intervention |
| Family history of premature coronary heart disease | 4182335 Family history of ischemic heart disease, 4326027 Family history of coronary arteriosclerosis (the age of the relative is not recorded) |

A record anywhere in the history counts, so a smoker who stopped years ago counts as a current
smoker. Two trial risk factors are left out: HDL cholesterol below 40 mg/dL (a measured value) and
extra-articular RA disease.

**Background methotrexate** has no rule yet: there is none for a drug that has to be taken alongside
the index drug.

## 4. Outcomes

| Outcome | Definition |
|---|---|
| Major adverse cardiovascular events (primary) | Inpatient or ER acute myocardial infarction or stroke (the LEGEND concept sets). The trial also counts cardiovascular death; the sequences carry no cause of death, and an outcome cannot mix the death token with an inpatient-only concept set, so death is left out and analysed as `All-cause mortality` |
| Malignancy excluding non-melanoma skin cancer (primary) | 443392 (Malignant neoplastic disease) and descendants, without basal and squamous cell carcinoma of skin |
| Acute myocardial infarction, Stroke, Venous thromboembolic events | From `legend_hypertension/outcomes.csv` |
| All-cause mortality | The death token |
| Herpes zoster | 443943 and descendants. LEGEND has 440329 (Herpes zoster without complication, non-standard) as a negative control, but zoster is an expected effect of tofacitinib here |
| Pneumonia, Sepsis | Inpatient or ER; they stand in for the trial's serious infections |

The 20 negative controls are taken from the LEGEND list, without the ones that tofacitinib, a TNF
inhibitor or rheumatoid arthritis itself may affect: infections (herpes zoster, HPV, onychomycosis,
vaginitis), inflammatory and immune conditions (Crohn's disease, contact dermatitis, tear film
insufficiency), conditions of the joints and hands, smoking, and states that follow surgery.

## 5. Checks before a run

Every concept id was looked up in a CDM 5.4 vocabulary (`CONCEPT.csv`): the drugs are the
standard RxNorm ingredients of their names, the conditions standard SNOMED concepts of their names.
On the machine of the run, check that its vocabulary resolves them too:

1. `study_definition.py --study oral_surveillance --list` and `--reference`.
2. `outcome_definitions.py --study oral_surveillance --vocab_path <vocabulary> --output_dir <dir>`.

## 6. Running it

```bash
STUDY=oral_surveillance FOLLOW_UP_DAYS=1095 bash scripts/run_causal_study.sh
```

The trial followed patients for a median of about 4 years; the default `FOLLOW_UP_DAYS=365` will see
few MACE and cancer events. A longer follow-up needs `GENERATION_MAX_NEW_TOKENS` large enough for the
generated trajectories to reach it.
