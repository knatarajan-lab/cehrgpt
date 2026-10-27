#!/usr/bin/env bash
# =============================================================================
# LEGEND-HTN replication of one pair of drug classes (symmetric combined-population design)
# =============================================================================
#
# The two classes (TARGET_CLASS and COMPARATOR_CLASS, by default ACE inhibitors vs thiazide or
# thiazide-like diuretics), the ingredients of each, the exclusion list and the rules of the cohorts
# are the LEGEND-HTN definitions bundled with the package (legend_study.py), and the outcomes are
# the LEGEND outcomes (legend_outcomes.py).
#
# Study design
# ------------
#   Population  : ALL patients who initiated a drug of either class (new users: no
#                 antihypertensive before the index date, one ingredient in the first week,
#                 a hypertension diagnosis in the year before, a year of history).
#   Arm A (target): Each patient's drug initiation is stripped from their
#                 context and replaced by a frequency-sampled target concept.
#                 Model generates the future conditional on target.
#   Arm B (comparator): Same stripped context, but the drug slot is filled
#                 with a frequency-sampled comparator concept instead.
#                 Model generates the future conditional on comparator.
#
# Why this is symmetric
# ---------------------
#   In the original (asymmetric) design the study population is exclusively
#   target initiators.  Their pre-index history forms the context; the target
#   initiation is kept or swapped.  This means the model is always
#   conditioning on target patient characteristics.
#
#   Here we pool BOTH drug-class patients, strip the actual drug from
#   every patient's context, and inject the arm-assigned drug.  The same
#   set of pre-index histories is exposed to both treatments, eliminating
#   the population-composition asymmetry.
#
# Pipeline overview
# -----------------
#   Step 1a : Extract drug initiation sequences — target as source,
#             comparator as comparator (to learn comparator ingredient frequencies).
#   Step 1b : Extract drug initiation sequences — comparator as source,
#             target as comparator (to learn target ingredient frequencies).
#   Step 2  : Run create_drug_swap_context.py twice — once with target as
#             source (producing target_pts_target + target_pts_comparator contexts)
#             and once with comparator as source (comparator_pts_comparator +
#             comparator_pts_target).  Then concat into:
#               Arm A (target)     = target_pts_target + comparator_pts_target
#               Arm B (comparator) = target_pts_comparator + comparator_pts_comparator
#             Also merges both drug_info tables into one.
#   Step 3  : Generate N trajectories per patient per arm.
#   Step 4  : Estimate hazard ratios from generated trajectories.
#
# Outcomes
# --------
#   Step 0 reads the LEGEND-HTN outcomes and negative controls, taken from the SQL of the LEGEND
#   study (bundled with the package) and writes outcome_definitions.json.  The 15 outcomes that the study
#   links to an inpatient or ER visit (acute myocardial infarction, acute pancreatitis, acute
#   renal failure, anaphylactoid reaction, angioedema, cardiovascular disease,
#   gastrointestinal bleeding, hemorrhagic stroke, hospitalization with heart failure,
#   ischemic stroke, rhabdomyolysis, stroke, sudden cardiac death, transient ischemic attack
#   and unstable angina) only count inside such a visit, as does coronary heart disease in
#   the SQL.
#
# Published LEGEND-HTN results for ACE inhibitors vs thiazides, to compare with when those are the
# two classes (target hazard ratio, 95% CI):
#   AMI           : 0.99  (0.87–1.13)
#   Heart failure : 1.18  (1.07–1.30)
#   Stroke        : 1.09  (0.99–1.20)
#
# Usage
# -----
#   Edit the CONFIGURATION section, then:
#     bash run_legend_htn_acei_vs_thiazide_symmetric.sh
#
#   To skip steps already completed:
#     SKIP_STEP1=true SKIP_STEP2=true bash run_legend_htn_acei_vs_thiazide_symmetric.sh
#
#   To run both arms on a single GPU sequentially:
#     SINGLE_GPU=true bash run_legend_htn_acei_vs_thiazide_symmetric.sh
#
#
#   Another pair of classes (see legend_study.py --list):
#     TARGET_CLASS="Angiotensin receptor blockers (ARBs)" \
#     COMPARATOR_CLASS="Dihydropyridine calcium channel blockers (dCCB)" \
#       bash run_legend_htn_acei_vs_thiazide_symmetric.sh
# =============================================================================
set -euo pipefail

# =============================================================================
# STUDY — the two drug classes to compare
#
#   The classes, their ingredients, the exclusion list (every other antihypertensive ingredient)
#   and the rules of the cohorts (indication, washout, exposure gap, monotherapy window, minimum
#   time at risk) are the LEGEND-HTN definitions bundled with the package; see
#   `python src/cehrgpt/analysis/counterfactual/legend_study.py --list` for the classes.
#   Export a variable of the same name before running to change one of the rules.
# =============================================================================
TARGET_CLASS="${TARGET_CLASS:-ACE inhibitors}"
COMPARATOR_CLASS="${COMPARATOR_CLASS:-Thiazide or thiazide-like diuretics}"
STUDY_SCRIPT="src/cehrgpt/analysis/counterfactual/legend_study.py"
_STUDY_ENV="$(python "${STUDY_SCRIPT}" --target "${TARGET_CLASS}" --comparator "${COMPARATOR_CLASS}")"
eval "${_STUDY_ENV}"

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================

PATIENT_SEQUENCE_PATH="/data/patient_sequence"
VOCAB_PATH="/data/omop_vocab"
MODEL_PATH="/models/cehrgpt"
TOKENIZER_PATH="/models/tokenizer"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/legend_htn_${TARGET_LABEL}_vs_${COMPARATOR_LABEL}}"

# Number of stochastic trajectories per patient per arm
NUM_TRAJECTORIES=5

# Maximum patients to draw from each drug class before combining.
# Set to "" to use all patients from each class.
MAX_TARGET_PATIENTS=500
MAX_COMPARATOR_PATIENTS=500

BATCH_SIZE=8
GENERATION_INPUT_LENGTH=2048
GENERATION_MAX_NEW_TOKENS=1024
FOLLOW_UP_DAYS=365
NUM_WORKERS=4

# Number of parallel workers for Step 1 sequence extraction
EXTRACTION_NUM_WORKERS=10

# Minimum number of tokens required in the pre-drug context (Step 1)
MIN_CONTEXT_LENGTH=4

# Random seed for frequency-based drug sampling in Step 2
RANDOM_SEED=42

# Set to "true" to re-run Step 1 even when output files already exist
OVERWRITE=false

# ERA_GAP_DAYS (two drug events within this many days belong to the same era; the era end is
# stored in drug_info/ and bounds the follow-up in Step 4) and MIN_PRIOR_OBSERVATION_DAYS (washout:
# days of observed history before the index drug) come from the study definition above.

# LEGEND's cohort and time-at-risk rules (CreateExposureCohortsHypertension.sql and the
# cmAnalysisList of the study):
#   * monotherapy: exactly one antihypertensive ingredient may start in the first
#     MONOTHERAPY_WINDOW_DAYS days from the index date (set "" for the older same-visit checks);
#   * the hypertension diagnosis has to be recorded in the POPULATION_WINDOW_DAYS days up to the
#     index date;
#   * patients with the outcome before the index date are left out of that outcome;
#   * time at risk starts the day after the index date and needs MIN_DAYS_AT_RISK days; the
#     on-treatment design ends it when the continuous exposure to the patient's own class ends
#     (a gap above ERA_GAP_DAYS), the intent-to-treat design with the observation of the patient.
#     Starting another drug does not end it (CENSOR_AT_SWITCH=true does).
TIME_AT_RISK="${TIME_AT_RISK:-on_treatment}"      # or intent_to_treat
CENSOR_AT_SWITCH="${CENSOR_AT_SWITCH:-false}"

# Set to "true" to suppress the other arm's drugs and the exclusion drugs during generation.
# LEGEND does not (patients may add drugs), so the default is false.
SUPPRESS_CONCEPTS="${SUPPRESS_CONCEPTS:-false}"

# -----------------------------------------------------------------------------
# GPU configuration
# -----------------------------------------------------------------------------
GPU_TARGET=0
GPU_COMPARATOR=1
SINGLE_GPU=false

# Several GPUs for the generation step: "all" or a list such as "0,1,2,3".  One worker per GPU
# generates a share of the batches of both arms, so every GPU stays busy; an interrupted run
# continues where it stopped (rerun the script with SKIP_STEP1=true SKIP_STEP2=true).  Takes
# precedence over GPU_TARGET / GPU_COMPARATOR / SINGLE_GPU.  Empty = the settings above.
GPU_IDS="${GPU_IDS:-}"

# -----------------------------------------------------------------------------
# Skip flags — set to "true" to skip steps already completed
# -----------------------------------------------------------------------------
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false

# =============================================================================
# ARM NAMES
# =============================================================================
ARM_A="${TARGET_LABEL}"
ARM_B="${COMPARATOR_LABEL}"

# Outcomes: the LEGEND-HTN outcomes and negative controls (https://github.com/OHDSI/Legend),
# bundled with the package in cehrgpt/analysis/counterfactual/legend_hypertension (see source.json
# there for the LEGEND version).  Step 0 resolves their concepts (descendants, exclusions) with
# VOCAB_PATH and keeps the inpatient / ER visit restriction of the outcomes that need one;
# outcomes the token sequences cannot represent are listed in Step 0 and left out.

# Further outcomes by concept_id (descendants included), optional
OUTCOME_CONCEPT_IDS=""
# ... of which these must occur inside an inpatient or ER visit (9201/9203/262)
INPATIENT_OUTCOME_CONCEPT_IDS=""

# Optional control outcomes (leave empty for none).  A negative control is an outcome
# neither drug is expected to affect (expected HR = 1); a positive control has a known
# effect, given as concept_id or concept_id:expected_hr (e.g. "4329847:1.5").  They are
# extracted together with the primary outcomes and analysed in the same way; the HR
# summary labels them and control_outcome_summary.csv reports the systematic bias.
NEGATIVE_CONTROL_OUTCOME_IDS=""
POSITIVE_CONTROL_OUTCOME_IDS=""

# Everything the extraction step has to record (the :expected_hr suffixes are removed).
EXTRACT_OUTCOME_CONCEPT_IDS="${OUTCOME_CONCEPT_IDS}"
for _control_ids in "${NEGATIVE_CONTROL_OUTCOME_IDS}" "${POSITIVE_CONTROL_OUTCOME_IDS}"; do
    if [ -n "${_control_ids}" ]; then
        EXTRACT_OUTCOME_CONCEPT_IDS="${EXTRACT_OUTCOME_CONCEPT_IDS},$(echo "${_control_ids}" | sed 's/:[^,]*//g')"
    fi
done

# Overlap (positivity) check in the swap step: a patient is kept only when the model's
# preference score P(comparator class) / (P(source class) + P(comparator class)) at the
# index position lies in [trim, 1 - trim].  A patient that fails is dropped from every
# arm and from the observed analysis.  Set to "" to turn the check off.
PREFERENCE_SCORE_TRIM="${PREFERENCE_SCORE_TRIM:-0.05}"

# =============================================================================
# DERIVED PATHS
# =============================================================================

# Separate extraction output directories per drug class
TARGET_CONTEXT_DIR="${OUTPUT_ROOT}/target_contexts"
COMPARATOR_CONTEXT_DIR="${OUTPUT_ROOT}/comparator_contexts"

# Combined / symmetric contexts produced in Step 2
SYMMETRIC_DIR="${OUTPUT_ROOT}/symmetric_contexts"
TARGET_OVERLAP_SCORES="${SYMMETRIC_DIR}/target_pts_overlap_scores.parquet"
COMPARATOR_OVERLAP_SCORES="${SYMMETRIC_DIR}/comparator_pts_overlap_scores.parquet"
ARM_A_CTX="${SYMMETRIC_DIR}/${ARM_A}_arm_ctx.parquet"
ARM_B_CTX="${SYMMETRIC_DIR}/${ARM_B}_arm_ctx.parquet"
COMBINED_DRUG_INFO="${SYMMETRIC_DIR}/drug_info"

TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"
OUTCOME_DEFS_DIR="${OUTPUT_ROOT}/outcome_definitions"
OUTCOME_DEFINITIONS="${OUTCOME_DEFS_DIR}/outcome_definitions.json"
OUTCOME_SCRIPT="src/cehrgpt/analysis/counterfactual/legend_outcomes.py"

EXTRACT_SCRIPT="src/cehrgpt/analysis/counterfactual/extract_drug_initiation_sequences.py"
SWAP_SCRIPT="src/cehrgpt/analysis/counterfactual/create_drug_swap_context.py"
GENERATE_SCRIPT="src/cehrgpt/analysis/counterfactual/generate_counterfactual_sequences.py"
HR_SCRIPT="src/cehrgpt/analysis/counterfactual/hazard_ratio_estimation.py"

# =============================================================================
# HELPERS
# =============================================================================

_STEP_T0=0

_step_start() {
    _STEP_T0=$(date +%s)
}

_step_end() {
    local elapsed=$(( $(date +%s) - _STEP_T0 ))
    echo "Elapsed: $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
}

# =============================================================================
# STEP 0 — Outcome definitions of the LEGEND study (bundled with the package)
#
#   Writes outcome_definitions.json (the concepts of every outcome, with descendants and
#   exclusions resolved, and its inpatient / ER restriction) and the raw concept sets.
# =============================================================================
echo "============================================================"
echo "STEP 0: LEGEND outcome definitions  [$(date '+%H:%M:%S')]"
echo "============================================================"
mkdir -p "${OUTCOME_DEFS_DIR}"
python "${OUTCOME_SCRIPT}" \
    --indication  Hypertension \
    --vocab_path  "${VOCAB_PATH}" \
    --output_dir  "${OUTCOME_DEFS_DIR}"
echo ""

# =============================================================================
# STEP 1 — Extract drug initiation sequences for both drug classes
#
#   1a: target as source, comparator as comparator.
#       Produces target patients' treated_context and drug_info.
#       comparator patients in drug_info provide empirical comparator
#       ingredient frequencies needed for frequency-weighted sampling in Step 2.
#
#   1b: comparator as source, target as comparator.
#       Produces comparator patients' treated_context and drug_info.
#       target patients in drug_info provide empirical target ingredient
#       frequencies needed for frequency-weighted sampling in Step 2.
# =============================================================================
if [ "${SKIP_STEP1}" = "true" ]; then
    echo "Skipping Step 1 (SKIP_STEP1=true)"
else
    _EXTRACT_EXTRA_ARGS=()
    if [ "${OVERWRITE}" = "true" ]; then
        _EXTRACT_EXTRA_ARGS+=(--overwrite)
    fi
    if [ -n "${POPULATION_CONCEPT_IDS}" ]; then
        _EXTRACT_EXTRA_ARGS+=(--population_concept_ids "${POPULATION_CONCEPT_IDS}")
    fi
    if [ -n "${EXCLUSION_CONCEPT_IDS}" ]; then
        _EXTRACT_EXTRA_ARGS+=(--exclusion_concept_ids "${EXCLUSION_CONCEPT_IDS}")
    fi
    _EXTRACT_EXTRA_ARGS+=(--min_prior_observation_days "${MIN_PRIOR_OBSERVATION_DAYS}")
    if [ -n "${MONOTHERAPY_WINDOW_DAYS}" ]; then
        _EXTRACT_EXTRA_ARGS+=(--monotherapy_window_days "${MONOTHERAPY_WINDOW_DAYS}")
    fi
    if [ -n "${POPULATION_CONCEPT_IDS}" ] && [ -n "${POPULATION_WINDOW_DAYS}" ]; then
        _EXTRACT_EXTRA_ARGS+=(--population_window_days "${POPULATION_WINDOW_DAYS}")
    fi
    if [ "${CENSOR_AT_SWITCH}" = "true" ]; then
        _EXTRACT_EXTRA_ARGS+=(--censor_at_switch)
    fi

    # --- Step 1a: target patients ---
    echo "============================================================"
    echo "STEP 1a: Extract ${TARGET_NAME} patient sequences  [$(date '+%H:%M:%S')]"
    echo "  Source     : ${TARGET_NAME} (${TARGET_CONCEPT_IDS})"
    echo "  Comparator : ${COMPARATOR_NAME} (for frequency estimation)"
    echo "============================================================"

    mkdir -p "${TARGET_CONTEXT_DIR}"
    _step_start
    python "${EXTRACT_SCRIPT}" \
        --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${TARGET_CONCEPT_IDS}" \
        --comparator_concept_ids  "${COMPARATOR_CONCEPT_IDS}" \
        --output_dir              "${TARGET_CONTEXT_DIR}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
        --outcome_concept_ids     "${EXTRACT_OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --outcome_definitions_path "${OUTCOME_DEFINITIONS}" \
        --era_gap_days            "${ERA_GAP_DAYS}" \
        --min_context_length      "${MIN_CONTEXT_LENGTH}" \
        --num_workers             "${EXTRACTION_NUM_WORKERS}" \
        "${_EXTRACT_EXTRA_ARGS[@]}"
    _step_end
    echo "Step 1a complete."
    echo ""

    # --- Step 1b: comparator patients ---
    echo "============================================================"
    echo "STEP 1b: Extract ${COMPARATOR_NAME} patient sequences  [$(date '+%H:%M:%S')]"
    echo "  Source     : ${COMPARATOR_NAME} (${COMPARATOR_CONCEPT_IDS})"
    echo "  Comparator : ${TARGET_NAME} (for frequency estimation)"
    echo "============================================================"

    mkdir -p "${COMPARATOR_CONTEXT_DIR}"
    _step_start
    python "${EXTRACT_SCRIPT}" \
        --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${COMPARATOR_CONCEPT_IDS}" \
        --comparator_concept_ids  "${TARGET_CONCEPT_IDS}" \
        --output_dir              "${COMPARATOR_CONTEXT_DIR}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
        --outcome_concept_ids     "${EXTRACT_OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --outcome_definitions_path "${OUTCOME_DEFINITIONS}" \
        --era_gap_days            "${ERA_GAP_DAYS}" \
        --min_context_length      "${MIN_CONTEXT_LENGTH}" \
        --num_workers             "${EXTRACTION_NUM_WORKERS}" \
        "${_EXTRACT_EXTRA_ARGS[@]}"
    _step_end
    echo "Step 1b complete."
    echo ""
fi

# =============================================================================
# STEP 2 — Create symmetric arm contexts by running create_drug_swap_context.py
#           twice (once per drug class as the source), then concatenating.
#
#   Run 2a — target as source:
#     • target_pts_target_ctx.parquet   : target patients, their actual target drug
#                                     (treated path, written as-is)
#     • target_pts_comparator_ctx.parquet: target patients, comparator frequency-swapped
#
#   Run 2b — comparator as source:
#     • comparator_pts_comparator_ctx.parquet : comparator patients, actual comparator
#     • comparator_pts_target_ctx.parquet     : comparator patients, target freq-swapped
#
#   Arm A (target)      = concat(target_pts_target_ctx,     comparator_pts_target_ctx)
#   Arm B (comparator)  = concat(target_pts_comparator_ctx, comparator_pts_comparator_ctx)
#   drug_info         = concat(target_drug_info,         comparator_drug_info)
#
#   Both arms cover the same combined patient pool → symmetric comparison.
# =============================================================================
if [ "${SKIP_STEP2}" = "true" ]; then
    echo "Skipping Step 2 (SKIP_STEP2=true)"
else
    echo "============================================================"
    echo "STEP 2: Create symmetric arm contexts  [$(date '+%H:%M:%S')]"
    echo "  ${TARGET_NAME} patient contexts    : ${TARGET_CONTEXT_DIR}/treated_context"
    echo "  ${COMPARATOR_NAME} patient contexts: ${COMPARATOR_CONTEXT_DIR}/treated_context"
    echo "  Combined drug info       : ${COMBINED_DRUG_INFO}"
    echo "  Random seed              : ${RANDOM_SEED}"
    echo "============================================================"

    mkdir -p "${SYMMETRIC_DIR}"

    # Intermediate per-run output files
    _TARGET_PTS_TARGET_CTX="${SYMMETRIC_DIR}/target_pts_target_ctx.parquet"
    _TARGET_PTS_COMPARATOR_CTX="${SYMMETRIC_DIR}/target_pts_comparator_ctx.parquet"
    _COMPARATOR_PTS_COMPARATOR_CTX="${SYMMETRIC_DIR}/comparator_pts_comparator_ctx.parquet"
    _COMPARATOR_PTS_TARGET_CTX="${SYMMETRIC_DIR}/comparator_pts_target_ctx.parquet"

    _step_start

    # --- 2a: target patients → both arm contexts ---
    echo "  Step 2a: ${TARGET_NAME} patients (source=${TARGET_NAME}, comparator=${COMPARATOR_NAME}) …"
    _OVERLAP_TARGET_ARGS=()
    if [ -n "${PREFERENCE_SCORE_TRIM}" ]; then
        _OVERLAP_TARGET_ARGS=(
            --model_path "${MODEL_PATH}"
            --preference_score_trim "${PREFERENCE_SCORE_TRIM}"
            --overlap_scores_path "${TARGET_OVERLAP_SCORES}"
            --context_length "${GENERATION_INPUT_LENGTH}"
            --batch_size "${BATCH_SIZE}"
        )
    fi
    _MAX_TARGET_ARG=()
    if [ -n "${MAX_TARGET_PATIENTS:-}" ]; then
        _MAX_TARGET_ARG=(--max_source_patients "${MAX_TARGET_PATIENTS}")
    fi
    python "${SWAP_SCRIPT}" \
        --treated_context_path   "${TARGET_CONTEXT_DIR}/treated_context" \
        --drug_info_path         "${TARGET_CONTEXT_DIR}/drug_info" \
        --vocab_path             "${VOCAB_PATH}" \
        --tokenizer_path         "${TOKENIZER_PATH}" \
        --source_concept_ids     "${TARGET_CONCEPT_IDS}" \
        --comparator_concept_ids "${COMPARATOR_CONCEPT_IDS}" \
        --output_path            "${_TARGET_PTS_COMPARATOR_CTX}" \
        --also_write_treated_path "${_TARGET_PTS_TARGET_CTX}" \
        --seed                   "${RANDOM_SEED}" \
        "${_OVERLAP_TARGET_ARGS[@]}" \
        "${_MAX_TARGET_ARG[@]}"

    # --- 2b: comparator patients → both arm contexts ---
    echo "  Step 2b: ${COMPARATOR_NAME} patients (source=${COMPARATOR_NAME}, comparator=${TARGET_NAME}) …"
    _OVERLAP_COMPARATOR_ARGS=()
    if [ -n "${PREFERENCE_SCORE_TRIM}" ]; then
        _OVERLAP_COMPARATOR_ARGS=(
            --model_path "${MODEL_PATH}"
            --preference_score_trim "${PREFERENCE_SCORE_TRIM}"
            --overlap_scores_path "${COMPARATOR_OVERLAP_SCORES}"
            --context_length "${GENERATION_INPUT_LENGTH}"
            --batch_size "${BATCH_SIZE}"
        )
    fi
    _MAX_COMPARATOR_ARG=()
    if [ -n "${MAX_COMPARATOR_PATIENTS:-}" ]; then
        _MAX_COMPARATOR_ARG=(--max_source_patients "${MAX_COMPARATOR_PATIENTS}")
    fi
    python "${SWAP_SCRIPT}" \
        --treated_context_path   "${COMPARATOR_CONTEXT_DIR}/treated_context" \
        --drug_info_path         "${COMPARATOR_CONTEXT_DIR}/drug_info" \
        --vocab_path             "${VOCAB_PATH}" \
        --tokenizer_path         "${TOKENIZER_PATH}" \
        --source_concept_ids     "${COMPARATOR_CONCEPT_IDS}" \
        --comparator_concept_ids "${TARGET_CONCEPT_IDS}" \
        --output_path            "${_COMPARATOR_PTS_TARGET_CTX}" \
        --also_write_treated_path "${_COMPARATOR_PTS_COMPARATOR_CTX}" \
        --seed                   "${RANDOM_SEED}" \
        "${_OVERLAP_COMPARATOR_ARGS[@]}" \
        "${_MAX_COMPARATOR_ARG[@]}"

    # --- Concat arm contexts and drug_info ---
    echo "  Combining arm contexts and drug_info …"
    _TARGET_PTS_TARGET_CTX="${_TARGET_PTS_TARGET_CTX}" \
    _TARGET_PTS_COMPARATOR_CTX="${_TARGET_PTS_COMPARATOR_CTX}" \
    _COMPARATOR_PTS_COMPARATOR_CTX="${_COMPARATOR_PTS_COMPARATOR_CTX}" \
    _COMPARATOR_PTS_TARGET_CTX="${_COMPARATOR_PTS_TARGET_CTX}" \
    ARM_A_CTX="${ARM_A_CTX}" ARM_B_CTX="${ARM_B_CTX}" \
    COMBINED_DRUG_INFO="${COMBINED_DRUG_INFO}" \
    TARGET_CONTEXT_DIR="${TARGET_CONTEXT_DIR}" \
    COMPARATOR_CONTEXT_DIR="${COMPARATOR_CONTEXT_DIR}" \
    python - <<'PYEOF'
import polars as pl, os
from pathlib import Path

def read_dir_or_file(path):
    p = Path(path)
    if p.is_dir():
        return pl.read_parquet(str(p / "*.parquet"))
    return pl.read_parquet(str(p))

# Arm A (target): target patients with target + comparator patients with target swap
arm_a = pl.concat([
    pl.read_parquet(os.environ["_TARGET_PTS_TARGET_CTX"]),
    pl.read_parquet(os.environ["_COMPARATOR_PTS_TARGET_CTX"]),
])
arm_a.write_parquet(os.environ["ARM_A_CTX"])
print(f"  Arm A (target):     {len(arm_a):,} patients  → {os.environ['ARM_A_CTX']}")

# Arm B (comparator): target patients with comparator swap + comparator patients with comparator
arm_b = pl.concat([
    pl.read_parquet(os.environ["_TARGET_PTS_COMPARATOR_CTX"]),
    pl.read_parquet(os.environ["_COMPARATOR_PTS_COMPARATOR_CTX"]),
])
arm_b.write_parquet(os.environ["ARM_B_CTX"])
print(f"  Arm B (comparator): {len(arm_b):,} patients  → {os.environ['ARM_B_CTX']}")

# Combined drug_info
drug_info = pl.concat([
    read_dir_or_file(os.path.join(os.environ["TARGET_CONTEXT_DIR"], "drug_info")),
    read_dir_or_file(os.path.join(os.environ["COMPARATOR_CONTEXT_DIR"], "drug_info")),
])
di_out = os.environ["COMBINED_DRUG_INFO"]
Path(di_out).mkdir(parents=True, exist_ok=True)
drug_info.write_parquet(os.path.join(di_out, "drug_info.parquet"))
print(f"  Combined drug_info: {len(drug_info):,} rows  → {di_out}/drug_info.parquet")
PYEOF

    _step_end
    echo "Step 2 complete."
    echo ""
fi

# =============================================================================
# STEP 3 — Generate trajectories for both arms
# =============================================================================
if [ "${SKIP_STEP3}" = "true" ]; then
    echo "Skipping Step 3 (SKIP_STEP3=true)"
else
    echo "============================================================"
    echo "STEP 3: Generate trajectories  (${NUM_TRAJECTORIES} per patient per arm)  [$(date '+%H:%M:%S')]"
    if [ -n "${GPU_IDS}" ]; then
        echo "  Mode              : one worker per GPU (GPU_IDS=${GPU_IDS}), both arms in one run"
    elif [ "${SINGLE_GPU}" = "true" ]; then
        echo "  Mode              : sequential (SINGLE_GPU=true, GPU ${GPU_TARGET})"
    else
        echo "  GPU ${GPU_TARGET} → ${ARM_A}/     (conditioned on ${TARGET_NAME} initiation)"
        echo "  GPU ${GPU_COMPARATOR} → ${ARM_B}/ (conditioned on ${COMPARATOR_NAME} initiation)"
    fi
    echo "  Suppress concepts : ${SUPPRESS_CONCEPTS}"
    echo "  Era gap days      : ${ERA_GAP_DAYS}"
    echo "============================================================"

    mkdir -p "${TRAJ_DIR}"

    _SUPPRESS_ARGS_A=()
    _SUPPRESS_ARGS_B=()
    if [ "${SUPPRESS_CONCEPTS}" = "true" ]; then
        _SUPPRESS_CONCEPTS_A="${COMPARATOR_CONCEPT_IDS},${EXCLUSION_CONCEPT_IDS}"
        _SUPPRESS_CONCEPTS_B="${TARGET_CONCEPT_IDS},${EXCLUSION_CONCEPT_IDS}"
        _SUPPRESS_ARGS_A=(--vocab_path "${VOCAB_PATH}" --arm_suppress_concepts "${ARM_A}:${_SUPPRESS_CONCEPTS_A}")
        _SUPPRESS_ARGS_B=(--vocab_path "${VOCAB_PATH}" --arm_suppress_concepts "${ARM_B}:${_SUPPRESS_CONCEPTS_B}")
    fi

    _COMMON_GENERATE_ARGS=(
        --model_name_or_path        "${MODEL_PATH}"
        --tokenizer_path            "${TOKENIZER_PATH}"
        --output_dir                "${TRAJ_DIR}"
        --num_trajectories          "${NUM_TRAJECTORIES}"
        --batch_size                "${BATCH_SIZE}"
        --generation_input_length   "${GENERATION_INPUT_LENGTH}"
        --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}"
        --num_workers               "${NUM_WORKERS}"
    )

    _step_start

    if [ -n "${GPU_IDS}" ]; then
        _ARM_ARGS=(--arm_context "${ARM_A}:${ARM_A_CTX}" --arm_context "${ARM_B}:${ARM_B_CTX}")
        _MULTI_SUPPRESS_ARGS=()
        if [ "${SUPPRESS_CONCEPTS}" = "true" ]; then
            _MULTI_SUPPRESS_ARGS=(
                --vocab_path "${VOCAB_PATH}"
                --arm_suppress_concepts "${ARM_A}:${_SUPPRESS_CONCEPTS_A}"
                --arm_suppress_concepts "${ARM_B}:${_SUPPRESS_CONCEPTS_B}"
            )
        fi
        python "${GENERATE_SCRIPT}" \
            "${_ARM_ARGS[@]}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_MULTI_SUPPRESS_ARGS[@]}" \
            --gpu_ids "${GPU_IDS}" \
            > "${OUTPUT_ROOT}/generate.log" 2>&1
    elif [ "${SINGLE_GPU}" = "true" ]; then
        echo "Running ${ARM_A} arm …"
        CUDA_VISIBLE_DEVICES="${GPU_TARGET}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_A}:${ARM_A_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_A[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_A}.log" 2>&1
        echo "Running ${ARM_B} arm …"
        CUDA_VISIBLE_DEVICES="${GPU_TARGET}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_B}:${ARM_B_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_B[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_B}.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="${GPU_TARGET}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_A}:${ARM_A_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_A[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_A}.log" 2>&1 &
        PID_A=$!

        CUDA_VISIBLE_DEVICES="${GPU_COMPARATOR}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_B}:${ARM_B_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_B[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_B}.log" 2>&1 &
        PID_B=$!

        echo "${ARM_A} generation   PID=${PID_A}  (log: ${OUTPUT_ROOT}/generate_${ARM_A}.log)"
        echo "${ARM_B} generation   PID=${PID_B}  (log: ${OUTPUT_ROOT}/generate_${ARM_B}.log)"
        echo "Waiting for both to complete …"

        wait ${PID_A}
        STATUS_A=$?
        wait ${PID_B}
        STATUS_B=$?

        if [ ${STATUS_A} -ne 0 ]; then
            echo "ERROR: ${ARM_A} generation failed (exit ${STATUS_A}). Check ${OUTPUT_ROOT}/generate_${ARM_A}.log"
            exit ${STATUS_A}
        fi
        if [ ${STATUS_B} -ne 0 ]; then
            echo "ERROR: ${ARM_B} generation failed (exit ${STATUS_B}). Check ${OUTPUT_ROOT}/generate_${ARM_B}.log"
            exit ${STATUS_B}
        fi
    fi

    _step_end
    echo "Step 3 complete."
    echo ""
fi

# =============================================================================
# STEP 4 — Estimate Hazard Ratios (target vs comparator)
# =============================================================================
if [ "${SKIP_STEP4}" = "true" ]; then
    echo "Skipping Step 4 (SKIP_STEP4=true)"
else
    echo "============================================================"
    echo "STEP 4: Hazard ratio estimation  (${FOLLOW_UP_DAYS}-day follow-up)  [$(date '+%H:%M:%S')]"
    echo "  Arm A (treated)    : ${ARM_A}"
    echo "  Arm B (comparator) : ${ARM_B}"
    echo "  Drug info          : ${COMBINED_DRUG_INFO}"
    echo "  Outcomes: ${OUTCOME_DEFINITIONS}"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    _step_start
    _HR_EXTRA_ARGS=()
    _HR_EXTRA_ARGS+=(--prior_outcome_path "${TARGET_CONTEXT_DIR}/observed_outcomes" "${COMPARATOR_CONTEXT_DIR}/observed_outcomes")
    _HR_EXTRA_ARGS+=(--time_at_risk "${TIME_AT_RISK}" --min_days_at_risk "${MIN_DAYS_AT_RISK}")
    if [ "${CENSOR_AT_SWITCH}" = "true" ]; then
        _HR_EXTRA_ARGS+=(--censor_at_switch)
    fi
    if [ -n "${NEGATIVE_CONTROL_OUTCOME_IDS}" ]; then
        _HR_EXTRA_ARGS+=(--negative_control_outcome_ids "${NEGATIVE_CONTROL_OUTCOME_IDS}")
    fi
    if [ -n "${POSITIVE_CONTROL_OUTCOME_IDS}" ]; then
        _HR_EXTRA_ARGS+=(--positive_control_outcome_ids "${POSITIVE_CONTROL_OUTCOME_IDS}")
    fi
    if [ -n "${PREFERENCE_SCORE_TRIM}" ]; then
        _HR_EXTRA_ARGS+=(--overlap_scores_path "${TARGET_OVERLAP_SCORES}" "${COMPARATOR_OVERLAP_SCORES}")
    fi
    python "${HR_SCRIPT}" \
        --trajectories_dir        "${TRAJ_DIR}" \
        --drug_info_path          "${COMBINED_DRUG_INFO}" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --outcome_definitions_path "${OUTCOME_DEFINITIONS}" \
        --follow_up_days          "${FOLLOW_UP_DAYS}" \
        --arm_a                   "${ARM_A}" \
        --arm_b                   "${ARM_B}" \
        --output_dir              "${RESULTS_DIR}" \
        --vocab_path              "${VOCAB_PATH}" \
        --arm_a_concept_ids       "${TARGET_CONCEPT_IDS}" \
        --arm_b_concept_ids       "${COMPARATOR_CONCEPT_IDS}" \
        --exclusion_concept_ids   "${EXCLUSION_CONCEPT_IDS}" \
        --era_gap_days            "${ERA_GAP_DAYS}" \
        "${_HR_EXTRA_ARGS[@]}"
    _step_end

    echo "Step 4 complete."
    echo ""
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo "============================================================"
echo "Done.  Output directory: ${OUTPUT_ROOT}"
echo ""
echo "Key files:"
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv   (generated ${TARGET_NAME} vs generated ${COMPARATOR_NAME})"
echo "  ${RESULTS_DIR}/control_outcome_summary.csv   (negative / positive controls, when given)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
if [ "${TARGET_CLASS}" = "ACE inhibitors" ] && [ "${COMPARATOR_CLASS}" = "Thiazide or thiazide-like diuretics" ]; then
    echo "Published LEGEND-HTN reference HRs (ACE inhibitors vs thiazides):"
    echo "  AMI           : 0.99  (95% CI 0.87–1.13)"
    echo "  Heart failure : 1.18  (95% CI 1.07–1.30)"
    echo "  Stroke        : 1.09  (95% CI 0.99–1.20)"
fi
echo "============================================================"
