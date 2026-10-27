#!/usr/bin/env bash
# =============================================================================
# Causal analysis of one pair of exposures, driven by a study definition
# =============================================================================
#
# The script knows no study.  A study definition (STUDY: the name of a bundled study such as
# legend_hypertension, or a folder; format in src/cehrgpt/analysis/counterfactual/studies/README.md)
# gives the exposures and the pair to compare, the cohort rules (indication, washout, exposure gap,
# monotherapy window, minimum time at risk), the exclusion list (every other drug of the study) and the
# outcomes with their negative controls.  TARGET and COMPARATOR pick two of its exposures by name; by
# default the pair of the study.  The cohorts are new users of the target or the comparator.
#
# Design (DESIGN)
# ---------------
#   symmetric (default)
#     The patients of BOTH classes are pooled.  Each patient's history is run once with the
#     target drug and once with the comparator (a frequency-sampled ingredient of the class) in
#     place of the actual drug, so both arms cover the same combined population and the
#     population-composition asymmetry of the other design is gone.
#       Arm A (target)     = target patients as they are + comparator patients swapped to target
#       Arm B (comparator) = target patients swapped to comparator + comparator patients as they are
#
#   target_initiators
#     Only the patients who initiated the target class.
#       Arm A (target)     = their histories with the target drug they received
#       Arm B (comparator) = the same histories with the drug swapped to a comparator ingredient
#                            (sampled by the frequency among the real comparator patients)
#     The comparator patients are extracted too: they give the ingredient frequencies and the
#     observed comparator cohort.
#
#   In both designs each patient produces N trajectories per arm, the swap step checks overlap
#   (a patient the model finds implausible for the other drug is dropped from every arm), and the
#   observed hazard ratio of the real target and comparator cohorts is computed for reference.
#
# Pipeline
# --------
#   Step 0  : the outcomes of the study, resolved for the vocabulary
#   Step 1  : extract drug initiation sequences (symmetric: once per exposure as the source)
#   Step 2  : build the arm contexts by swapping the drug (symmetric: in both directions)
#   Step 3  : generate N trajectories per patient per arm
#   Step 4  : estimate hazard ratios from the generated trajectories, and the observed ones
#
# Usage
# -----
#   Edit the CONFIGURATION section, then:
#     bash scripts/run_causal_study.sh
#
#   The other design:
#     DESIGN=target_initiators bash scripts/run_causal_study.sh
#
#   Another pair of exposures of the study (python .../study_definition.py --list shows them):
#     TARGET="Angiotensin receptor blockers (ARBs)" \
#     COMPARATOR="Dihydropyridine calcium channel blockers (dCCB)" \
#       bash scripts/run_causal_study.sh
#
#   Another study:
#     STUDY=/path/to/my_study bash scripts/run_causal_study.sh
#
#   To skip steps already completed (e.g. after a failure in step 3):
#     SKIP_STEP1=true SKIP_STEP2=true bash scripts/run_causal_study.sh
#
#   To run both arms on a single GPU sequentially:
#     SINGLE_GPU=true bash scripts/run_causal_study.sh
#
# =============================================================================
set -euo pipefail

# =============================================================================
# STUDY — the study definition and the two exposures to compare
#
#   Everything about the drugs, the cohorts and the outcomes comes from the study definition.
#   Export a variable of the same name before running to change one of its rules (the windows,
#   the washout, the labels of the arms).
# =============================================================================
STUDY="${STUDY:-legend_hypertension}"
TARGET="${TARGET:-}"                  # empty: the target of the default pair of the study
COMPARATOR="${COMPARATOR:-}"          # empty: the comparator of the default pair
EXPOSURE_LEVEL="${EXPOSURE_LEVEL:-}"  # e.g. "Drug class"; empty: the default level of the study
STUDY_SCRIPT="src/cehrgpt/analysis/counterfactual/study_definition.py"
_STUDY_ARGS=(--study "${STUDY}")
if [ -n "${TARGET}" ]; then _STUDY_ARGS+=(--target "${TARGET}"); fi
if [ -n "${COMPARATOR}" ]; then _STUDY_ARGS+=(--comparator "${COMPARATOR}"); fi
if [ -n "${EXPOSURE_LEVEL}" ]; then _STUDY_ARGS+=(--level "${EXPOSURE_LEVEL}"); fi
_STUDY_ENV="$(python "${STUDY_SCRIPT}" "${_STUDY_ARGS[@]}")"
eval "${_STUDY_ENV}"

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================

# symmetric or target_initiators (see the header)
DESIGN="${DESIGN:-symmetric}"
case "${DESIGN}" in
    symmetric|target_initiators) ;;
    *) echo "ERROR: DESIGN must be symmetric or target_initiators, got '${DESIGN}'" >&2; exit 1 ;;
esac

# Inputs: set them here or on the command line (PATIENT_SEQUENCE_PATH=... bash scripts/run_causal_study.sh).
#   PATIENT_SEQUENCE_PATH : the patient sequences (a parquet file or a folder of parquet files)
#   VOCAB_PATH            : the OMOP vocabulary, a folder that contains concept_ancestor/
#   MODEL_PATH            : the CEHR-GPT model
#   TOKENIZER_PATH        : its tokenizer; CEHR-GPT saves it next to the model, so that is the default
PATIENT_SEQUENCE_PATH="${PATIENT_SEQUENCE_PATH:-/data/patient_sequence}"
VOCAB_PATH="${VOCAB_PATH:-/data/omop_vocab}"
MODEL_PATH="${MODEL_PATH:-/models/cehrgpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
# Everything is written here; the default is a folder in the current directory.
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/output/${STUDY_LABEL}_${TARGET_LABEL}_vs_${COMPARATOR_LABEL}_${DESIGN}}"

# Settings like the ones below can be given on the command line too, e.g.
#   NUM_TRAJECTORIES=10 BATCH_SIZE=16 bash scripts/run_causal_study.sh

# Number of stochastic trajectories per patient per arm
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-5}"

# Maximum patients to draw (random sample) from the target patients and, in the symmetric
# design, from the comparator patients before combining.  Set to "" to use all patients
# (MAX_TARGET_PATIENTS="" bash scripts/run_causal_study.sh).
MAX_TARGET_PATIENTS="${MAX_TARGET_PATIENTS-500}"
MAX_COMPARATOR_PATIENTS="${MAX_COMPARATOR_PATIENTS-500}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GENERATION_INPUT_LENGTH="${GENERATION_INPUT_LENGTH:-2048}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-1024}"
FOLLOW_UP_DAYS="${FOLLOW_UP_DAYS:-365}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Number of parallel workers for Step 1 sequence extraction
EXTRACTION_NUM_WORKERS="${EXTRACTION_NUM_WORKERS:-10}"

# Minimum number of tokens required in the pre-drug context (Step 1)
MIN_CONTEXT_LENGTH="${MIN_CONTEXT_LENGTH:-4}"

# Random seed for frequency-based drug sampling in Step 2
RANDOM_SEED="${RANDOM_SEED:-42}"

# Set to "true" to re-run Step 1 even when output files already exist
OVERWRITE="${OVERWRITE:-false}"

# ERA_GAP_DAYS (two drug events within this many days belong to the same era; the era end is
# stored in drug_info/ and bounds the follow-up in Step 4) and MIN_PRIOR_OBSERVATION_DAYS (washout:
# days of observed history before the index drug) come from the study definition above.

# The cohort and time-at-risk rules come from the study definition:
#   * monotherapy: exactly one drug of the study may start in the first MONOTHERAPY_WINDOW_DAYS
#     days from the index date (empty: the older same-visit checks);
#   * the indication has to be recorded in the POPULATION_WINDOW_DAYS days up to the index date;
#   * patients with the outcome before the index date are left out of that outcome;
#   * time at risk starts the day after the index date and needs MIN_DAYS_AT_RISK days; the
#     on-treatment design ends it when the continuous exposure to the patient's own exposure ends
#     (a gap above ERA_GAP_DAYS), the intent-to-treat design with the observation of the patient.
#     Starting another drug does not end it (CENSOR_AT_SWITCH=true does).
TIME_AT_RISK="${TIME_AT_RISK:-on_treatment}"      # or intent_to_treat
CENSOR_AT_SWITCH="${CENSOR_AT_SWITCH:-false}"

# Set to "true" to suppress the other arm's drugs and the exclusion drugs during generation.
# Patients may add drugs in practice, so the default is false.
SUPPRESS_CONCEPTS="${SUPPRESS_CONCEPTS:-false}"

# -----------------------------------------------------------------------------
# GPU configuration
# -----------------------------------------------------------------------------
# GPU device IDs for the target arm (GPU_TARGET) and the comparator arm (GPU_COMPARATOR).
# Set SINGLE_GPU=true to run both arms sequentially on GPU_TARGET (e.g. when only
# one GPU is available or for debugging).
GPU_TARGET="${GPU_TARGET:-0}"
GPU_COMPARATOR="${GPU_COMPARATOR:-1}"
SINGLE_GPU="${SINGLE_GPU:-false}"

# Several GPUs for the generation step: "all" or a list such as "0,1,2,3".  The GPUs are divided
# into one set per arm (in proportion to the batches each arm has left) and one worker per GPU
# generates a share of the batches of its arm, so the arms run at the same time.  With a single
# GPU the arms run one after the other.  An interrupted run continues where it stopped (rerun the
# script with SKIP_STEP1=true SKIP_STEP2=true).  Takes precedence over GPU_TARGET / GPU_COMPARATOR /
# SINGLE_GPU.  Empty = the settings above.
GPU_IDS="${GPU_IDS:-}"

# -----------------------------------------------------------------------------
# Skip flags — set to "true" to skip steps already completed
# -----------------------------------------------------------------------------
SKIP_STEP1="${SKIP_STEP1:-false}"
SKIP_STEP2="${SKIP_STEP2:-false}"
SKIP_STEP3="${SKIP_STEP3:-false}"
SKIP_STEP4="${SKIP_STEP4:-false}"

# =============================================================================
# ARM NAMES — used as sub-directory names under TRAJ_DIR and as arm labels
# =============================================================================
ARM_A="${TARGET_LABEL}"
ARM_B="${COMPARATOR_LABEL}"

# Outcomes: the outcomes and negative controls of the study (outcomes.csv).  Step 0 resolves their
# concepts (descendants, exclusions) with VOCAB_PATH and keeps the inpatient / ER visit restriction of
# the outcomes that need one.

# Further outcomes by concept_id (descendants included), optional
OUTCOME_CONCEPT_IDS="${OUTCOME_CONCEPT_IDS-}"
# ... of which these must occur inside an inpatient or ER visit (9201/9203/262)
INPATIENT_OUTCOME_CONCEPT_IDS="${INPATIENT_OUTCOME_CONCEPT_IDS-}"

# Optional control outcomes (leave empty for none).  A negative control is an outcome
# neither drug is expected to affect (expected HR = 1); a positive control has a known
# effect, given as concept_id or concept_id:expected_hr (e.g. "4329847:1.5").  They are
# extracted together with the primary outcomes and analysed in the same way; the HR
# summary labels them and control_outcome_summary.csv reports the systematic bias.
NEGATIVE_CONTROL_OUTCOME_IDS="${NEGATIVE_CONTROL_OUTCOME_IDS-}"
POSITIVE_CONTROL_OUTCOME_IDS="${POSITIVE_CONTROL_OUTCOME_IDS-}"

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
PREFERENCE_SCORE_TRIM="${PREFERENCE_SCORE_TRIM-0.05}"

# The swap puts a drug of the other exposure into each history, so the model has to know that drug.
# By default the run stops when the tokenizer does not know one of the ingredients.  Set to "true" to
# leave those ingredients out (the others are sampled by their relative frequency; the share of the
# real prescriptions that is left out is printed).  Patients whose own drug the tokenizer does not
# know are always left out.
DROP_UNKNOWN_COMPARATORS="${DROP_UNKNOWN_COMPARATORS:-false}"

# =============================================================================
# DERIVED PATHS
# =============================================================================

# Extraction output with the target class as the source (Step 1a); the symmetric design adds one
# with the comparator class as the source (Step 1b).
TARGET_CONTEXT_DIR="${OUTPUT_ROOT}/target_contexts"
COMPARATOR_CONTEXT_DIR="${OUTPUT_ROOT}/comparator_contexts"

SWAP_DIR="${OUTPUT_ROOT}/swap_contexts"
TARGET_OVERLAP_SCORES="${SWAP_DIR}/target_pts_overlap_scores.parquet"
COMPARATOR_OVERLAP_SCORES="${SWAP_DIR}/comparator_pts_overlap_scores.parquet"

# What Step 2 writes for the target patients (2a) and, symmetric, for the comparator patients (2b)
TARGET_PTS_TARGET_CTX="${SWAP_DIR}/target_pts_target_ctx.parquet"
TARGET_PTS_COMPARATOR_CTX="${SWAP_DIR}/target_pts_comparator_ctx.parquet"
COMPARATOR_PTS_COMPARATOR_CTX="${SWAP_DIR}/comparator_pts_comparator_ctx.parquet"
COMPARATOR_PTS_TARGET_CTX="${SWAP_DIR}/comparator_pts_target_ctx.parquet"

# The contexts of the two arms, and what Step 4 reads about the patients
if [ "${DESIGN}" = "symmetric" ]; then
    ARM_A_CTX="${SWAP_DIR}/${ARM_A}_arm_ctx.parquet"
    ARM_B_CTX="${SWAP_DIR}/${ARM_B}_arm_ctx.parquet"
    DRUG_INFO="${SWAP_DIR}/drug_info"
    PRIOR_OUTCOME_PATHS=("${TARGET_CONTEXT_DIR}/observed_outcomes" "${COMPARATOR_CONTEXT_DIR}/observed_outcomes")
    OVERLAP_SCORE_PATHS=("${TARGET_OVERLAP_SCORES}" "${COMPARATOR_OVERLAP_SCORES}")
else
    ARM_A_CTX="${TARGET_PTS_TARGET_CTX}"
    ARM_B_CTX="${TARGET_PTS_COMPARATOR_CTX}"
    DRUG_INFO="${TARGET_CONTEXT_DIR}/drug_info"
    PRIOR_OUTCOME_PATHS=("${TARGET_CONTEXT_DIR}/observed_outcomes")
    OVERLAP_SCORE_PATHS=("${TARGET_OVERLAP_SCORES}")
fi

TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"
OUTCOME_DEFS_DIR="${OUTPUT_ROOT}/outcome_definitions"
OUTCOME_DEFINITIONS="${OUTCOME_DEFS_DIR}/outcome_definitions.json"
OUTCOME_SCRIPT="src/cehrgpt/analysis/counterfactual/outcome_definitions.py"

EXTRACT_SCRIPT="src/cehrgpt/analysis/counterfactual/extract_drug_initiation_sequences.py"
SWAP_SCRIPT="src/cehrgpt/analysis/counterfactual/create_drug_swap_context.py"
GENERATE_SCRIPT="src/cehrgpt/analysis/counterfactual/generate_counterfactual_sequences.py"
HR_SCRIPT="src/cehrgpt/analysis/counterfactual/hazard_ratio_estimation.py"
PLOT_SCRIPT="src/cehrgpt/analysis/counterfactual/overlap_plot.py"

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

# (Arrays that can be empty are expanded as ${a[@]+"${a[@]}"}, which also works in bash 3.2 with set -u.)
# Check the inputs and the output folder before anything runs, so a wrong path is reported at once.
_fail() {
    echo "ERROR: $1" >&2
    exit 1
}
_require_path() {   # name, path, what it should be
    [ -e "$2" ] || _fail "$1 does not exist: $2 ($3). Set $1=... on the command line."
}
mkdir -p "${OUTPUT_ROOT}" 2>/dev/null || _fail "cannot create OUTPUT_ROOT=${OUTPUT_ROOT}. Set OUTPUT_ROOT=... to a folder you can write to."
_require_path VOCAB_PATH "${VOCAB_PATH}/concept_ancestor" "the OMOP vocabulary, a folder with concept_ancestor/"
if [ "${SKIP_STEP1}" != "true" ]; then
    _require_path PATIENT_SEQUENCE_PATH "${PATIENT_SEQUENCE_PATH}" "the patient sequences"
fi
if [ "${SKIP_STEP1}" != "true" ] || [ "${SKIP_STEP2}" != "true" ] || [ "${SKIP_STEP3}" != "true" ]; then
    _require_path MODEL_PATH "${MODEL_PATH}" "the CEHR-GPT model"
    _require_path TOKENIZER_PATH "${TOKENIZER_PATH}" "the tokenizer, saved next to the model by default"
fi

echo "Study: ${STUDY_NAME}   Design: ${DESIGN}   ${TARGET_NAME} vs ${COMPARATOR_NAME}   output: ${OUTPUT_ROOT}"
echo ""

# =============================================================================
# STEP 0 — The outcomes of the study
#
#   Writes outcome_definitions.json (the concepts of every outcome, with descendants and
#   exclusions resolved, and its inpatient / ER restriction) and the raw concept sets.
# =============================================================================
echo "============================================================"
echo "STEP 0: outcome definitions of ${STUDY_NAME}  [$(date '+%H:%M:%S')]"
echo "============================================================"
mkdir -p "${OUTCOME_DEFS_DIR}"
python "${OUTCOME_SCRIPT}" \
    --study       "${STUDY_DIR}" \
    --vocab_path  "${VOCAB_PATH}" \
    --output_dir  "${OUTCOME_DEFS_DIR}"
echo ""

# =============================================================================
# STEP 1 — Extract drug initiation sequences
#
#   1a: target as source, comparator as comparator.
#       Produces the target patients' treated_context and drug_info.  The comparator patients
#       in drug_info give the empirical frequency of each comparator ingredient (for the
#       frequency-weighted swap in Step 2) and the observed comparator cohort.
#
#   1b (symmetric only): comparator as source, target as comparator.
#       Produces the comparator patients' treated_context and drug_info.  The target patients
#       in drug_info give the empirical frequency of each target ingredient.
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
    if [ -n "${POPULATION_CONCEPT_IDS}" ]; then
        echo "  Population filter: ${POPULATION_CONCEPT_IDS} (within ${POPULATION_WINDOW_DAYS:-any} days)"
    fi
    if [ -n "${EXCLUSION_CONCEPT_IDS}" ]; then
        echo "  Exclusion (new-user filter): enabled ($(echo "${EXCLUSION_CONCEPT_IDS}" | tr ',' '\n' | wc -l | tr -d ' ') ingredients)"
    fi
    echo "  Min context length: ${MIN_CONTEXT_LENGTH}"
    echo "  Overwrite: ${OVERWRITE}"
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

    if [ "${DESIGN}" = "symmetric" ]; then
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
fi

# =============================================================================
# STEP 2 — Create the arm contexts by running create_drug_swap_context.py
#
#   2a — target as source:
#     • target_pts_target_ctx.parquet         : target patients, their actual target drug
#     • target_pts_comparator_ctx.parquet     : target patients, comparator frequency-swapped
#
#   2b (symmetric only) — comparator as source:
#     • comparator_pts_comparator_ctx.parquet : comparator patients, their actual drug
#     • comparator_pts_target_ctx.parquet     : comparator patients, target frequency-swapped
#
#   target_initiators:
#     Arm A (target)     = target_pts_target_ctx
#     Arm B (comparator) = target_pts_comparator_ctx
#   symmetric:
#     Arm A (target)     = concat(target_pts_target_ctx,     comparator_pts_target_ctx)
#     Arm B (comparator) = concat(target_pts_comparator_ctx, comparator_pts_comparator_ctx)
#     drug_info          = concat(target drug_info, comparator drug_info)
#     Both arms cover the same combined patient pool.
#
#   A patient that fails the overlap check (PREFERENCE_SCORE_TRIM) is dropped from both arms.
# =============================================================================
if [ "${SKIP_STEP2}" = "true" ]; then
    echo "Skipping Step 2 (SKIP_STEP2=true)"
else
    echo "============================================================"
    echo "STEP 2: Create the arm contexts (${DESIGN})  [$(date '+%H:%M:%S')]"
    echo "  ${TARGET_NAME} patient contexts: ${TARGET_CONTEXT_DIR}/treated_context"
    if [ "${DESIGN}" = "symmetric" ]; then
        echo "  ${COMPARATOR_NAME} patient contexts: ${COMPARATOR_CONTEXT_DIR}/treated_context"
        echo "  Combined drug info: ${DRUG_INFO}"
    fi
    echo "  Random seed: ${RANDOM_SEED}"
    echo "============================================================"

    mkdir -p "${SWAP_DIR}"
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
    _DROP_UNKNOWN_ARG=()
    if [ "${DROP_UNKNOWN_COMPARATORS}" = "true" ]; then
        _DROP_UNKNOWN_ARG=(--drop_unknown_comparators)
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
        --output_path            "${TARGET_PTS_COMPARATOR_CTX}" \
        --also_write_treated_path "${TARGET_PTS_TARGET_CTX}" \
        --seed                   "${RANDOM_SEED}" \
        ${_OVERLAP_TARGET_ARGS[@]+"${_OVERLAP_TARGET_ARGS[@]}"} \
        ${_DROP_UNKNOWN_ARG[@]+"${_DROP_UNKNOWN_ARG[@]}"} \
        ${_MAX_TARGET_ARG[@]+"${_MAX_TARGET_ARG[@]}"}

    if [ "${DESIGN}" = "symmetric" ]; then
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
            --output_path            "${COMPARATOR_PTS_TARGET_CTX}" \
            --also_write_treated_path "${COMPARATOR_PTS_COMPARATOR_CTX}" \
            --seed                   "${RANDOM_SEED}" \
            ${_OVERLAP_COMPARATOR_ARGS[@]+"${_OVERLAP_COMPARATOR_ARGS[@]}"} \
            ${_DROP_UNKNOWN_ARG[@]+"${_DROP_UNKNOWN_ARG[@]}"} \
            ${_MAX_COMPARATOR_ARG[@]+"${_MAX_COMPARATOR_ARG[@]}"}

        # --- Concat arm contexts and drug_info ---
        echo "  Combining arm contexts and drug_info …"
        TARGET_PTS_TARGET_CTX="${TARGET_PTS_TARGET_CTX}" \
        TARGET_PTS_COMPARATOR_CTX="${TARGET_PTS_COMPARATOR_CTX}" \
        COMPARATOR_PTS_COMPARATOR_CTX="${COMPARATOR_PTS_COMPARATOR_CTX}" \
        COMPARATOR_PTS_TARGET_CTX="${COMPARATOR_PTS_TARGET_CTX}" \
        ARM_A_CTX="${ARM_A_CTX}" ARM_B_CTX="${ARM_B_CTX}" \
        COMBINED_DRUG_INFO="${DRUG_INFO}" \
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
    pl.read_parquet(os.environ["TARGET_PTS_TARGET_CTX"]),
    pl.read_parquet(os.environ["COMPARATOR_PTS_TARGET_CTX"]),
])
arm_a.write_parquet(os.environ["ARM_A_CTX"])

# Arm B (comparator): target patients with comparator swap + comparator patients with comparator
arm_b = pl.concat([
    pl.read_parquet(os.environ["TARGET_PTS_COMPARATOR_CTX"]),
    pl.read_parquet(os.environ["COMPARATOR_PTS_COMPARATOR_CTX"]),
])
arm_b.write_parquet(os.environ["ARM_B_CTX"])

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
    fi
    _step_end

    echo ""
    echo "Arm sizes:"
    ARM_A_CTX="${ARM_A_CTX}" ARM_B_CTX="${ARM_B_CTX}" \
    ARM_A="${ARM_A}" ARM_B="${ARM_B}" \
    python - <<'PYEOF'
import polars as pl, os
for label, key in [("Target arm", "ARM_A"), ("Comparator arm", "ARM_B")]:
    f = os.environ[f"{key}_CTX"]
    arm = os.environ[key]
    n = len(pl.read_parquet(f))
    print(f"  {label} ({arm}): {n:,} patients  → {f}")
PYEOF

    # The overlap of the preference scores of the two groups of patients, before any GPU time is spent
    if [ -n "${PREFERENCE_SCORE_TRIM}" ]; then
        mkdir -p "${RESULTS_DIR}"
        _PLOT_ARGS=(--target_scores "${TARGET_OVERLAP_SCORES}")
        if [ "${DESIGN}" = "symmetric" ]; then
            _PLOT_ARGS+=(--comparator_scores "${COMPARATOR_OVERLAP_SCORES}")
        fi
        python "${PLOT_SCRIPT}" "${_PLOT_ARGS[@]}" \
            --trim "${PREFERENCE_SCORE_TRIM}" \
            --target_name "${TARGET_NAME}" --comparator_name "${COMPARATOR_NAME}" \
            --output "${RESULTS_DIR}/preference_score_overlap.png"
    fi

    echo "Step 2 complete."
    echo ""
fi

# =============================================================================
# STEP 3 — Generate trajectories for both arms
#
#   Default: the target arm on GPU_TARGET and the comparator arm on GPU_COMPARATOR, at the
#   same time.  SINGLE_GPU=true runs them one after the other on GPU_TARGET; GPU_IDS divides
#   the GPUs between the two arms (one GPU: the arms one after the other).
# =============================================================================
if [ "${SKIP_STEP3}" = "true" ]; then
    echo "Skipping Step 3 (SKIP_STEP3=true)"
else
    echo "============================================================"
    echo "STEP 3: Generate trajectories  (${NUM_TRAJECTORIES} per patient per arm)  [$(date '+%H:%M:%S')]"
    if [ -n "${GPU_IDS}" ]; then
        echo "  Mode              : GPUs divided between the arms, one worker per GPU (GPU_IDS=${GPU_IDS})"
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

    # Concepts to suppress per arm:
    #   arm_a (target) suppresses the comparator drugs + all other drugs of the study
    #   arm_b (comparator) suppresses the target drugs + all other drugs of the study
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
            ${_MULTI_SUPPRESS_ARGS[@]+"${_MULTI_SUPPRESS_ARGS[@]}"} \
            --gpu_ids "${GPU_IDS}" \
            > "${OUTPUT_ROOT}/generate.log" 2>&1
    elif [ "${SINGLE_GPU}" = "true" ]; then
        echo "Running ${ARM_A} arm …"
        CUDA_VISIBLE_DEVICES="${GPU_TARGET}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_A}:${ARM_A_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            ${_SUPPRESS_ARGS_A[@]+"${_SUPPRESS_ARGS_A[@]}"} \
            > "${OUTPUT_ROOT}/generate_${ARM_A}.log" 2>&1
        echo "Running ${ARM_B} arm …"
        CUDA_VISIBLE_DEVICES="${GPU_TARGET}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_B}:${ARM_B_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            ${_SUPPRESS_ARGS_B[@]+"${_SUPPRESS_ARGS_B[@]}"} \
            > "${OUTPUT_ROOT}/generate_${ARM_B}.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="${GPU_TARGET}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_A}:${ARM_A_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            ${_SUPPRESS_ARGS_A[@]+"${_SUPPRESS_ARGS_A[@]}"} \
            > "${OUTPUT_ROOT}/generate_${ARM_A}.log" 2>&1 &
        PID_A=$!

        CUDA_VISIBLE_DEVICES="${GPU_COMPARATOR}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_B}:${ARM_B_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            ${_SUPPRESS_ARGS_B[@]+"${_SUPPRESS_ARGS_B[@]}"} \
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
#
#   (a) generated trajectories: arm A vs arm B, written to hazard_ratio_summary.csv;
#   (b) the observed cohorts: the real target and comparator patients of Step 1a, written to
#       observed_hazard_ratio_summary.csv, as a reference for (a).
#   Both leave out the patients that had an outcome before the index date (for that outcome)
#   and the patients dropped by the overlap check of Step 2.
# =============================================================================
if [ "${SKIP_STEP4}" = "true" ]; then
    echo "Skipping Step 4 (SKIP_STEP4=true)"
else
    echo "============================================================"
    echo "STEP 4: Hazard ratio estimation  (${FOLLOW_UP_DAYS}-day follow-up)  [$(date '+%H:%M:%S')]"
    echo "  Arm A (treated)    : ${ARM_A}"
    echo "  Arm B (comparator) : ${ARM_B}"
    echo "  Drug info          : ${DRUG_INFO}"
    echo "  Outcomes: ${OUTCOME_DEFINITIONS}"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    _step_start
    _HR_EXTRA_ARGS=()
    _HR_EXTRA_ARGS+=(--prior_outcome_path "${PRIOR_OUTCOME_PATHS[@]}")
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
        _HR_EXTRA_ARGS+=(--overlap_scores_path "${OVERLAP_SCORE_PATHS[@]}")
    fi

    # (a) generated trajectories
    python "${HR_SCRIPT}" \
        --trajectories_dir        "${TRAJ_DIR}" \
        --drug_info_path          "${DRUG_INFO}" \
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

    # (b) observed cohorts: Step 1a extracted both the real target and the real comparator patients
    python "${HR_SCRIPT}" \
        --drug_info_path          "${TARGET_CONTEXT_DIR}/drug_info" \
        --observed_outcomes_path  "${TARGET_CONTEXT_DIR}/observed_outcomes" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --outcome_definitions_path "${OUTCOME_DEFINITIONS}" \
        --follow_up_days          "${FOLLOW_UP_DAYS}" \
        --arm_a                   "${ARM_A}" \
        --arm_b                   "${ARM_B}" \
        --output_dir              "${RESULTS_DIR}" \
        --vocab_path              "${VOCAB_PATH}" \
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
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv           (generated ${TARGET_NAME} vs generated ${COMPARATOR_NAME})"
echo "  ${RESULTS_DIR}/observed_hazard_ratio_summary.csv  (observed ${TARGET_NAME} vs observed ${COMPARATOR_NAME})"
echo "  ${RESULTS_DIR}/control_outcome_summary.csv        (negative / positive controls, when given)"
echo "  ${RESULTS_DIR}/preference_score_overlap.png       (overlap of the two groups of patients; _summary.csv beside it)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
# reference results of the study for this pair, when it has any
python "${STUDY_SCRIPT}" "${_STUDY_ARGS[@]}" --reference
echo "============================================================"
