#!/usr/bin/env bash
# =============================================================================
# LEGEND-HTN Replication: ACEi vs Thiazide (within-patient counterfactual)
# =============================================================================
#
# Study design
# ------------
#   Population  : Patients who initiated an ACE inhibitor (new users).
#   Arm A (ACEi): History up to and including the ACEi initiation visit.
#                 Model generates the future conditional on ACEi.
#   Arm B (Thiazide counterfactual):
#                 Same history, but the ACEi concept is swapped to a
#                 thiazide concept sampled proportional to the empirical
#                 frequency of thiazide ingredients in the data.
#                 Model generates the counterfactual future as if the
#                 patient had received a thiazide instead.
#   Each patient produces N trajectories in each arm.
#   HR is computed by comparing ACEi vs Thiazide generated trajectories.
#
# Why we also extract thiazide patients in Step 1
# -----------------------------------------------
#   We need the empirical frequency of each thiazide ingredient (HCTZ,
#   chlorthalidone, indapamide, metolazone) to weight the drug swap in
#   Step 2.  We obtain this by including thiazide patients in the
#   extraction so they appear in drug_info.parquet.  Thiazide patients
#   are NOT used in any other step.
#
# Outcomes (OMOP concept_ids)
# ----------------------------
#    312327  Acute myocardial infarction
#    316139  Heart failure
#   4310996  Ischemic stroke
#    376713  Cerebral hemorrhage
#   3655355  Erectile dysfunction
#    437833  Hypokalemia
#    434610  Hyperkalemia
#
# Published LEGEND-HTN reference HRs (ACEi vs Thiazide):
#   AMI           : 0.99  (95% CI 0.87–1.13)
#   Heart failure : 1.18  (95% CI 1.07–1.30)  [ACEi worse]
#   Stroke        : 1.09  (95% CI 0.99–1.20)
#
# Usage
# -----
#   Edit the CONFIGURATION section, then:
#     bash run_legend_htn_acei_vs_thiazide.sh
#
#   To skip steps already completed (e.g. after a failure in step 3):
#     SKIP_STEP1=true SKIP_STEP2=true bash run_legend_htn_acei_vs_thiazide.sh
#
#   To run both arms on a single GPU sequentially:
#     SINGLE_GPU=true bash run_legend_htn_acei_vs_thiazide.sh
#
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================

PATIENT_SEQUENCE_PATH="/data/patient_sequence"
VOCAB_PATH="/data/omop_vocab"
MODEL_PATH="/models/cehrgpt"
TOKENIZER_PATH="/models/tokenizer"
OUTPUT_ROOT="/data/legend_htn_acei_vs_thiazide"

# Number of stochastic trajectories per patient per arm
NUM_TRAJECTORIES=5

# Maximum ACEi patients to use (random sample); set to "" to use all patients
MAX_ACEI_PATIENTS=500

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

# Drug era persistence window in days.
# Two drug events within this many days belong to the same era.
# The era end time is stored in drug_info/ and used by Step 4 as an additional
# censoring bound (treatment discontinuation or switch to another antihypertensive).
ERA_GAP_DAYS=30

# Set to "true" to suppress competing drug concepts during generation so the
# model cannot generate opposite-arm or excluded antihypertensive tokens.
# Suppressed per arm: arm_a suppresses thiazide+exclusion; arm_b suppresses acei+exclusion.
SUPPRESS_CONCEPTS=true

# Optional: comma-separated OMOP concept_ids restricting the eligible population.
# Each id is expanded to descendants via concept_ancestor; only patients whose
# pre-drug history contains at least one matching concept are kept.
# Example: "320128" filters to patients with an essential hypertension diagnosis.
# Leave empty ("") to apply no population filter.
POPULATION_CONCEPT_IDS=""

# Exclusion concept IDs — "new users" eligibility criterion.
# Patients who have ANY of these drugs in their pre-index history are excluded.
# This enforces a treatment-naive design: only patients who have never received
# another first-line or second-line antihypertensive before their index drug are
# eligible.  Leave empty ("") to disable exclusion filtering.
#
# ARBs
#   40235485  Azilsartan     1351557  Candesartan   1346686  Eprosartan
#    1347384  Irbesartan     1367500  Losartan     40226742  Olmesartan
#    1317640  Telmisartan    1308842  Valsartan
# Dihydropyridine CCBs (dCCBs)
#    1332418  Amlodipine     1353776  Felodipine    1326012  Isradipine
#    1318137  Nicardipine    1318853  Nifedipine    1319880  Nisoldipine
# Non-dihydropyridine CCBs (ndCCBs)
#    1328165  Diltiazem      1307863  Verapamil
# Second-line agents
#    1319998  Acebutolol     1317967  Aliskiren      991382  Amiloride
#    1314002  Atenolol       1322081  Betaxolol     1338005  Bisoprolol
#     932745  Bumetanide     1346823  Carvedilol    1398937  Clonidine
#    1363053  Doxazosin      1309799  Eplerenone     956874  Furosemide
#    1344965  Guanfacine     1373928  Hydralazine   1386957  Labetalol
#    1305447  Methyldopa     1307046  Metoprolol    1309068  Minoxidil
#    1313200  Nadolol        1314577  Nebivolol     1327978  Penbutolol
#    1345858  Pindolol       1350489  Prazosin      1353766  Propranolol
#     970250  Spironolactone 1341238  Terazosin      942350  Torsemide
#     904542  Triamterene
EXCLUSION_CONCEPT_IDS="40235485,1351557,1346686,1347384,1367500,40226742,1317640,1308842,1332418,1353776,1326012,1318137,1318853,1319880,1328165,1307863,1319998,1317967,991382,1314002,1322081,1338005,932745,1346823,1398937,1363053,1309799,956874,1344965,1373928,1386957,1305447,1307046,1309068,1313200,1314577,1327978,1345858,1350489,1353766,970250,1341238,942350,904542"

# -----------------------------------------------------------------------------
# GPU configuration
# -----------------------------------------------------------------------------
# GPU device IDs for the ACEi arm (GPU_ACEI) and Thiazide arm (GPU_THIAZIDE).
# Set SINGLE_GPU=true to run both arms sequentially on GPU_ACEI (e.g. when only
# one GPU is available or for debugging).
GPU_ACEI=0
GPU_THIAZIDE=1
SINGLE_GPU=false

# -----------------------------------------------------------------------------
# Skip flags — set to "true" to skip steps already completed
# -----------------------------------------------------------------------------
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false

# =============================================================================
# ARM NAMES — used as sub-directory names under TRAJ_DIR and as arm labels
# =============================================================================
ARM_A="acei"
ARM_B="thiazide"

# =============================================================================
# DRUG CONCEPT IDs
# =============================================================================

ACEI_CONCEPT_IDS="1308216,1334456,1341927,1335471,1331235,1340128,1363749,1310756,1373225,1342439"
# 1308216  Lisinopril
# 1334456  Ramipril
# 1341927  Enalapril
# 1335471  Benazepril
# 1331235  Quinapril
# 1340128  Captopril
# 1363749  Fosinopril
# 1310756  Moexipril
# 1373225  Perindopril
# 1342439  Trandolapril

THIAZIDE_CONCEPT_IDS="1395058,974166,978555,907013"
# 1395058  Chlorthalidone
# 974166   Hydrochlorothiazide
# 978555   Indapamide
# 907013   Metolazone

# Outcome concept IDs
OUTCOME_CONCEPT_IDS="312327,316139,4310996,376713,3655355,437833,434610"
# 312327   Acute myocardial infarction
# 316139   Heart failure
# 4310996  Ischemic stroke
# 376713   Cerebral hemorrhage
# 3655355  Erectile dysfunction
# 437833   Hypokalemia
# 434610   Hyperkalemia

# =============================================================================
# DERIVED PATHS
# =============================================================================

CONTEXT_DIR="${OUTPUT_ROOT}/contexts"
SWAP_DIR="${OUTPUT_ROOT}/swap_contexts"
TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"

TREATED_CTX="${CONTEXT_DIR}/treated_context"
DRUG_INFO="${CONTEXT_DIR}/drug_info"

# Arm A (ACEi) — actual context; Arm B (Thiazide) — counterfactual context
ARM_A_CTX="${SWAP_DIR}/${ARM_A}_arm_ctx.parquet"
ARM_B_CTX="${SWAP_DIR}/${ARM_B}_arm_ctx.parquet"

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
# STEP 1 — Extract drug initiation sequences
#
#   Include BOTH ACEi AND thiazide patients so that drug_info.parquet
#   contains real thiazide prescriptions.  These are used in Step 2 to
#   compute the empirical frequency of each thiazide ingredient for
#   frequency-weighted sampling.  Thiazide patients are not used elsewhere.
# =============================================================================
if [ "${SKIP_STEP1}" = "true" ]; then
    echo "Skipping Step 1 (SKIP_STEP1=true)"
else
    echo "============================================================"
    echo "STEP 1: Extract drug initiation sequences  [$(date '+%H:%M:%S')]"
    echo "  ACEi patients   : ${ACEI_CONCEPT_IDS}"
    echo "  Thiazide (freq) : ${THIAZIDE_CONCEPT_IDS}"
    if [ -n "${POPULATION_CONCEPT_IDS}" ]; then
        echo "  Population filter: ${POPULATION_CONCEPT_IDS}"
    fi
    if [ -n "${EXCLUSION_CONCEPT_IDS}" ]; then
        echo "  Exclusion (new-user filter): enabled (${#EXCLUSION_CONCEPT_IDS} chars)"
    fi
    echo "  Min context length: ${MIN_CONTEXT_LENGTH}"
    echo "  Overwrite: ${OVERWRITE}"
    echo "============================================================"

    mkdir -p "${CONTEXT_DIR}"

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

    _step_start
    python "${EXTRACT_SCRIPT}" \
        --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${ACEI_CONCEPT_IDS}" \
        --comparator_concept_ids  "${THIAZIDE_CONCEPT_IDS}" \
        --output_dir              "${CONTEXT_DIR}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --era_gap_days            "${ERA_GAP_DAYS}" \
        --min_context_length      "${MIN_CONTEXT_LENGTH}" \
        --num_workers             "${EXTRACTION_NUM_WORKERS}" \
        "${_EXTRACT_EXTRA_ARGS[@]}"
    _step_end

    echo "Step 1 complete."
    echo ""
fi

# =============================================================================
# STEP 2 — Create the two arm contexts for ACEi patients
#
#   For each ACEi patient we produce:
#     Arm A (${ARM_A}_arm_ctx) : treated_context as-is (ACEi in context)
#     Arm B (${ARM_B}_arm_ctx) : treated_context with ACEi concept swapped
#                                to a thiazide concept sampled proportional
#                                to the empirical thiazide frequency derived
#                                from real thiazide patients in drug_info.parquet
# =============================================================================
if [ "${SKIP_STEP2}" = "true" ]; then
    echo "Skipping Step 2 (SKIP_STEP2=true)"
else
    echo "============================================================"
    echo "STEP 2: Create ACEi (actual) and Thiazide (counterfactual) contexts  [$(date '+%H:%M:%S')]"
    echo "  Source      : ACEi patients"
    echo "  Comparator  : thiazide (frequency-sampled from real thiazide Rx)"
    echo "  Random seed : ${RANDOM_SEED}"
    echo "============================================================"

    mkdir -p "${SWAP_DIR}"

    _SWAP_EXTRA_ARGS=(--seed "${RANDOM_SEED}")
    if [ -n "${MAX_ACEI_PATIENTS:-}" ]; then
        _SWAP_EXTRA_ARGS+=(--max_source_patients "${MAX_ACEI_PATIENTS}")
    fi

    _step_start
    python "${SWAP_SCRIPT}" \
        --treated_context_path    "${TREATED_CTX}" \
        --drug_info_path          "${DRUG_INFO}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${ACEI_CONCEPT_IDS}" \
        --comparator_concept_ids  "${THIAZIDE_CONCEPT_IDS}" \
        --output_path             "${ARM_B_CTX}" \
        --also_write_treated_path "${ARM_A_CTX}" \
        "${_SWAP_EXTRA_ARGS[@]}"
    _step_end

    echo ""
    echo "Arm sizes:"
    ARM_A_CTX="${ARM_A_CTX}" ARM_B_CTX="${ARM_B_CTX}" \
    ARM_A="${ARM_A}" ARM_B="${ARM_B}" \
    python - <<'PYEOF'
import polars as pl, os
for label, key in [("ACEi arm (actual)", "ARM_A"), ("Thiazide arm (counterfactual)", "ARM_B")]:
    f = os.environ[f"{key}_CTX"]
    arm = os.environ[key]
    n = len(pl.read_parquet(f))
    print(f"  {label} ({arm}): {n:,} patients")
PYEOF

    echo "Step 2 complete."
    echo ""
fi

# =============================================================================
# STEP 3 — Generate trajectories for both arms
#
#   When SINGLE_GPU=false (default):
#     GPU_ACEI:    ${ARM_A}/  — N futures per patient conditioned on ACEi
#     GPU_THIAZIDE: ${ARM_B}/ — N counterfactual futures conditioned on thiazide
#     Both processes run simultaneously; the script waits for both to finish.
#
#   When SINGLE_GPU=true:
#     Both arms run sequentially on GPU_ACEI (useful when only one GPU is
#     available or for debugging).
# =============================================================================
if [ "${SKIP_STEP3}" = "true" ]; then
    echo "Skipping Step 3 (SKIP_STEP3=true)"
else
    echo "============================================================"
    echo "STEP 3: Generate trajectories  (${NUM_TRAJECTORIES} per patient per arm)  [$(date '+%H:%M:%S')]"
    if [ "${SINGLE_GPU}" = "true" ]; then
        echo "  Mode   : sequential (SINGLE_GPU=true, GPU ${GPU_ACEI})"
    else
        echo "  GPU ${GPU_ACEI} → ${ARM_A}/     (conditioned on ACEi initiation)"
        echo "  GPU ${GPU_THIAZIDE} → ${ARM_B}/ (counterfactual: ACEi concept swapped to thiazide)"
    fi
    echo "============================================================"

    mkdir -p "${TRAJ_DIR}"

    # Competing concept IDs to suppress per arm:
    #   arm_a (ACEi) suppresses thiazide + all excluded antihypertensives
    #   arm_b (thiazide) suppresses ACEi  + all excluded antihypertensives
    _SUPPRESS_ARGS_A=()
    _SUPPRESS_ARGS_B=()
    if [ "${SUPPRESS_CONCEPTS}" = "true" ]; then
        _SUPPRESS_CONCEPTS_A="${THIAZIDE_CONCEPT_IDS},${EXCLUSION_CONCEPT_IDS}"
        _SUPPRESS_CONCEPTS_B="${ACEI_CONCEPT_IDS},${EXCLUSION_CONCEPT_IDS}"
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

    if [ "${SINGLE_GPU}" = "true" ]; then
        echo "Running ${ARM_A} arm …"
        CUDA_VISIBLE_DEVICES="${GPU_ACEI}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_A}:${ARM_A_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_A[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_A}.log" 2>&1
        echo "Running ${ARM_B} arm …"
        CUDA_VISIBLE_DEVICES="${GPU_ACEI}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_B}:${ARM_B_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_B[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_B}.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="${GPU_ACEI}" python "${GENERATE_SCRIPT}" \
            --arm_context "${ARM_A}:${ARM_A_CTX}" \
            "${_COMMON_GENERATE_ARGS[@]}" \
            "${_SUPPRESS_ARGS_A[@]}" \
            > "${OUTPUT_ROOT}/generate_${ARM_A}.log" 2>&1 &
        PID_A=$!

        CUDA_VISIBLE_DEVICES="${GPU_THIAZIDE}" python "${GENERATE_SCRIPT}" \
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
# STEP 4 — Estimate Hazard Ratios (ACEi vs Thiazide)
# =============================================================================
if [ "${SKIP_STEP4}" = "true" ]; then
    echo "Skipping Step 4 (SKIP_STEP4=true)"
else
    echo "============================================================"
    echo "STEP 4: Hazard ratio estimation  (${FOLLOW_UP_DAYS}-day follow-up)  [$(date '+%H:%M:%S')]"
    echo "  Arm A (treated)    : ${ARM_A}"
    echo "  Arm B (comparator) : ${ARM_B}"
    echo "  Outcomes:"
    echo "     312327  Acute myocardial infarction"
    echo "     316139  Heart failure"
    echo "    4310996  Ischemic stroke"
    echo "     376713  Cerebral hemorrhage"
    echo "    3655355  Erectile dysfunction"
    echo "     437833  Hypokalemia"
    echo "     434610  Hyperkalemia"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    _step_start
    python "${HR_SCRIPT}" \
        --trajectories_dir        "${TRAJ_DIR}" \
        --drug_info_path          "${DRUG_INFO}" \
        --observed_outcomes_path  "${CONTEXT_DIR}/observed_outcomes" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --follow_up_days          "${FOLLOW_UP_DAYS}" \
        --arm_a                   "${ARM_A}" \
        --arm_b                   "${ARM_B}" \
        --output_dir              "${RESULTS_DIR}" \
        --vocab_path              "${VOCAB_PATH}" \
        --arm_a_concept_ids       "${ACEI_CONCEPT_IDS}" \
        --arm_b_concept_ids       "${THIAZIDE_CONCEPT_IDS}" \
        --exclusion_concept_ids   "${EXCLUSION_CONCEPT_IDS}" \
        --era_gap_days            "${ERA_GAP_DAYS}"
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
echo "  ${RESULTS_DIR}/observed_hazard_ratio_summary.csv  (observed ACEi vs observed Thiazide from Step 1 data)"
echo "  ${RESULTS_DIR}/faithfulness_summary.csv           (generated Thiazide vs observed Thiazide, expect HR≈1)"
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv           (generated ACEi vs generated Thiazide)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
echo "Published LEGEND-HTN reference HRs (ACEi vs Thiazide):"
echo "  AMI           : 0.99  (95% CI 0.87–1.13)"
echo "  Heart failure : 1.18  (95% CI 1.07–1.30)"
echo "  Stroke        : 1.09  (95% CI 0.99–1.20)"
echo "============================================================"
