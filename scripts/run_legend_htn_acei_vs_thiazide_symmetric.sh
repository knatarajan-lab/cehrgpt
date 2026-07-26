#!/usr/bin/env bash
# =============================================================================
# LEGEND-HTN Replication: ACEi vs Thiazide (symmetric combined-population design)
# =============================================================================
#
# Study design
# ------------
#   Population  : ALL patients who initiated either an ACE inhibitor OR a
#                 thiazide/thiazide-like diuretic (new users).
#   Arm A (ACEi): Each patient's drug initiation is stripped from their
#                 context and replaced by a frequency-sampled ACEi concept.
#                 Model generates the future conditional on ACEi.
#   Arm B (Thiazide): Same stripped context, but the drug slot is filled
#                 with a frequency-sampled thiazide concept instead.
#                 Model generates the future conditional on thiazide.
#
# Why this is symmetric
# ---------------------
#   In the original (asymmetric) design the study population is exclusively
#   ACEi initiators.  Their pre-index history forms the context; the ACEi
#   initiation is kept or swapped.  This means the model is always
#   conditioning on ACEi patient characteristics.
#
#   Here we pool BOTH drug-class patients, strip the actual drug from
#   every patient's context, and inject the arm-assigned drug.  The same
#   set of pre-index histories is exposed to both treatments, eliminating
#   the population-composition asymmetry.
#
# Pipeline overview
# -----------------
#   Step 1a : Extract drug initiation sequences — ACEi as source,
#             thiazide as comparator (to learn thiazide ingredient frequencies).
#   Step 1b : Extract drug initiation sequences — thiazide as source,
#             ACEi as comparator (to learn ACEi ingredient frequencies).
#   Step 2  : Run create_drug_swap_context.py twice — once with ACEi as
#             source (producing acei_pts_acei + acei_pts_thiazide contexts)
#             and once with thiazide as source (thiazide_pts_thiazide +
#             thiazide_pts_acei).  Then concat into:
#               Arm A (ACEi)     = acei_pts_acei + thiazide_pts_acei
#               Arm B (Thiazide) = acei_pts_thiazide + thiazide_pts_thiazide
#             Also merges both drug_info tables into one.
#   Step 3  : Generate N trajectories per patient per arm.
#   Step 4  : Estimate hazard ratios from generated trajectories.
#
# Outcomes (OMOP concept_ids)
# ----------------------------
#    312327  Acute myocardial infarction
#    316139  Heart failure
#   4310996  Ischemic stroke
#    376713  Cerebral hemorrhage
#   3655355  Erectile dysfunction
#    437833   Hypokalemia
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
#     bash run_legend_htn_acei_vs_thiazide_symmetric.sh
#
#   To skip steps already completed:
#     SKIP_STEP1=true SKIP_STEP2=true bash run_legend_htn_acei_vs_thiazide_symmetric.sh
#
#   To run both arms on a single GPU sequentially:
#     SINGLE_GPU=true bash run_legend_htn_acei_vs_thiazide_symmetric.sh
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
OUTPUT_ROOT="/data/legend_htn_acei_vs_thiazide_symmetric"

# Number of stochastic trajectories per patient per arm
NUM_TRAJECTORIES=5

# Maximum patients to draw from each drug class before combining.
# Set to "" to use all patients from each class.
MAX_ACEI_PATIENTS=500
MAX_THIAZIDE_PATIENTS=500

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
ERA_GAP_DAYS="${ERA_GAP_DAYS:-30}"

# Set to "true" to suppress competing drug concepts during generation.
SUPPRESS_CONCEPTS="${SUPPRESS_CONCEPTS:-true}"

# Optional: comma-separated OMOP concept_ids restricting the eligible population.
# Leave empty ("") to apply no population filter.
POPULATION_CONCEPT_IDS=""

# Exclusion concept IDs — "new users" eligibility criterion.
# Patients who have ANY of these drugs in their pre-index history are excluded.
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
#    1344965  Guanfacine     1373928  Hydralazine    1386957  Labetalol
#    1305447  Methyldopa     1307046  Metoprolol    1309068  Minoxidil
#    1313200  Nadolol        1314577  Nebivolol      1327978  Penbutolol
#    1345858  Pindolol       1350489  Prazosin       1353766  Propranolol
#     970250  Spironolactone 1341238  Terazosin       942350  Torsemide
#     904542  Triamterene
EXCLUSION_CONCEPT_IDS="40235485,1351557,1346686,1347384,1367500,40226742,1317640,1308842,1332418,1353776,1326012,1318137,1318853,1319880,1328165,1307863,1319998,1317967,991382,1314002,1322081,1338005,932745,1346823,1398937,1363053,1309799,956874,1344965,1373928,1386957,1305447,1307046,1309068,1313200,1314577,1327978,1345858,1350489,1353766,970250,1341238,942350,904542"

# -----------------------------------------------------------------------------
# GPU configuration
# -----------------------------------------------------------------------------
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
# ARM NAMES
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

OUTCOME_CONCEPT_IDS="312327,316139,4310996,376713,3655355,437833,434610"
# 312327   Acute myocardial infarction
# 316139   Heart failure
# 4310996  Ischemic stroke
# 376713   Cerebral hemorrhage
# 3655355  Erectile dysfunction
# 437833   Hypokalemia
# 434610   Hyperkalemia

INPATIENT_OUTCOME_CONCEPT_IDS="312327,316139,4310996,376713"

# =============================================================================
# DERIVED PATHS
# =============================================================================

# Separate extraction output directories per drug class
ACEI_CONTEXT_DIR="${OUTPUT_ROOT}/acei_contexts"
THIAZIDE_CONTEXT_DIR="${OUTPUT_ROOT}/thiazide_contexts"

# Combined / symmetric contexts produced in Step 2
SYMMETRIC_DIR="${OUTPUT_ROOT}/symmetric_contexts"
ARM_A_CTX="${SYMMETRIC_DIR}/${ARM_A}_arm_ctx.parquet"
ARM_B_CTX="${SYMMETRIC_DIR}/${ARM_B}_arm_ctx.parquet"
COMBINED_DRUG_INFO="${SYMMETRIC_DIR}/drug_info"

TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"

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
# STEP 1 — Extract drug initiation sequences for both drug classes
#
#   1a: ACEi as source, thiazide as comparator.
#       Produces ACEi patients' treated_context and drug_info.
#       Thiazide patients in drug_info provide empirical thiazide
#       ingredient frequencies needed for frequency-weighted sampling in Step 2.
#
#   1b: Thiazide as source, ACEi as comparator.
#       Produces thiazide patients' treated_context and drug_info.
#       ACEi patients in drug_info provide empirical ACEi ingredient
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

    # --- Step 1a: ACEi patients ---
    echo "============================================================"
    echo "STEP 1a: Extract ACEi patient sequences  [$(date '+%H:%M:%S')]"
    echo "  Source     : ACEi (${ACEI_CONCEPT_IDS})"
    echo "  Comparator : thiazide (for frequency estimation)"
    echo "============================================================"

    mkdir -p "${ACEI_CONTEXT_DIR}"
    _step_start
    python "${EXTRACT_SCRIPT}" \
        --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${ACEI_CONCEPT_IDS}" \
        --comparator_concept_ids  "${THIAZIDE_CONCEPT_IDS}" \
        --output_dir              "${ACEI_CONTEXT_DIR}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --era_gap_days            "${ERA_GAP_DAYS}" \
        --min_context_length      "${MIN_CONTEXT_LENGTH}" \
        --num_workers             "${EXTRACTION_NUM_WORKERS}" \
        "${_EXTRACT_EXTRA_ARGS[@]}"
    _step_end
    echo "Step 1a complete."
    echo ""

    # --- Step 1b: Thiazide patients ---
    echo "============================================================"
    echo "STEP 1b: Extract thiazide patient sequences  [$(date '+%H:%M:%S')]"
    echo "  Source     : thiazide (${THIAZIDE_CONCEPT_IDS})"
    echo "  Comparator : ACEi (for frequency estimation)"
    echo "============================================================"

    mkdir -p "${THIAZIDE_CONTEXT_DIR}"
    _step_start
    python "${EXTRACT_SCRIPT}" \
        --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${THIAZIDE_CONCEPT_IDS}" \
        --comparator_concept_ids  "${ACEI_CONCEPT_IDS}" \
        --output_dir              "${THIAZIDE_CONTEXT_DIR}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
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
#   Run 2a — ACEi as source:
#     • acei_pts_acei_ctx.parquet   : ACEi patients, their actual ACEi drug
#                                     (treated path, written as-is)
#     • acei_pts_thiazide_ctx.parquet: ACEi patients, thiazide frequency-swapped
#
#   Run 2b — Thiazide as source:
#     • thiazide_pts_thiazide_ctx.parquet : Thiazide patients, actual thiazide
#     • thiazide_pts_acei_ctx.parquet     : Thiazide patients, ACEi freq-swapped
#
#   Arm A (ACEi)      = concat(acei_pts_acei_ctx,     thiazide_pts_acei_ctx)
#   Arm B (Thiazide)  = concat(acei_pts_thiazide_ctx, thiazide_pts_thiazide_ctx)
#   drug_info         = concat(acei_drug_info,         thiazide_drug_info)
#
#   Both arms cover the same combined patient pool → symmetric comparison.
# =============================================================================
if [ "${SKIP_STEP2}" = "true" ]; then
    echo "Skipping Step 2 (SKIP_STEP2=true)"
else
    echo "============================================================"
    echo "STEP 2: Create symmetric arm contexts  [$(date '+%H:%M:%S')]"
    echo "  ACEi patient contexts    : ${ACEI_CONTEXT_DIR}/treated_context"
    echo "  Thiazide patient contexts: ${THIAZIDE_CONTEXT_DIR}/treated_context"
    echo "  Combined drug info       : ${COMBINED_DRUG_INFO}"
    echo "  Random seed              : ${RANDOM_SEED}"
    echo "============================================================"

    mkdir -p "${SYMMETRIC_DIR}"

    # Intermediate per-run output files
    _ACEI_PTS_ACEI_CTX="${SYMMETRIC_DIR}/acei_pts_acei_ctx.parquet"
    _ACEI_PTS_THIAZIDE_CTX="${SYMMETRIC_DIR}/acei_pts_thiazide_ctx.parquet"
    _THIAZIDE_PTS_THIAZIDE_CTX="${SYMMETRIC_DIR}/thiazide_pts_thiazide_ctx.parquet"
    _THIAZIDE_PTS_ACEI_CTX="${SYMMETRIC_DIR}/thiazide_pts_acei_ctx.parquet"

    _step_start

    # --- 2a: ACEi patients → both arm contexts ---
    echo "  Step 2a: ACEi patients (source=ACEi, comparator=thiazide) …"
    _MAX_ACEI_ARG=()
    if [ -n "${MAX_ACEI_PATIENTS:-}" ]; then
        _MAX_ACEI_ARG=(--max_source_patients "${MAX_ACEI_PATIENTS}")
    fi
    python "${SWAP_SCRIPT}" \
        --treated_context_path   "${ACEI_CONTEXT_DIR}/treated_context" \
        --drug_info_path         "${ACEI_CONTEXT_DIR}/drug_info" \
        --vocab_path             "${VOCAB_PATH}" \
        --source_concept_ids     "${ACEI_CONCEPT_IDS}" \
        --comparator_concept_ids "${THIAZIDE_CONCEPT_IDS}" \
        --output_path            "${_ACEI_PTS_THIAZIDE_CTX}" \
        --also_write_treated_path "${_ACEI_PTS_ACEI_CTX}" \
        --seed                   "${RANDOM_SEED}" \
        "${_MAX_ACEI_ARG[@]}"

    # --- 2b: Thiazide patients → both arm contexts ---
    echo "  Step 2b: Thiazide patients (source=thiazide, comparator=ACEi) …"
    _MAX_THIAZIDE_ARG=()
    if [ -n "${MAX_THIAZIDE_PATIENTS:-}" ]; then
        _MAX_THIAZIDE_ARG=(--max_source_patients "${MAX_THIAZIDE_PATIENTS}")
    fi
    python "${SWAP_SCRIPT}" \
        --treated_context_path   "${THIAZIDE_CONTEXT_DIR}/treated_context" \
        --drug_info_path         "${THIAZIDE_CONTEXT_DIR}/drug_info" \
        --vocab_path             "${VOCAB_PATH}" \
        --source_concept_ids     "${THIAZIDE_CONCEPT_IDS}" \
        --comparator_concept_ids "${ACEI_CONCEPT_IDS}" \
        --output_path            "${_THIAZIDE_PTS_ACEI_CTX}" \
        --also_write_treated_path "${_THIAZIDE_PTS_THIAZIDE_CTX}" \
        --seed                   "${RANDOM_SEED}" \
        "${_MAX_THIAZIDE_ARG[@]}"

    # --- Concat arm contexts and drug_info ---
    echo "  Combining arm contexts and drug_info …"
    _ACEI_PTS_ACEI_CTX="${_ACEI_PTS_ACEI_CTX}" \
    _ACEI_PTS_THIAZIDE_CTX="${_ACEI_PTS_THIAZIDE_CTX}" \
    _THIAZIDE_PTS_THIAZIDE_CTX="${_THIAZIDE_PTS_THIAZIDE_CTX}" \
    _THIAZIDE_PTS_ACEI_CTX="${_THIAZIDE_PTS_ACEI_CTX}" \
    ARM_A_CTX="${ARM_A_CTX}" ARM_B_CTX="${ARM_B_CTX}" \
    COMBINED_DRUG_INFO="${COMBINED_DRUG_INFO}" \
    ACEI_CONTEXT_DIR="${ACEI_CONTEXT_DIR}" \
    THIAZIDE_CONTEXT_DIR="${THIAZIDE_CONTEXT_DIR}" \
    python - <<'PYEOF'
import polars as pl, os
from pathlib import Path

def read_dir_or_file(path):
    p = Path(path)
    if p.is_dir():
        return pl.read_parquet(str(p / "*.parquet"))
    return pl.read_parquet(str(p))

# Arm A (ACEi): ACEi patients with ACEi + thiazide patients with ACEi swap
arm_a = pl.concat([
    pl.read_parquet(os.environ["_ACEI_PTS_ACEI_CTX"]),
    pl.read_parquet(os.environ["_THIAZIDE_PTS_ACEI_CTX"]),
])
arm_a.write_parquet(os.environ["ARM_A_CTX"])
print(f"  Arm A (ACEi):     {len(arm_a):,} patients  → {os.environ['ARM_A_CTX']}")

# Arm B (Thiazide): ACEi patients with thiazide swap + thiazide patients with thiazide
arm_b = pl.concat([
    pl.read_parquet(os.environ["_ACEI_PTS_THIAZIDE_CTX"]),
    pl.read_parquet(os.environ["_THIAZIDE_PTS_THIAZIDE_CTX"]),
])
arm_b.write_parquet(os.environ["ARM_B_CTX"])
print(f"  Arm B (Thiazide): {len(arm_b):,} patients  → {os.environ['ARM_B_CTX']}")

# Combined drug_info
drug_info = pl.concat([
    read_dir_or_file(os.path.join(os.environ["ACEI_CONTEXT_DIR"], "drug_info")),
    read_dir_or_file(os.path.join(os.environ["THIAZIDE_CONTEXT_DIR"], "drug_info")),
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
    if [ "${SINGLE_GPU}" = "true" ]; then
        echo "  Mode              : sequential (SINGLE_GPU=true, GPU ${GPU_ACEI})"
    else
        echo "  GPU ${GPU_ACEI} → ${ARM_A}/     (conditioned on ACEi initiation)"
        echo "  GPU ${GPU_THIAZIDE} → ${ARM_B}/ (conditioned on thiazide initiation)"
    fi
    echo "  Suppress concepts : ${SUPPRESS_CONCEPTS}"
    echo "  Era gap days      : ${ERA_GAP_DAYS}"
    echo "============================================================"

    mkdir -p "${TRAJ_DIR}"

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
    echo "  Drug info          : ${COMBINED_DRUG_INFO}"
    echo "  Outcomes:"
    echo "     312327  Acute myocardial infarction"
    echo "     316139  Heart failure"
    echo "    4310996  Ischemic stroke"
    echo "     376713  Cerebral hemorrhage"
    echo "    3655355  Erectile dysfunction"
    echo "     437833   Hypokalemia"
    echo "     434610  Hyperkalemia"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    _step_start
    python "${HR_SCRIPT}" \
        --trajectories_dir        "${TRAJ_DIR}" \
        --drug_info_path          "${COMBINED_DRUG_INFO}" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
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
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv   (generated ACEi vs generated Thiazide)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
echo "Published LEGEND-HTN reference HRs (ACEi vs Thiazide):"
echo "  AMI           : 0.99  (95% CI 0.87–1.13)"
echo "  Heart failure : 1.18  (95% CI 1.07–1.30)"
echo "  Stroke        : 1.09  (95% CI 0.99–1.20)"
echo "============================================================"
