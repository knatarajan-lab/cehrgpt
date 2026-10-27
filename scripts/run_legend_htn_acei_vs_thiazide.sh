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
# Outcomes
# --------
#   Step 0 reads the LEGEND-HTN outcomes and negative controls from the SQL of the LEGEND
#   study (LEGEND_DIR) and writes outcome_definitions.json.  The 15 outcomes that the study
#   links to an inpatient or ER visit (acute myocardial infarction, acute pancreatitis, acute
#   renal failure, anaphylactoid reaction, angioedema, cardiovascular disease,
#   gastrointestinal bleeding, hemorrhagic stroke, hospitalization with heart failure,
#   ischemic stroke, rhabdomyolysis, stroke, sudden cardiac death, transient ischemic attack
#   and unstable angina) only count inside such a visit, as does coronary heart disease in
#   the SQL.
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
ERA_GAP_DAYS="${ERA_GAP_DAYS:-30}"
# New-user washout: days of observed history required before the index drug.
MIN_PRIOR_OBSERVATION_DAYS="${MIN_PRIOR_OBSERVATION_DAYS:-365}"

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
MONOTHERAPY_WINDOW_DAYS="${MONOTHERAPY_WINDOW_DAYS:-7}"
POPULATION_WINDOW_DAYS="${POPULATION_WINDOW_DAYS:-365}"
TIME_AT_RISK="${TIME_AT_RISK:-on_treatment}"      # or intent_to_treat
MIN_DAYS_AT_RISK="${MIN_DAYS_AT_RISK:-1}"
CENSOR_AT_SWITCH="${CENSOR_AT_SWITCH:-false}"

# Set to "true" to suppress the other arm's drugs and the exclusion drugs during generation.
# LEGEND does not (patients may add drugs), so the default is false.
SUPPRESS_CONCEPTS="${SUPPRESS_CONCEPTS:-false}"

# Comma-separated OMOP concept_ids restricting the eligible population: one of them (with
# descendants) has to be recorded within POPULATION_WINDOW_DAYS up to the index date.
# 316866 = Hypertensive disorder, as in LEGEND.  Leave empty ("") for no filter.
POPULATION_CONCEPT_IDS="316866"

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

# Several GPUs for the generation step: "all" or a list such as "0,1,2,3".  One worker per GPU
# generates a share of the batches of both arms, so every GPU stays busy; an interrupted run
# continues where it stopped (rerun the script with SKIP_STEP1=true SKIP_STEP2=true).  Takes
# precedence over GPU_ACEI / GPU_THIAZIDE / SINGLE_GPU.  Empty = the settings above.
GPU_IDS="${GPU_IDS:-}"

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

# Outcomes: the LEGEND-HTN outcomes and negative controls, read from the SQL of the LEGEND study
# (https://github.com/OHDSI/Legend).  The concepts of every outcome (descendants, exclusions) and
# the inpatient / ER visit restriction of the outcomes that need one come from there; outcomes the
# token sequences cannot represent are listed in Step 0 and left out.
LEGEND_DIR="${LEGEND_DIR:-/data/Legend}"

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

CONTEXT_DIR="${OUTPUT_ROOT}/contexts"
SWAP_DIR="${OUTPUT_ROOT}/swap_contexts"
OVERLAP_SCORES="${SWAP_DIR}/overlap_scores.parquet"
TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"
OUTCOME_DEFS_DIR="${OUTPUT_ROOT}/outcome_definitions"
OUTCOME_DEFINITIONS="${OUTCOME_DEFS_DIR}/outcome_definitions.json"
OUTCOME_SCRIPT="src/cehrgpt/analysis/counterfactual/legend_outcomes.py"

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
# STEP 0 — Outcome definitions from the SQL of the LEGEND study
#
#   Writes outcome_definitions.json (the concepts of every outcome, with descendants and
#   exclusions resolved, and its inpatient / ER restriction) and the raw concept sets.
# =============================================================================
echo "============================================================"
echo "STEP 0: LEGEND outcome definitions  [$(date '+%H:%M:%S')]"
echo "============================================================"
mkdir -p "${OUTCOME_DEFS_DIR}"
python "${OUTCOME_SCRIPT}" \
    --legend_dir  "${LEGEND_DIR}" \
    --indication  Hypertension \
    --vocab_path  "${VOCAB_PATH}" \
    --output_dir  "${OUTCOME_DEFS_DIR}"
echo ""

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

    _step_start
    python "${EXTRACT_SCRIPT}" \
        --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
        --vocab_path              "${VOCAB_PATH}" \
        --source_concept_ids      "${ACEI_CONCEPT_IDS}" \
        --comparator_concept_ids  "${THIAZIDE_CONCEPT_IDS}" \
        --output_dir              "${CONTEXT_DIR}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
        --outcome_concept_ids     "${EXTRACT_OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --outcome_definitions_path "${OUTCOME_DEFINITIONS}" \
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
    if [ -n "${PREFERENCE_SCORE_TRIM}" ]; then
        _SWAP_EXTRA_ARGS+=(
            --model_path "${MODEL_PATH}"
            --preference_score_trim "${PREFERENCE_SCORE_TRIM}"
            --overlap_scores_path "${OVERLAP_SCORES}"
            --context_length "${GENERATION_INPUT_LENGTH}"
            --batch_size "${BATCH_SIZE}"
        )
    fi
    if [ -n "${MAX_ACEI_PATIENTS:-}" ]; then
        _SWAP_EXTRA_ARGS+=(--max_source_patients "${MAX_ACEI_PATIENTS}")
    fi

    _step_start
    python "${SWAP_SCRIPT}" \
        --treated_context_path    "${TREATED_CTX}" \
        --drug_info_path          "${DRUG_INFO}" \
        --vocab_path              "${VOCAB_PATH}" \
        --tokenizer_path          "${TOKENIZER_PATH}" \
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
    if [ -n "${GPU_IDS}" ]; then
        echo "  Mode              : one worker per GPU (GPU_IDS=${GPU_IDS}), both arms in one run"
    elif [ "${SINGLE_GPU}" = "true" ]; then
        echo "  Mode              : sequential (SINGLE_GPU=true, GPU ${GPU_ACEI})"
    else
        echo "  GPU ${GPU_ACEI} → ${ARM_A}/     (conditioned on ACEi initiation)"
        echo "  GPU ${GPU_THIAZIDE} → ${ARM_B}/ (counterfactual: ACEi concept swapped to thiazide)"
    fi
    echo "  Suppress concepts : ${SUPPRESS_CONCEPTS}"
    echo "  Era gap days      : ${ERA_GAP_DAYS}"
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
    echo "  Outcomes: ${OUTCOME_DEFINITIONS}"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    _step_start
    _HR_EXTRA_ARGS=()
    _HR_EXTRA_ARGS+=(--prior_outcome_path "${CONTEXT_DIR}/observed_outcomes")
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
        _HR_EXTRA_ARGS+=(--overlap_scores_path "${OVERLAP_SCORES}")
    fi
    python "${HR_SCRIPT}" \
        --trajectories_dir        "${TRAJ_DIR}" \
        --drug_info_path          "${DRUG_INFO}" \
        --observed_outcomes_path  "${CONTEXT_DIR}/observed_outcomes" \
        --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
        --inpatient_outcome_concept_ids "${INPATIENT_OUTCOME_CONCEPT_IDS}" \
        --outcome_definitions_path "${OUTCOME_DEFINITIONS}" \
        --follow_up_days          "${FOLLOW_UP_DAYS}" \
        --arm_a                   "${ARM_A}" \
        --arm_b                   "${ARM_B}" \
        --output_dir              "${RESULTS_DIR}" \
        --vocab_path              "${VOCAB_PATH}" \
        --arm_a_concept_ids       "${ACEI_CONCEPT_IDS}" \
        --arm_b_concept_ids       "${THIAZIDE_CONCEPT_IDS}" \
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
echo "  ${RESULTS_DIR}/observed_hazard_ratio_summary.csv  (observed ACEi vs observed Thiazide from Step 1 data)"
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv           (generated ACEi vs generated Thiazide)"
echo "  ${RESULTS_DIR}/control_outcome_summary.csv        (negative / positive controls, when given)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
echo "Published LEGEND-HTN reference HRs (ACEi vs Thiazide):"
echo "  AMI           : 0.99  (95% CI 0.87–1.13)"
echo "  Heart failure : 1.18  (95% CI 1.07–1.30)"
echo "  Stroke        : 1.09  (95% CI 0.99–1.20)"
echo "============================================================"
