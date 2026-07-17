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
#   4329847  Myocardial infarction
#    316139  Heart failure
#    432922  Ischemic stroke
#   4319452  Hemorrhagic stroke
#   3655355  Impotence
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

# Set to "true" to suppress opposite-drug tokens during generation (recommended).
# Set to "false" to disable suppression (e.g. for ablation studies).
SUPPRESS_CONCEPTS=true

# =============================================================================
# DRUG CONCEPT IDs
# NOTE: 1395058 appears in both lists (Quinapril / Chlorthalidone) — verify.
# =============================================================================

ACEI_CONCEPT_IDS="1308216,1346654,1332418,135376,1395058,1310756,1363749,1328956,1373355,1307046"

# Outcome concept IDs
OUTCOME_CONCEPT_IDS="4329847,316139,432922,4319452,3655355,437833,434610"
# 4329847  Myocardial infarction
# 316139   Heart failure
# 432922   Ischemic stroke
# 4319452  Hemorrhagic stroke
# 3655355  Impotence
# 437833   Hypokalemia
# 434610   Hyperkalemia
# Lisinopril, Ramipril, Enalapril, Benazepril, Quinapril(*),
# Captopril, Fosinopril, Moexipril, Perindopril, Trandolapril

THIAZIDE_CONCEPT_IDS="1395058,974166,978555,907013"
# Chlorthalidone(*), HCTZ, Indapamide, Metolazone

# =============================================================================
# DERIVED PATHS
# =============================================================================

CONTEXT_DIR="${OUTPUT_ROOT}/contexts"
SWAP_DIR="${OUTPUT_ROOT}/swap_contexts"
TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"

TREATED_CTX="${CONTEXT_DIR}/treated_context"
DRUG_INFO="${CONTEXT_DIR}/drug_info"

# ACEi patients — actual ACEi context (arm A input)
ACEI_ARM_CTX="${SWAP_DIR}/acei_arm_ctx.parquet"
# ACEi patients — thiazide counterfactual context (arm B input)
THIAZIDE_ARM_CTX="${SWAP_DIR}/thiazide_arm_ctx.parquet"

EXTRACT_SCRIPT="src/cehrgpt/analysis/counterfactual/extract_drug_initiation_sequences.py"
SWAP_SCRIPT="src/cehrgpt/analysis/counterfactual/create_drug_swap_context.py"
GENERATE_SCRIPT="src/cehrgpt/analysis/counterfactual/generate_counterfactual_sequences.py"
HR_SCRIPT="src/cehrgpt/analysis/counterfactual/hazard_ratio_estimation.py"

# =============================================================================
# STEP 1 — Extract drug initiation sequences
#
#   Include BOTH ACEi AND thiazide patients so that drug_info.parquet
#   contains real thiazide prescriptions.  These are used in Step 2 to
#   compute the empirical frequency of each thiazide ingredient for
#   frequency-weighted sampling.  Thiazide patients are not used elsewhere.
# =============================================================================
echo "============================================================"
echo "STEP 1: Extract drug initiation sequences"
echo "  ACEi patients   : ${ACEI_CONCEPT_IDS}"
echo "  Thiazide (freq) : ${THIAZIDE_CONCEPT_IDS}"
echo "============================================================"

mkdir -p "${CONTEXT_DIR}"

python "${EXTRACT_SCRIPT}" \
    --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
    --vocab_path              "${VOCAB_PATH}" \
    --source_concept_ids      "${ACEI_CONCEPT_IDS}" \
    --comparator_concept_ids  "${THIAZIDE_CONCEPT_IDS}" \
    --output_dir              "${CONTEXT_DIR}" \
    --tokenizer_path          "${TOKENIZER_PATH}" \
    --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
    --min_context_length      4 \
    --num_workers             "${EXTRACTION_NUM_WORKERS}"

echo "Step 1 complete."
echo ""

# =============================================================================
# STEP 2 — Create the two arm contexts for ACEi patients
#
#   For each ACEi patient we produce:
#     Arm A (acei_arm_ctx)     : treated_context as-is (ACEi in context)
#     Arm B (thiazide_arm_ctx) : treated_context with ACEi concept swapped
#                                to a thiazide concept sampled proportional
#                                to the empirical thiazide frequency derived
#                                from real thiazide patients in drug_info.parquet
# =============================================================================
echo "============================================================"
echo "STEP 2: Create ACEi (actual) and Thiazide (counterfactual) contexts"
echo "  Source      : ACEi patients"
echo "  Comparator  : thiazide (frequency-sampled from real thiazide Rx)"
echo "============================================================"

mkdir -p "${SWAP_DIR}"

_SWAP_EXTRA_ARGS=()
if [ -n "${MAX_ACEI_PATIENTS:-}" ]; then
    _SWAP_EXTRA_ARGS+=(--max_source_patients "${MAX_ACEI_PATIENTS}")
fi

python "${SWAP_SCRIPT}" \
    --treated_context_path    "${TREATED_CTX}" \
    --drug_info_path          "${DRUG_INFO}" \
    --vocab_path              "${VOCAB_PATH}" \
    --source_concept_ids      "${ACEI_CONCEPT_IDS}" \
    --comparator_concept_ids  "${THIAZIDE_CONCEPT_IDS}" \
    --output_path             "${THIAZIDE_ARM_CTX}" \
    --also_write_treated_path "${ACEI_ARM_CTX}" \
    "${_SWAP_EXTRA_ARGS[@]}"

echo ""
echo "Arm sizes:"
SWAP_DIR="${SWAP_DIR}" python - <<'PYEOF'
import polars as pl, os
swap_dir = os.environ["SWAP_DIR"]
for label, f in [("ACEi arm (actual)", f"{swap_dir}/acei_arm_ctx.parquet"),
                 ("Thiazide arm (counterfactual)", f"{swap_dir}/thiazide_arm_ctx.parquet")]:
    n = len(pl.read_parquet(f))
    print(f"  {label}: {n:,} patients")
PYEOF

echo "Step 2 complete."
echo ""

# =============================================================================
# STEP 3 — Generate trajectories for both arms in parallel on two GPUs
#
#   GPU 0: acei/     — N futures per patient conditioned on ACEi receipt
#   GPU 1: thiazide/ — N counterfactual futures conditioned on thiazide
#
#   Both processes run simultaneously; the script waits for both to finish.
# =============================================================================
echo "============================================================"
echo "STEP 3: Generate trajectories  (${NUM_TRAJECTORIES} per patient per arm)"
echo "  GPU 0 → acei/     (conditioned on ACEi initiation)"
echo "  GPU 1 → thiazide/ (counterfactual: ACEi concept swapped to thiazide)"
echo "============================================================"

mkdir -p "${TRAJ_DIR}"

_SUPPRESS_ARGS_ACEI=()
_SUPPRESS_ARGS_THIAZIDE=()
if [ "${SUPPRESS_CONCEPTS}" = "true" ]; then
    _SUPPRESS_ARGS_ACEI=(--vocab_path "${VOCAB_PATH}" --arm_suppress_concepts "acei:${THIAZIDE_CONCEPT_IDS}")
    _SUPPRESS_ARGS_THIAZIDE=(--vocab_path "${VOCAB_PATH}" --arm_suppress_concepts "thiazide:${ACEI_CONCEPT_IDS}")
fi

CUDA_VISIBLE_DEVICES=0 python "${GENERATE_SCRIPT}" \
    --arm_context "acei:${ACEI_ARM_CTX}" \
    --model_name_or_path        "${MODEL_PATH}" \
    --tokenizer_path            "${TOKENIZER_PATH}" \
    --output_dir                "${TRAJ_DIR}" \
    --num_trajectories          "${NUM_TRAJECTORIES}" \
    --batch_size                "${BATCH_SIZE}" \
    --generation_input_length   "${GENERATION_INPUT_LENGTH}" \
    --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}" \
    --num_workers               "${NUM_WORKERS}" \
    "${_SUPPRESS_ARGS_ACEI[@]}" \
    > "${OUTPUT_ROOT}/generate_acei.log" 2>&1 &
PID_ACEI=$!

CUDA_VISIBLE_DEVICES=1 python "${GENERATE_SCRIPT}" \
    --arm_context "thiazide:${THIAZIDE_ARM_CTX}" \
    --model_name_or_path        "${MODEL_PATH}" \
    --tokenizer_path            "${TOKENIZER_PATH}" \
    --output_dir                "${TRAJ_DIR}" \
    --num_trajectories          "${NUM_TRAJECTORIES}" \
    --batch_size                "${BATCH_SIZE}" \
    --generation_input_length   "${GENERATION_INPUT_LENGTH}" \
    --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}" \
    --num_workers               "${NUM_WORKERS}" \
    "${_SUPPRESS_ARGS_THIAZIDE[@]}" \
    > "${OUTPUT_ROOT}/generate_thiazide.log" 2>&1 &
PID_THIAZIDE=$!

echo "ACEi generation   PID=${PID_ACEI}  (log: ${OUTPUT_ROOT}/generate_acei.log)"
echo "Thiazide generation PID=${PID_THIAZIDE}  (log: ${OUTPUT_ROOT}/generate_thiazide.log)"
echo "Waiting for both to complete …"

wait ${PID_ACEI}
STATUS_ACEI=$?
wait ${PID_THIAZIDE}
STATUS_THIAZIDE=$?

if [ ${STATUS_ACEI} -ne 0 ]; then
    echo "ERROR: ACEi generation failed (exit ${STATUS_ACEI}). Check ${OUTPUT_ROOT}/generate_acei.log"
    exit ${STATUS_ACEI}
fi
if [ ${STATUS_THIAZIDE} -ne 0 ]; then
    echo "ERROR: Thiazide generation failed (exit ${STATUS_THIAZIDE}). Check ${OUTPUT_ROOT}/generate_thiazide.log"
    exit ${STATUS_THIAZIDE}
fi

echo "Step 3 complete."
echo ""

# =============================================================================
# STEP 4 — Estimate Hazard Ratios (ACEi vs Thiazide)
# =============================================================================
echo "============================================================"
echo "STEP 4: Hazard ratio estimation  (${FOLLOW_UP_DAYS}-day follow-up)"
echo "  Outcomes:"
echo "    4329847  Myocardial infarction"
echo "     316139  Heart failure"
echo "     432922  Ischemic stroke"
echo "    4319452  Hemorrhagic stroke"
echo "    3655355  Impotence"
echo "     437833  Hypokalemia"
echo "     434610  Hyperkalemia"
echo "============================================================"

mkdir -p "${RESULTS_DIR}"

python "${HR_SCRIPT}" \
    --trajectories_dir        "${TRAJ_DIR}" \
    --drug_info_path          "${DRUG_INFO}" \
    --observed_outcomes_path  "${CONTEXT_DIR}/observed_outcomes" \
    --outcome_concept_ids     "${OUTCOME_CONCEPT_IDS}" \
    --follow_up_days          "${FOLLOW_UP_DAYS}" \
    --output_dir              "${RESULTS_DIR}" \
    --vocab_path              "${VOCAB_PATH}" \
    --arm_a_concept_ids       "${ACEI_CONCEPT_IDS}" \
    --arm_b_concept_ids       "${THIAZIDE_CONCEPT_IDS}"

echo "Step 4 complete."
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "============================================================"
echo "Done.  Output directory: ${OUTPUT_ROOT}"
echo ""
echo "Key files:"
echo "  ${RESULTS_DIR}/observed_hazard_ratio_summary.csv  (Step 1.5: observed ACEi vs observed Thiazide)"
echo "  ${RESULTS_DIR}/faithfulness_summary.csv           (Step 4:   generated Thiazide vs observed Thiazide, expect HR≈1)"
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv           (Step 4:   generated ACEi vs generated Thiazide)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
echo "Published LEGEND-HTN reference HRs (ACEi vs Thiazide):"
echo "  AMI           : 0.99  (95% CI 0.87–1.13)"
echo "  Heart failure : 1.18  (95% CI 1.07–1.30)"
echo "  Stroke        : 1.09  (95% CI 0.99–1.20)"
echo "============================================================"
