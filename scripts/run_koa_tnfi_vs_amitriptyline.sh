#!/usr/bin/env bash
# =============================================================================
# KOA Study: TNFi vs Amitriptyline — Total Knee Replacement outcome
# =============================================================================
#
# Study design
# ------------
#   Population  : Knee osteoarthritis (KOA) patients who newly initiated
#                 a TNF inhibitor (new users with prior KOA diagnosis).
#   Arm A (TNFi):
#                 History up to and including the TNFi initiation visit.
#                 Model generates the future conditional on TNFi.
#   Arm B (Amitriptyline counterfactual):
#                 Same history, but the TNFi concept is swapped to an
#                 amitriptyline concept sampled proportional to the empirical
#                 frequency of amitriptyline drugs among real amitriptyline
#                 initiators.  Model generates the counterfactual future.
#   Each patient produces N trajectories in each arm.
#   HR is computed by comparing TNFi vs amitriptyline generated
#   trajectories for the TKR outcome.
#
# Population filter
# -----------------
#   Only patients with ≥1 KOA concept in their pre-drug history are kept
#   (via --population_concept_ids in Step 1).
#   Amitriptyline patients are included in Step 1 only to estimate the
#   empirical frequency of amitriptyline formulations for the drug swap.
#
# Outcome (OMOP concept_ids)
# --------------------------
#   43531648  Total knee replacement
#    2105103  Total knee replacement (additional code)
#
# Population concepts (OMOP)
# --------------------------
#   4079750  Primary osteoarthritis of knee (SNOMED 57667006)
#
# Drug concept IDs (OMOP ingredient)
# -----------------------------------
#   TNFi          : 1119119, 4299871, 937368, 912263, 19041065
#   Amitriptyline : 710062
#
# Usage
# -----
#   Edit the CONFIGURATION section, then:
#     nohup bash scripts/run_koa_tnfi_vs_amitriptyline.sh \
#         > koa_tnfi_vs_amitriptyline.log 2>&1 &
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
OUTPUT_ROOT="/data/koa_tnfi_vs_amitriptyline"

# Number of stochastic trajectories per patient per arm
NUM_TRAJECTORIES=10

# Maximum TNFi patients to use (random sample); set to "" to use all
MAX_SOURCE_PATIENTS=10000

BATCH_SIZE=8
GENERATION_INPUT_LENGTH=2048
GENERATION_MAX_NEW_TOKENS=1024
FOLLOW_UP_DAYS=730
NUM_WORKERS=4

# Number of parallel workers for Step 1 sequence extraction
EXTRACTION_NUM_WORKERS=10

# =============================================================================
# DRUG CONCEPT IDs (OMOP ingredient)
# =============================================================================

TNFI_CONCEPT_IDS="1119119,4299871,937368,912263,19041065"
# TNF inhibitors (Etanercept, Adalimumab, Infliximab, Certolizumab, Golimumab)

AMITRIPTYLINE_CONCEPT_IDS="710062"
# Amitriptyline (TCA)

# =============================================================================
# POPULATION CONCEPT IDs (OMOP condition)
# =============================================================================

KOA_CONCEPT_IDS="4079750"
# Primary osteoarthritis of knee (SNOMED 57667006)
# Add additional KOA-related concept_ids separated by commas if needed

# =============================================================================
# OUTCOME CONCEPT IDs (OMOP procedure)
# =============================================================================

TKR_CONCEPT_IDS="43531648,2105103"
# Total knee replacement

# =============================================================================
# DERIVED PATHS
# =============================================================================

CONTEXT_DIR="${OUTPUT_ROOT}/contexts"
SWAP_DIR="${OUTPUT_ROOT}/swap_contexts"
TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"

TREATED_CTX="${CONTEXT_DIR}/treated_context"
DRUG_INFO="${CONTEXT_DIR}/drug_info"

# TNFi patients — actual TNFi context (arm A input)
TNFI_ARM_CTX="${SWAP_DIR}/tnfi_arm_ctx.parquet"
# TNFi patients — amitriptyline counterfactual context (arm B input)
AMITRIPTYLINE_ARM_CTX="${SWAP_DIR}/amitriptyline_arm_ctx.parquet"

EXTRACT_SCRIPT="src/cehrgpt/analysis/counterfactual/extract_drug_initiation_sequences.py"
SWAP_SCRIPT="src/cehrgpt/analysis/counterfactual/create_drug_swap_context.py"
GENERATE_SCRIPT="src/cehrgpt/analysis/counterfactual/generate_counterfactual_sequences.py"
HR_SCRIPT="src/cehrgpt/analysis/counterfactual/hazard_ratio_estimation.py"

# =============================================================================
# STEP 1 — Extract drug initiation sequences (KOA population only)
#
#   Include BOTH TNFi AND amitriptyline patients so that drug_info/
#   contains real amitriptyline prescriptions for frequency estimation in
#   Step 2.  Population filter requires a KOA diagnosis in pre-drug history.
# =============================================================================
echo "============================================================"
echo "STEP 1: Extract drug initiation sequences (KOA population)"
echo "  Source      : ${TNFI_CONCEPT_IDS}  (TNFi)"
echo "  Comparator  : ${AMITRIPTYLINE_CONCEPT_IDS}  (Amitriptyline)"
echo "  Population  : ${KOA_CONCEPT_IDS}  (KOA diagnosis required)"
echo "  Outcome     : ${TKR_CONCEPT_IDS}  (Total Knee Replacement)"
echo "============================================================"

mkdir -p "${CONTEXT_DIR}"

python "${EXTRACT_SCRIPT}" \
    --patient_sequence_path   "${PATIENT_SEQUENCE_PATH}" \
    --vocab_path              "${VOCAB_PATH}" \
    --source_concept_ids      "${TNFI_CONCEPT_IDS}" \
    --comparator_concept_ids  "${AMITRIPTYLINE_CONCEPT_IDS}" \
    --population_concept_ids  "${KOA_CONCEPT_IDS}" \
    --output_dir              "${CONTEXT_DIR}" \
    --tokenizer_path          "${TOKENIZER_PATH}" \
    --outcome_concept_ids     "${TKR_CONCEPT_IDS}" \
    --min_context_length      4 \
    --num_workers             "${EXTRACTION_NUM_WORKERS}"

echo "Step 1 complete."
echo ""

# =============================================================================
# STEP 2 — Create TNFi (actual) and amitriptyline (counterfactual) contexts
#
#   For each TNFi patient we produce:
#     Arm A (tnfi_arm_ctx)          : treated_context as-is
#     Arm B (amitriptyline_arm_ctx) : treated_context with TNFi concept
#                                     swapped to a frequency-sampled amitriptyline
#                                     concept (empirical distribution from real
#                                     amitriptyline patients in drug_info/)
# =============================================================================
echo "============================================================"
echo "STEP 2: Create TNFi (actual) and Amitriptyline (counterfactual) contexts"
echo "  Source      : TNFi patients"
echo "  Comparator  : Amitriptyline (frequency-sampled from real Rx)"
echo "============================================================"

mkdir -p "${SWAP_DIR}"

_SWAP_EXTRA_ARGS=()
if [ -n "${MAX_SOURCE_PATIENTS:-}" ]; then
    _SWAP_EXTRA_ARGS+=(--max_source_patients "${MAX_SOURCE_PATIENTS}")
fi

python "${SWAP_SCRIPT}" \
    --treated_context_path    "${TREATED_CTX}" \
    --drug_info_path          "${DRUG_INFO}" \
    --vocab_path              "${VOCAB_PATH}" \
    --source_concept_ids      "${TNFI_CONCEPT_IDS}" \
    --comparator_concept_ids  "${AMITRIPTYLINE_CONCEPT_IDS}" \
    --output_path             "${AMITRIPTYLINE_ARM_CTX}" \
    --also_write_treated_path "${TNFI_ARM_CTX}" \
    "${_SWAP_EXTRA_ARGS[@]}"

echo ""
echo "Arm sizes:"
SWAP_DIR="${SWAP_DIR}" python - <<'PYEOF'
import polars as pl, os
swap_dir = os.environ["SWAP_DIR"]
for label, f in [("TNFi arm (actual)", f"{swap_dir}/tnfi_arm_ctx.parquet"),
                 ("Amitriptyline arm (counterfactual)", f"{swap_dir}/amitriptyline_arm_ctx.parquet")]:
    n = len(pl.read_parquet(f))
    print(f"  {label}: {n:,} patients")
PYEOF

echo "Step 2 complete."
echo ""

# =============================================================================
# STEP 3 — Generate trajectories for both arms in parallel on two GPUs
#
#   GPU 0: tnfi/           — N futures conditioned on TNFi receipt
#   GPU 1: amitriptyline/  — N counterfactual futures (TNFi→amitriptyline)
# =============================================================================
echo "============================================================"
echo "STEP 3: Generate trajectories  (${NUM_TRAJECTORIES} per patient per arm)"
echo "  GPU 0 → tnfi/          (conditioned on TNFi initiation)"
echo "  GPU 1 → amitriptyline/ (counterfactual: TNFi swapped to amitriptyline)"
echo "============================================================"

mkdir -p "${TRAJ_DIR}"

CUDA_VISIBLE_DEVICES=0 python "${GENERATE_SCRIPT}" \
    --arm_context "tnfi:${TNFI_ARM_CTX}" \
    --model_name_or_path        "${MODEL_PATH}" \
    --tokenizer_path            "${TOKENIZER_PATH}" \
    --output_dir                "${TRAJ_DIR}" \
    --num_trajectories          "${NUM_TRAJECTORIES}" \
    --batch_size                "${BATCH_SIZE}" \
    --generation_input_length   "${GENERATION_INPUT_LENGTH}" \
    --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}" \
    --num_workers               "${NUM_WORKERS}" \
    > "${OUTPUT_ROOT}/generate_tnfi.log" 2>&1 &
PID_TNFI=$!

CUDA_VISIBLE_DEVICES=1 python "${GENERATE_SCRIPT}" \
    --arm_context "amitriptyline:${AMITRIPTYLINE_ARM_CTX}" \
    --model_name_or_path        "${MODEL_PATH}" \
    --tokenizer_path            "${TOKENIZER_PATH}" \
    --output_dir                "${TRAJ_DIR}" \
    --num_trajectories          "${NUM_TRAJECTORIES}" \
    --batch_size                "${BATCH_SIZE}" \
    --generation_input_length   "${GENERATION_INPUT_LENGTH}" \
    --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}" \
    --num_workers               "${NUM_WORKERS}" \
    > "${OUTPUT_ROOT}/generate_amitriptyline.log" 2>&1 &
PID_AMITRIPTYLINE=$!

echo "TNFi generation          PID=${PID_TNFI}           (log: ${OUTPUT_ROOT}/generate_tnfi.log)"
echo "Amitriptyline generation PID=${PID_AMITRIPTYLINE}  (log: ${OUTPUT_ROOT}/generate_amitriptyline.log)"
echo "Waiting for both to complete …"

wait ${PID_TNFI}
STATUS_TNFI=$?
wait ${PID_AMITRIPTYLINE}
STATUS_AMITRIPTYLINE=$?

if [ ${STATUS_TNFI} -ne 0 ]; then
    echo "ERROR: TNFi generation failed (exit ${STATUS_TNFI}). Check ${OUTPUT_ROOT}/generate_tnfi.log"
    exit ${STATUS_TNFI}
fi
if [ ${STATUS_AMITRIPTYLINE} -ne 0 ]; then
    echo "ERROR: Amitriptyline generation failed (exit ${STATUS_AMITRIPTYLINE}). Check ${OUTPUT_ROOT}/generate_amitriptyline.log"
    exit ${STATUS_AMITRIPTYLINE}
fi

echo "Step 3 complete."
echo ""

# =============================================================================
# STEP 4 — Estimate Hazard Ratios (TNFi vs Amitriptyline)
# =============================================================================
echo "============================================================"
echo "STEP 4: Hazard ratio estimation  (${FOLLOW_UP_DAYS}-day follow-up)"
echo "  Outcome:"
echo "    ${TKR_CONCEPT_IDS}  Total Knee Replacement"
echo "============================================================"

mkdir -p "${RESULTS_DIR}"

python "${HR_SCRIPT}" \
    --trajectories_dir        "${TRAJ_DIR}" \
    --drug_info_path          "${DRUG_INFO}" \
    --outcome_concept_ids     "${TKR_CONCEPT_IDS}" \
    --follow_up_days          "${FOLLOW_UP_DAYS}" \
    --output_dir              "${RESULTS_DIR}" \
    --vocab_path              "${VOCAB_PATH}" \
    --arm_a                   "tnfi" \
    --arm_b                   "amitriptyline" \
    --arm_a_concept_ids       "${TNFI_CONCEPT_IDS}" \
    --arm_b_concept_ids       "${AMITRIPTYLINE_CONCEPT_IDS}"

echo "Step 4 complete."
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "============================================================"
echo "Done.  Output directory: ${OUTPUT_ROOT}"
echo ""
echo "Key files:"
echo "  ${RESULTS_DIR}/faithfulness_summary.csv  (generated Amitriptyline vs observed Amitriptyline, expect HR≈1)"
echo "  ${RESULTS_DIR}/hazard_ratio_summary.csv  (generated TNFi vs generated Amitriptyline)"
echo "  ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
echo "Concept IDs used:"
echo "  TNFi          : ${TNFI_CONCEPT_IDS}"
echo "  Amitriptyline : ${AMITRIPTYLINE_CONCEPT_IDS}"
echo "  KOA filter    : ${KOA_CONCEPT_IDS}"
echo "  TKR outcome   : ${TKR_CONCEPT_IDS}"
echo "============================================================"
