#!/usr/bin/env bash
# =============================================================================
# LEGEND-HTN Replication: ACEi vs Thiazide
# =============================================================================
#
# End-to-end counterfactual comparative effectiveness study comparing
# ACE inhibitors (ACEi) to thiazide diuretics for cardiovascular outcomes,
# replicating the primary comparison from LEGEND-HTN (Suchard et al., Lancet 2019).
#
# Study design
# ------------
#   Population : New users of ACEi OR thiazide diuretics with ≥365 days
#                of prior observation and a hypertension diagnosis.
#   Counterfactual approach:
#     - For each patient who received ACEi: generate trajectories under
#       (a) ACEi (their actual drug) and (b) thiazide (drug swap → HCTZ).
#     - For each patient who received thiazide: generate trajectories under
#       (a) thiazide (actual) and (b) ACEi (drug swap → lisinopril).
#     - Pool all "ACEi" trajectories vs all "Thiazide" trajectories → HR.
#   Outcomes : Acute MI, Hospitalized heart failure, Ischemic stroke,
#              Hemorrhagic stroke (LEGEND-HTN primary outcomes).
#
# OMOP concept IDs used
# ---------------------
#   ACEi class ancestor     : 21600381
#   Thiazide class ancestor : 21601664
#   Lisinopril (ACEi swap)  : 1308216
#   HCTZ (thiazide swap)    :  974166
#   Acute MI outcome        : 4329847
#   Heart failure outcome   :  316139
#   Ischemic stroke outcome : 4110192
#   Hemorrhagic stroke      :  376713
#
# Expected LEGEND-HTN published HRs (ACEi vs Thiazide):
#   AMI           : 0.99 (95% CI 0.87–1.13)
#   Heart failure : 1.18 (95% CI 1.07–1.30)  [ACEi worse]
#   Stroke        : 1.09 (95% CI 0.99–1.20)
#
# Usage
# -----
#   Edit the CONFIGURATION section below, then:
#     bash run_legend_htn_acei_vs_thiazide.sh
#
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================

# Pre-tokenised patient sequence parquet file or folder
PATIENT_SEQUENCE_PATH="/data/patient_sequence"

# OMOP vocabulary root (must contain concept_ancestor/ sub-directory)
VOCAB_PATH="/data/omop_vocab"

# Pretrained CEHR-GPT model directory
MODEL_PATH="/models/cehrgpt"

# CehrGptTokenizer directory
TOKENIZER_PATH="/models/tokenizer"

# Root output directory (sub-directories are created automatically)
OUTPUT_ROOT="/data/legend_htn_acei_vs_thiazide"

# Number of stochastic trajectories per patient per arm
# Higher = more stable HR estimates; 50 is a good starting point.
NUM_TRAJECTORIES=50

# Batch size for GPU generation (reduce if OOM)
BATCH_SIZE=8

# Context window length fed to the model
GENERATION_INPUT_LENGTH=1024

# Max new tokens to generate per trajectory
GENERATION_MAX_NEW_TOKENS=1024

# Follow-up window in days for outcome ascertainment
FOLLOW_UP_DAYS=365

# Number of DataLoader workers (0 = main process only; increase if IO-bound)
NUM_WORKERS=4

# =============================================================================
# DERIVED PATHS (no need to edit below this line)
# =============================================================================

CONTEXT_DIR="${OUTPUT_ROOT}/contexts"
SWAP_DIR="${OUTPUT_ROOT}/swap_contexts"
TRAJ_DIR="${OUTPUT_ROOT}/trajectories"
RESULTS_DIR="${OUTPUT_ROOT}/results"

# Context parquets produced by extract_drug_initiation_sequences.py
TREATED_CTX="${CONTEXT_DIR}/treated_context.parquet"
DRUG_INFO="${CONTEXT_DIR}/drug_info.parquet"

# Arm-specific context parquets produced by create_drug_swap_context.py
ACEI_PTS_ACEI_CTX="${SWAP_DIR}/acei_pts_acei_ctx.parquet"
ACEI_PTS_THIAZIDE_CTX="${SWAP_DIR}/acei_pts_thiazide_ctx.parquet"
THIAZIDE_PTS_THIAZIDE_CTX="${SWAP_DIR}/thiazide_pts_thiazide_ctx.parquet"
THIAZIDE_PTS_ACEI_CTX="${SWAP_DIR}/thiazide_pts_acei_ctx.parquet"

# Pooled arm contexts fed to the generation step
ACEI_ARM_CTX="${SWAP_DIR}/acei_arm_ctx.parquet"
THIAZIDE_ARM_CTX="${SWAP_DIR}/thiazide_arm_ctx.parquet"

# Script paths (relative to repo root)
EXTRACT_SCRIPT="src/cehrgpt/analysis/counterfactual/extract_drug_initiation_sequences.py"
SWAP_SCRIPT="src/cehrgpt/analysis/counterfactual/create_drug_swap_context.py"
GENERATE_SCRIPT="src/cehrgpt/analysis/counterfactual/generate_counterfactual_sequences.py"
HR_SCRIPT="src/cehrgpt/analysis/counterfactual/hazard_ratio_estimation.py"

# =============================================================================
# STEP 1 — Extract partial histories up to first ACEi or thiazide exposure
# =============================================================================
echo "============================================================"
echo "STEP 1: Extract drug initiation sequences"
echo "  Population : new users of ACEi (21600381) OR Thiazide (21601664)"
echo "============================================================"

mkdir -p "${CONTEXT_DIR}"

python "${EXTRACT_SCRIPT}" \
    --patient_sequence_path "${PATIENT_SEQUENCE_PATH}" \
    --vocab_path            "${VOCAB_PATH}" \
    --drug_concept_ids      "21600381,21601664" \
    --output_dir            "${CONTEXT_DIR}" \
    --min_context_length    4

echo "Step 1 complete."
echo ""

# =============================================================================
# STEP 2 — Create drug-swap contexts
#   2a. ACEi patients  : save actual ACEi context + swap ACEi → HCTZ
#   2b. Thiazide patients : save actual thiazide context + swap thiazide → lisinopril
# =============================================================================
echo "============================================================"
echo "STEP 2a: ACEi patients — actual ACEi context + swap to HCTZ"
echo "============================================================"

mkdir -p "${SWAP_DIR}"

python "${SWAP_SCRIPT}" \
    --treated_context_path    "${TREATED_CTX}" \
    --drug_info_path          "${DRUG_INFO}" \
    --vocab_path              "${VOCAB_PATH}" \
    --source_class_id         21600381 \
    --comparator_concept_id   974166 \
    --output_path             "${ACEI_PTS_THIAZIDE_CTX}" \
    --also_write_treated_path "${ACEI_PTS_ACEI_CTX}"

echo ""
echo "============================================================"
echo "STEP 2b: Thiazide patients — actual thiazide context + swap to lisinopril"
echo "============================================================"

python "${SWAP_SCRIPT}" \
    --treated_context_path    "${TREATED_CTX}" \
    --drug_info_path          "${DRUG_INFO}" \
    --vocab_path              "${VOCAB_PATH}" \
    --source_class_id         21601664 \
    --comparator_concept_id   1308216 \
    --output_path             "${THIAZIDE_PTS_ACEI_CTX}" \
    --also_write_treated_path "${THIAZIDE_PTS_THIAZIDE_CTX}"

echo ""
echo "============================================================"
echo "STEP 2c: Pool contexts into two final arms"
echo "  ACEi arm     = ACEi patients (actual) + Thiazide patients (swapped to ACEi)"
echo "  Thiazide arm = Thiazide patients (actual) + ACEi patients (swapped to HCTZ)"
echo "============================================================"

python - <<'PYEOF'
import polars as pl, os, sys

swap_dir = os.environ.get("SWAP_DIR", sys.argv[1] if len(sys.argv) > 1 else ".")

acei_arm = pl.concat([
    pl.read_parquet(f"{swap_dir}/acei_pts_acei_ctx.parquet"),
    pl.read_parquet(f"{swap_dir}/thiazide_pts_acei_ctx.parquet"),
])
acei_arm.write_parquet(f"{swap_dir}/acei_arm_ctx.parquet")
print(f"ACEi arm: {len(acei_arm):,} patients")

thiazide_arm = pl.concat([
    pl.read_parquet(f"{swap_dir}/thiazide_pts_thiazide_ctx.parquet"),
    pl.read_parquet(f"{swap_dir}/acei_pts_thiazide_ctx.parquet"),
])
thiazide_arm.write_parquet(f"{swap_dir}/thiazide_arm_ctx.parquet")
print(f"Thiazide arm: {len(thiazide_arm):,} patients")
PYEOF

echo "Step 2 complete."
echo ""

# =============================================================================
# STEP 3 — Generate trajectories for both arms
# =============================================================================
echo "============================================================"
echo "STEP 3: Generate counterfactual trajectories"
echo "  ACEi arm     → ${TRAJ_DIR}/acei/"
echo "  Thiazide arm → ${TRAJ_DIR}/thiazide/"
echo "  Trajectories per patient: ${NUM_TRAJECTORIES}"
echo "============================================================"

mkdir -p "${TRAJ_DIR}"

python "${GENERATE_SCRIPT}" \
    --arm_context "acei:${ACEI_ARM_CTX}" \
    --arm_context "thiazide:${THIAZIDE_ARM_CTX}" \
    --model_name_or_path        "${MODEL_PATH}" \
    --tokenizer_path            "${TOKENIZER_PATH}" \
    --output_dir                "${TRAJ_DIR}" \
    --num_trajectories          "${NUM_TRAJECTORIES}" \
    --batch_size                "${BATCH_SIZE}" \
    --generation_input_length   "${GENERATION_INPUT_LENGTH}" \
    --generation_max_new_tokens "${GENERATION_MAX_NEW_TOKENS}" \
    --num_workers               "${NUM_WORKERS}"

echo "Step 3 complete."
echo ""

# =============================================================================
# STEP 4 — Estimate Hazard Ratios
# =============================================================================
echo "============================================================"
echo "STEP 4: Hazard ratio estimation (${FOLLOW_UP_DAYS}-day follow-up)"
echo "  Outcomes:"
echo "    4329847  Acute myocardial infarction"
echo "     316139  Hospitalized heart failure"
echo "    4110192  Ischemic stroke"
echo "     376713  Hemorrhagic stroke"
echo "============================================================"

mkdir -p "${RESULTS_DIR}"

python "${HR_SCRIPT}" \
    --trajectories_dir    "${TRAJ_DIR}" \
    --drug_info_path      "${DRUG_INFO}" \
    --outcome_concept_ids "4329847,316139,4110192,376713" \
    --follow_up_days      "${FOLLOW_UP_DAYS}" \
    --output_dir          "${RESULTS_DIR}"

echo "Step 4 complete."
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "============================================================"
echo "All steps complete.  Results:"
echo "  HR summary  : ${RESULTS_DIR}/hazard_ratio_summary.csv"
echo "  KM curves   : ${RESULTS_DIR}/km_<concept_id>.csv"
echo ""
echo "Published LEGEND-HTN reference HRs (ACEi vs Thiazide):"
echo "  AMI           : 0.99  (95% CI 0.87–1.13)"
echo "  Heart failure : 1.18  (95% CI 1.07–1.30)  [ACEi worse]"
echo "  Stroke        : 1.09  (95% CI 0.99–1.20)"
echo "============================================================"
