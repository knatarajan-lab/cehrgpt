#!/bin/bash

# Export environment variables
export OMOP_FOLDER=$1
export PATIENT_SPLITS_FOLDER=$2

# Echo input variables
echo "OMOP_FOLDER=$OMOP_FOLDER"
echo "PATIENT_SPLITS_FOLDER=$PATIENT_SPLITS_FOLDER"

# Helper function to check and create directories
create_directory_if_not_exists() {
    if [ ! -d "$1" ]; then
        echo "Creating $1"
        mkdir -p "$1"
    fi
}

#!/bin/bash

# Generate CAD CABG Cohort
echo "Generating cad_cabg"
create_directory_if_not_exists "$OMOP_FOLDER/cohorts/cad_cabg"

python -u -m cehrbert_data.prediction_cohorts.cad_cabg_cohort \
    -c cad_cabg_bow \
    -i "$OMOP_FOLDER" \
    -o "$OMOP_FOLDER/cohorts/cad_cabg/" \
    -dl 1985-01-01 -du 2023-12-31 \
    -l 18 -u 100 -ow 360 -ps 0 -pw 360 -hw 30 -f \
    --att_type cehr_bert \
    --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv

# Run Predictions on CAD CABG
echo "Run predictions on cad_cabg"
create_directory_if_not_exists "$OMOP_FOLDER/evaluation_gpt/cad_cabg"

if [ -n "$PATIENT_SPLITS_FOLDER" ]; then
    python -m cehrbert.evaluations.evaluation \
        -a baseline_model \
        -d "$OMOP_FOLDER/cohorts/cad_cabg/cad_cabg_bow/" \
        -ef "$OMOP_FOLDER/evaluation_gpt/cad_cabg/" \
        --patient_splits_folder "$PATIENT_SPLITS_FOLDER"
else
    python -m cehrbert.evaluations.evaluation \
        -a baseline_model \
        -d "$OMOP_FOLDER/cohorts/cad_cabg/cad_cabg_bow/" \
        -ef "$OMOP_FOLDER/evaluation_gpt/cad_cabg/"
fi

# Generate HF Readmission
echo "Generating hf_readmission"
create_directory_if_not_exists "$OMOP_FOLDER/cohorts/hf_readmission"

python -u -m cehrbert_data.prediction_cohorts.hf_readmission \
  -c hf_readmission_bow \
  -i "$OMOP_FOLDER" \
  -o "$OMOP_FOLDER/cohorts/hf_readmission" \
  -dl 1985-01-01 -du 2023-12-31 -l 18 -u 100 -ow 360 -ps 1 -pw 30 -f \
  --att_type cehr_bert \
  --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv \
  --is_remove_index_prediction_starts

# Run predictions on HF Readmission
echo "Run predictions on hf_readmission"
create_directory_if_not_exists "$OMOP_FOLDER/evaluation_gpt/hf_readmission"

if [ -n "$PATIENT_SPLITS_FOLDER" ]; then
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/hf_readmission/hf_readmission_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/hf_readmission/" \
      --patient_splits_folder "$PATIENT_SPLITS_FOLDER"
else
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/hf_readmission/hf_readmission_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/hf_readmission/"
fi

# Generate COPD Readmission
echo "Generating copd_readmission"
create_directory_if_not_exists "$OMOP_FOLDER/cohorts/copd_readmission"

python -u -m cehrbert_data.prediction_cohorts.copd_readmission \
  -c copd_readmission_bow \
  -i "$OMOP_FOLDER" \
  -o "$OMOP_FOLDER/cohorts/copd_readmission" \
  -dl 1985-01-01 -du 2023-12-31 -l 18 -u 100 -ow 360 -ps 1 -pw 30 -f \
  --att_type cehr_bert \
  --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv \
  --is_remove_index_prediction_starts

# Run predictions on COPD Readmission
echo "Run predictions on copd_readmission"
create_directory_if_not_exists "$OMOP_FOLDER/evaluation_gpt/copd_readmission"

if [ -n "$PATIENT_SPLITS_FOLDER" ]; then
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/copd_readmission/copd_readmission_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/copd_readmission/" \
      --patient_splits_folder "$PATIENT_SPLITS_FOLDER"
else
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/copd_readmission/copd_readmission_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/copd_readmission/"
fi

# Generate Hospitalization
echo "Generating hospitalization"
create_directory_if_not_exists "$OMOP_FOLDER/cohorts/hospitalization"

python -u -m cehrbert_data.prediction_cohorts.hospitalization \
  -c hospitalization_bow \
  -i "$OMOP_FOLDER" \
  -o "$OMOP_FOLDER/cohorts/hospitalization" \
  -dl 1985-01-01 -du 2023-12-31 -l 18 -u 100 -ow 540 -hw 180 -ps 0 -pw 360 -f -iw \
  --att_type cehr_bert \
  --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv

# Run predictions on Hospitalization
echo "Run predictions on hospitalization"
create_directory_if_not_exists "$OMOP_FOLDER/evaluation_gpt/hospitalization"

if [ -n "$PATIENT_SPLITS_FOLDER" ]; then
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/hospitalization/hospitalization_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/hospitalization/" \
      --patient_splits_folder "$PATIENT_SPLITS_FOLDER"
else
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/hospitalization/hospitalization_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/hospitalization/"
fi

# Generate AFIB Ischemic Stroke
echo "Generating afib_ischemic_stroke"
create_directory_if_not_exists "$OMOP_FOLDER/cohorts/afib_ischemic_stroke"

python -u -m cehrbert_data.prediction_cohorts.afib_ischemic_stroke \
  -c afib_ischemic_stroke_bow \
  -i "$OMOP_FOLDER" \
  -o "$OMOP_FOLDER/cohorts/afib_ischemic_stroke" \
  -dl 1985-01-01 -du 2023-12-31 -l 18 -u 100 -ow 720 -ps 0 -pw 360 -hw 30 -f \
  --att_type cehr_bert \
  --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv

# Run predictions on AFIB Ischemic Stroke
echo "Run predictions on afib_ischemic_stroke"
create_directory_if_not_exists "$OMOP_FOLDER/evaluation_gpt/afib_ischemic_stroke"

if [ -n "$PATIENT_SPLITS_FOLDER" ]; then
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/afib_ischemic_stroke/afib_ischemic_stroke_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/afib_ischemic_stroke/" \
      --patient_splits_folder "$PATIENT_SPLITS_FOLDER"
else
    python -m cehrbert.evaluations.evaluation \
      -a baseline_model \
      -d "$OMOP_FOLDER/cohorts/afib_ischemic_stroke/afib_ischemic_stroke_bow/" \
      -ef "$OMOP_FOLDER/evaluation_gpt/afib_ischemic_stroke/"
fi
