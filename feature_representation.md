# CEHR-GPT Feature Representation using Linear Probing

This guide covers the process of extracting meaningful patient embeddings from healthcare sequences using **linear probing** techniques for downstream prediction tasks such as disease prediction, patient clustering, and risk stratification.

## Prerequisites

Ensure you have:

1. **Trained CEHR-GPT Model**: Pre-trained model available at `$CEHR_GPT_MODEL_DIR`
2. **OMOP Data**: Healthcare data processed and ready for feature extraction
3. **Environment Setup**: Required environment variables configured

## Required Environment Variables

Set up the necessary directory paths:

```bash
# CEHR-GPT installation directory (auto-detect from git repository)
export CEHRGPT_HOME=$(git rev-parse --show-toplevel)
export CEHR_GPT_MODEL_DIR="/path/to/trained/model"

# Data directories
export OMOP_DIR="/path/to/omop/data"
export CEHR_GPT_DATA_DIR="/path/to/processed/data"
export CEHRGPT_FEATURES_DIR="/path/to/extracted/features"
```

## Step 1: Generate Prediction Labels

Create heart failure readmission labels compatible with MEDS schema for downstream prediction tasks:

```bash
python -u -m cehrbert_data.prediction_cohorts.hf_readmission \
   -c hf_readmission -i $OMOP_DIR -o $OMOP_DIR/labels \
   -dl 1985-01-01 -du 2023-12-31 \
   -l 18 -u 100 -ow 730 -ps 1 -pw 30 \
   --is_new_patient_representation \
   --should_construct_artificial_visits \
   --include_concept_list \
   --is_remove_index_prediction_starts \
   --meds_format \
   --exclude_features
```

### Parameter Explanation

- `-c hf_readmission`: Cohort name for heart failure readmission prediction
- `-i $OMOP_DIR`: Input directory containing OMOP data
- `-o $OMOP_DIR/labels`: Output directory for generated labels
- `-dl/-du`: Date range for patient inclusion (1985-2023)
- `-l 18 -u 100`: Age limits (18-100 years)
- `-ow 730`: Observation window in days (2 years)
- `-ps 1 -pw 30`: Prediction start (1 day) and window (30 days)
- `--is_remove_index_prediction_starts`: Remove cases where outcome events occur before prediction start date
- `--include_concept_list`: Include only concepts that are allowed in the model vocabulary
- `--meds_format`: Output in MEDS-compatible format



## Step 2: Extract Patient Features

Extract patient sequences using a 2-year observation window, focusing on key clinical events:

```bash
sh $CEHRGPT_HOME/scripts/extract_features_gpt.sh \
  --cohort-folder $OMOP_DIR/labels \
  --input-dir $OMOP_DIR \
  --output-dir "$CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure" \
  --observation-window 730
```
> **Tip**: This step requires pyspark, and please refer to **Spark Environment**: Configured Apache Spark (see [Spark Setup README](./spark_setup.md))

### Key Parameters

- `--cohort-folder`: Directory containing prediction labels
- `--input-dir`: Source OMOP data directory
- `--output-dir`: Output directory for extracted sequences
- `--patient-splits-folder`: Pre-defined train/validation/test splits
- `--ehr-tables`: Clinical tables to include in feature extraction
- `--observation-window`: Observation period in days (730 = 2 years)



## Step 3: Run Feature Extraction and Linear Probing

Execute CEHR-GPT feature extraction and train a linear regression model on the extracted patient representations:

```bash
sh $CEHRGPT_HOME/run_cehrgpt.sh \
  --base_dir="$CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences" \
  --dataset_prepared_path="$CEHR_GPT_DATA_DIR/dataset_prepared" \
  --model_path=$CEHR_GPT_MODEL_DIR \
  --output_dir=$CEHRGPT_FEATURES_DIR \
  --preprocessing_workers=8 \
  --model_name="cehrgpt"
```

This step performs both feature extraction from patient sequences and trains a linear regression model on the extracted patient representations for downstream prediction tasks.

### Parameter Details

- `--base_dir`: Directory containing prepared patient sequences
- `--dataset_prepared_path`: Path for preprocessed datasets
- `--model_path`: Location of trained CEHR-GPT model
- `--output_dir`: Output directory for extracted features and embeddings
- `--preprocessing_workers`: Number of parallel workers for data preprocessing
- `--model_name`: Model identifier for feature extraction
