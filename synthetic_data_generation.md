# CEHR-GPT Synthetic Data Generation

This guide covers generating synthetic patient data using CEHR-GPT, including comprehensive patient profiles with demographics, medical history, treatment courses, and outcomes while implementing privacy-preserving techniques to ensure generated data contains no identifiable information.

## Prerequisites

Ensure you have:

1. **Trained CEHR-GPT Model**: Pre-trained model and tokenizer available
2. **GPU Access**: CUDA-compatible GPU for efficient generation
3. **Spark Environment**: Configured Apache Spark for OMOP conversion (see [Spark Setup README](./spark_setup.md))

## Required Environment Variables

Set up the necessary directory paths:

```bash
# CEHR-GPT installation directory (auto-detect from git repository)
export CEHRGPT_HOME=$(git rev-parse --show-toplevel)
export OMOP_VOCAB_DIR="/path/to/source/omop/folder"
export CEHR_GPT_DATA_DIR="/path/to/training/data"
export CEHR_GPT_MODEL_DIR="/path/to/trained/model"
export SYNTHETIC_DATA_OUTPUT_DIR="/path/to/generated/synthetic/data"
export PRIVACY_ANALYSIS_OUTPUT_DIR="$SYNTHETIC_DATA_OUTPUT_DIR/privacy"
```

## Step 1: Generate Synthetic Sequences

Create synthetic patient sequences using the trained CEHR-GPT model:

```bash
export TRANSFORMERS_VERBOSITY=info
export CUDA_VISIBLE_DEVICES="0"

python -u -m cehrgpt.generation.generate_batch_hf_gpt_sequence \
  --model_folder $CEHR_GPT_MODEL_DIR \
  --tokenizer_folder $CEHR_GPT_MODEL_DIR \
  --output_folder $SYNTHETIC_DATA_OUTPUT_DIR \
  --num_of_patients 128 \
  --batch_size 16 \
  --buffer_size 128 \
  --context_window 4096 \
  --sampling_strategy TopPStrategy \
  --top_p 1.0 --temperature 1.0 --repetition_penalty 1.0 \
  --epsilon_cutoff 0.00 \
  --demographic_data_path $CEHR_GPT_DATA_DIR/patient_sequence/train
```
> **Tip**: The synthetic data generation script is designed for single GPU usage. To leverage multiple GPUs, run separate instances of the script on each GPU with different `CUDA_VISIBLE_DEVICES` settings for parallel processing.

### Parameter Details

- `--model_folder`: Directory containing the trained CEHR-GPT model
- `--tokenizer_folder`: Directory containing the model tokenizer
- `--output_folder`: Directory where synthetic sequences will be saved
- `--num_of_patients`: Number of synthetic patients to generate (128)
- `--batch_size`: Batch size for generation process (16)
- `--buffer_size`: Buffer size for sequence generation (128)
- `--context_window`: Maximum sequence length for generation (4096)
- `--sampling_strategy`: Sampling method for sequence generation (TopPStrategy)
- `--top_p`: Nucleus sampling parameter for diversity control (1.0)
- `--temperature`: Temperature for sampling randomness (1.0)
- `--repetition_penalty`: Penalty for repeated tokens (1.0)
- `--epsilon_cutoff`: Cutoff threshold for token filtering (0.00)
- `--demographic_data_path`: Path to demographic data templates

## Step 2: Convert to OMOP Format

Transform synthetic sequences back to OMOP Common Data Model format for seamless integration with existing healthcare systems:
> **Tips**: This step requires spark, please refer to **Spark Environment**: Configured Apache Spark (see [Spark Setup README](./spark_setup.md))
```bash
# Execute conversion pipeline
sh scripts/omop_pipeline.sh \
  --patient-sequence-folder=$SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/ \
  --omop-folder=$SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/restored_omop/ \
  --source-omop-folder=$OMOP_VOCAB_DIR \
  --cpu-cores=10
```

### Conversion Pipeline Parameters

- **Input Directory**: `$SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/` - Generated synthetic sequences
- **Output Directory**: `$SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/restored_omop/` - OMOP-formatted output
- **Vocabulary Directory**: `$OMOP_VOCAB_DIR` - OMOP vocabulary for concept mapping

## Step 3: Upload OMOP tables to SQL Server

You can upload the OMOP tables to a SQL server database, click on [credential sample example](sample_configs/credential_file_sample.ini) to see how the credential file is constructed.
```bash
# Set up the environment variable
export CREDENTIAL_PATH="/path/to/credential/path"
# Run the python script to upload OMOP tables
python -u -m cehrgpt.tools.upload_omop \
  --credential_path $CREDENTIAL_PATH\
  --input_folder $SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/restored_omop/
```
> **Tips**: This step requires spark, please refer to **Spark Environment**: Configured Apache Spark (see [Spark Setup README](./spark_setup.md)).
> Currently, `upload_omop.py` only supports SQL Server, for other databases, you need to adjust the script.

## Privacy Analyses

### 1. Attribute Inference Analysis
```bash
mkdir -p $SYNTHETIC_DATA_OUTPUT_DIR/aia;
python -u -m cehrgpt.analysis.privacy.attribute_inference \
--training_data_folder $CEHR_GPT_DATA_DIR/patient_sequence/train \
--output_folder $PRIVACY_ANALYSIS_OUTPUT_DIR/aia \
--synthetic_data_folder $SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/ \
--tokenizer_path $CEHR_GPT_MODEL_DIR \
--attribute_config analysis/privacy/attribute_inference_config.yml \
--n_iterations 10 --num_of_samples 10000
```

### 2. Membership Inference Analysis
```bash
mkdir -p $SYNTHETIC_DATA_OUTPUT_DIR/mia;
python -u -m cehrgpt.analysis.privacy.member_inference \
--training_data_folder  $CEHR_GPT_DATA_DIR/patient_sequence/train \
--evaluation_data_folder  $CEHR_GPT_DATA_DIR/patient_sequence/test \
--output_folder $PRIVACY_ANALYSIS_OUTPUT_DIR/mia \
--synthetic_data_folder $SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/ \
--tokenizer_path $CEHR_GPT_MODEL_DIR \
--n_iterations 10 --num_of_samples 10000
```
### 3. Nearest Neighbor Inference Analysis
```bash
mkdir -p $SYNTHETIC_DATA_OUTPUT_DIR/nearest_neighbor_inference;
python -u -m cehrgpt.analysis.privacy.nearest_neighbor_inference \
--training_data_folder $CEHR_GPT_DATA_DIR/patient_sequence/train \
--evaluation_data_folder $CEHR_GPT_DATA_DIR/patient_sequence/test \
--metrics_folder $SYNTHETIC_DATA_OUTPUT_DIR/nearest_neighbor_inference \
--synthetic_data_folder $SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/  \
--concept_tokenizer_path $CEHR_GPT_MODEL_DIR \
--n_iterations 10 --num_of_samples 10000
```
### 4. Re-identification Risk Inference Analysis
```bash
mkdir $SYNTHETIC_DATA_OUTPUT_DIR/reid;
python -u -m cehrgpt.analysis.privacy.reid_inference \
--training_data_folder $CEHR_GPT_DATA_DIR/patient_sequence/train \
--evaluation_data_folder $CEHR_GPT_DATA_DIR/patient_sequence/test \
--output_folder $SYNTHETIC_DATA_OUTPUT_DIR/reid \
--synthetic_data_folder $SYNTHETIC_DATA_OUTPUT_DIR/top_p10000/generated_sequences/
```

## Privacy and Compliance

The synthetic data generation implements advanced privacy-preserving techniques:

- **De-identification**: No real patient identifiers in generated sequences
- **Statistical Privacy**: Maintains aggregate population statistics without individual privacy risks
- **OMOP Compatibility**: Fully compatible with OMOP Common Data Model standards
- **Extensible Architecture**: Designed to adapt to new datasets and different EHR systems
