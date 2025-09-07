# CEHRGPT

[![PyPI - Version](https://img.shields.io/pypi/v/cehrgpt)](https://pypi.org/project/cehrgpt/)
![Python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)
[![tests](https://github.com/knatarajan-lab/cehrgpt/actions/workflows/tests.yaml/badge.svg)](https://github.com/knatarajan-lab/cehrgpt/actions/workflows/tests.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/knatarajan-lab/cehrgpt/blob/main/LICENSE)
[![contributors](https://img.shields.io/github/contributors/knatarajan-lab/cehrgpt.svg)](https://github.com/knatarajan-lab/cehrgpt/graphs/contributors)

CEHRGPT is a comprehensive foundation model for structured electronic health records (EHR) data that unifies three essential capabilities within a single architecture: feature representation, zero-shot prediction, and synthetic data generation.

## 🎯 Key Capabilities

### Feature Representation
Extract meaningful patient embeddings from sequences of medical events for downstream tasks such as disease prediction, patient clustering, and risk stratification.

### Zero-Shot Prediction
Generate outcome predictions directly from natural language prompts without requiring task-specific training, enabling rapid evaluation in low-label clinical settings.

### Synthetic Data Generation
- **Comprehensive Patient Profiles**: Generate complete patient data including demographics, medical history, treatment courses, and outcomes
- **Privacy-Preserving**: Implements advanced techniques to ensure generated data contains no identifiable information
- **OMOP Compatibility**: Fully compatible with the OMOP Common Data Model for seamless integration with existing healthcare systems
- **Extensible Architecture**: Designed to adapt to new datasets and different EHR systems

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/knatarajan-lab/cehrgpt.git
cd cehrgpt
pip install .
```

## 📋 Prerequisites

Before getting started, set up the required environment variables:

```bash
export CEHRGPT_HOME=$(git rev-parse --show-toplevel)
export OMOP_DIR=""                    # Path to your OMOP data
export CEHR_GPT_DATA_DIR=""          # Path for processed data storage
export CEHR_GPT_MODEL_DIR=""         # Path for model storage
```

Create the dataset cache directory:
```bash
mkdir $CEHR_GPT_DATA_DIR/dataset_prepared
```

## 🏗️ Model Training

### Step 1: Generate Pre-training Data

Configure Spark environment for data processing:

```bash
export SPARK_WORKER_INSTANCES="1"
export SPARK_WORKER_CORES="16"
export SPARK_EXECUTOR_CORES="2"
export SPARK_DRIVER_MEMORY="12g"
export SPARK_EXECUTOR_MEMORY="12g"
```

> **Note**: Adjust worker cores and executor memory based on your dataset size.

Generate pre-training data:

```bash
sh $CEHRGPT_HOME/scripts/create_cehrgpt_pretraining_data.sh \
  --input_folder $OMOP_DIR \
  --output_folder $CEHR_GPT_DATA_DIR \
  --start_date "1985-01-01"
```

### Step 2: Pre-train CEHR-GPT

Train the foundation model:

```bash
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner \
  --model_name_or_path $CEHR_GPT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
  --output_dir $CEHR_GPT_MODEL_DIR \
  --data_folder "$CEHR_GPT_DATA_DIR/patient_sequence/train" \
  --dataset_prepared_path "$CEHR_GPT_DATA_DIR/dataset_prepared" \
  --do_train true --seed 42 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 14 --max_position_embeddings 4096 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 16384 \
  --warmup_steps 0.01 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.0002 \
  --use_early_stopping --early_stopping_threshold 0.001
```

> **Tip**: Increase `max_position_embeddings` for longer context windows based on your use case.

## 🎯 Feature Representation

### Step 1: Generate Prediction Labels

Create heart failure readmission labels compatible with MEDS schema:

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

### Step 2: Extract Patient Features

Extract patient sequences using a 2-year observation window:

```bash
sh $CEHRGPT_HOME/scripts/extract_features_gpt.sh \
  --cohort-folder $OMOP_DIR/labels \
  --input-dir $OMOP_DIR \
  --output-dir "$CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure" \
  --observation-window 730
```

### Step 3: Run Feature Extraction

Execute CEHR-GPT feature extraction on phenotype tasks:

```bash
sh $CEHRGPT_HOME/run_cehrgpt.sh \
  --base_dir="$CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences" \
  --dataset_prepared_path="$CEHR_GPT_DATA_DIR/dataset_prepared" \
  --model_path=$CEHR_GPT_MODEL_DIR \
  --output_dir=$CEHRGPT_FEATURES_DIR \
  --preprocessing_workers=8 \
  --model_name="cehrgpt"
```

## 🔮 Zero-Shot Prediction

Perform zero-shot predictions for time-to-event analysis:

```bash
python -m cehrgpt.time_to_event.time_to_event_prediction \
  --batch_size 8 --context_window 4096 --sampling_strategy TopPStrategy --top_p 1.0 \
  --dataset_folder $CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences/hf_readmission/test \
  --num_return_sequences 50 \
  --task_config $CEHRGPT_HOME/src/cehrgpt/time_to_event/config/30_day_readmission.yaml
```

## 🧬 Synthetic Data Generation

### Generate Synthetic Sequences

Create synthetic patient sequences:

```bash
export TRANSFORMERS_VERBOSITY=info
export CUDA_VISIBLE_DEVICES="0"

python -u -m cehrgpt.generation.generate_batch_hf_gpt_sequence \
  --model_folder test_results \
  --tokenizer_folder test_results \
  --output_folder test_results \
  --num_of_patients 128 \
  --batch_size 32 \
  --buffer_size 128 \
  --context_window 1024 \
  --sampling_strategy TopPStrategy \
  --top_p 1.0 --temperature 1.0 --repetition_penalty 1.0 \
  --epsilon_cutoff 0.00 \
  --demographic_data_path sample_data/pretrain
```

### Convert to OMOP Format

Transform synthetic sequences back to OMOP format:

```bash
# Set up OMOP vocabulary path
export OMOP_VOCAB_DIR=""

# Configure Spark environment
export SPARK_WORKER_INSTANCES="1"
export SPARK_WORKER_CORES="8"
export SPARK_EXECUTOR_CORES="2"
export SPARK_DRIVER_MEMORY="2g"
export SPARK_EXECUTOR_MEMORY="2g"

# Execute conversion pipeline
sh scripts/omop_pipeline.sh \
  test_results/top_p10000/generated_sequences/ \
  test_results/top_p10000/restored_omop/ \
  $OMOP_VOCAB_DIR
```

## 📊 MEDS Support

CEHR-GPT supports the Medical Event Data Standard (MEDS) format for enhanced interoperability.

### Prerequisites

Configure MEDS-specific environment variables:

```bash
export CEHR_GPT_MODEL_DIR=""    # CEHR-GPT model directory
export MEDS_DIR=""              # MEDS data directory
export MEDS_READER_DIR=""       # MEDS reader output directory
```

### Step 1: Create MIMIC MEDS Data

Transform MIMIC files to MEDS format following the [MEDS_transforms](https://github.com/mmcdermott/MEDS_transforms/) repository instructions.

### Step 2: Prepare MEDS Reader

Convert MEDS data for CEHR-GPT compatibility:

```bash
meds_reader_convert $MEDS_DIR $MEDS_READER_DIR --num_threads 10
```

### Step 3: Pre-train with MEDS Data

Execute pre-training using MEDS format:

```bash
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner \
  --model_name_or_path $CEHR_GPT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
  --output_dir $CEHR_GPT_MODEL_DIR \
  --data_folder $MEDS_READER_DIR \
  --dataset_prepared_path "$CEHR_GPT_MODEL_DIR/dataset_prepared" \
  --do_train true --seed 42 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 14 --max_position_embeddings 8192 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 16384 \
  --warmup_steps 500 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.0002 \
  --use_early_stopping --early_stopping_threshold 0.001 \
  --is_data_in_meds --inpatient_att_function_type day \
  --att_function_type day --include_inpatient_hour_token \
  --include_auxiliary_token --include_demographic_prompt \
  --meds_to_cehrbert_conversion_type "MedsToBertMimic4"
```

### Step 4: Generate MEDS Trajectories

#### Environment Setup

Configure trajectory generation environment:

```bash
export MEDS_LABEL_COHORT_DIR=""     # Cohort labels directory (parquet files)
export MEDS_TRAJECTORY_DIR=""       # Trajectory output directory
```

#### Generate Synthetic Trajectories

Create patient trajectories with the trained model:

```bash
python -u -m cehrgpt.generation.cehrgpt_conditional_generation \
  --cohort_folder $MEDS_LABEL_COHORT_DIR \
  --data_folder $MEDS_READER_DIR \
  --dataset_prepared_path "$CEHR_GPT_MODEL_DIR/dataset_prepared" \
  --model_name_or_path $CEHR_GPT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
  --output_dir $MEDS_TRAJECTORY_DIR \
  --per_device_eval_batch_size 16 \
  --num_of_trajectories_per_sample 2 \
  --generation_input_length 4096 \
  --generation_max_new_tokens 4096 \
  --is_data_in_meds \
  --att_function_type day --inpatient_att_function_type day \
  --meds_to_cehrbert_conversion_type MedsToBertMimic4 \
  --include_auxiliary_token --include_demographic_prompt \
  --include_inpatient_hour_token
```

> **Important**: Ensure `generation_input_length` + `generation_max_new_tokens` ≤ `max_position_embeddings` (8192).

#### Parameter Reference

- `generation_input_length`: Input context length for generation
- `generation_max_new_tokens`: Maximum new tokens to generate
- `num_of_trajectories_per_sample`: Number of trajectories per patient sample

## 📖 Citation

If you use CEHRGPT in your research, please cite:

```bibtex
@article{cehrgpt2024,
  title={CEHRGPT: Synthetic Data Generation for Electronic Health Records},
  author={Natarajan, K and others},
  journal={arXiv preprint arXiv:2402.04400},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
