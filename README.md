# CEHRGPT

[![PyPI - Version](https://img.shields.io/pypi/v/cehrgpt)](https://pypi.org/project/cehrgpt/)
![Python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)
[![tests](https://github.com/knatarajan-lab/cehrgpt/actions/workflows/tests.yaml/badge.svg)](https://github.com/knatarajan-lab/cehrgpt/actions/workflows/tests.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/knatarajan-lab/cehrgpt/blob/main/LICENSE)
[![contributors](https://img.shields.io/github/contributors/knatarajan-lab/cehrgpt.svg)](https://github.com/knatarajan-lab/cehrgpt/graphs/contributors)

CEHRGPT is a multi-task foundation model for structured electronic health records (EHR) data that supports three capabilities: feature representation, zero-shot prediction, and synthetic data generation.

## 🎯 Key Capabilities

### Feature Representation
Extract meaningful patient embeddings from sequences of medical events using **linear probing** techniques for downstream tasks such as disease prediction, patient clustering, and risk stratification.

### Zero-Shot Prediction
Generate outcome predictions directly from prompts without requiring task-specific training, enabling rapid evaluation in low-label clinical settings.

### Synthetic Data Generation
Generate comprehensive patient profiles including demographics, medical history, treatment courses, and outcomes while implementing advanced privacy-preserving techniques to ensure generated data contains no identifiable information.
The platform is fully compatible with the OMOP Common Data Model for seamless integration with existing healthcare systems.
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

### Step 1: Generate Pre-training Data from OMOP

Generate the training data following the [Data Generation Instruction](./data_generation.md).

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
  --warmup_ratio 0.01 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.0002 \
  --use_early_stopping \
  --load_best_model_at_end true \
  --early_stopping_threshold 0.001
```

> **Tip**: Increase `max_position_embeddings` for longer context windows based on your use case.

## 🎯 Feature Representation

CEHR-GPT enables extraction of meaningful patient embeddings from medical event sequences using **linear probing** techniques for downstream prediction tasks. The feature representation pipeline includes label generation, patient sequence extraction, and linear regression model training on the extracted representations.

For detailed instructions including cohort creation, patient feature extraction, and linear probing evaluation, please follow the [Feature Representation Guide](./feature_representation.md).

## 🔮 Zero-Shot Prediction

CEHR-GPT can generate outcome predictions directly from clinical prompts without requiring task-specific training, making it ideal for rapid evaluation in low-label clinical settings. The zero-shot prediction capability performs time-to-event analysis by processing patient sequences and generating risk predictions based on learned medical patterns.

For complete setup instructions including label generation, sequence preparation, and prediction execution, please follow the [Zero-Shot Prediction Guide](./zero_shot_prediction.md).

## 🧬 Synthetic Data Generation

CEHR-GPT generates comprehensive synthetic patient profiles including demographics, medical history, treatment courses, and outcomes while implementing advanced privacy-preserving techniques. The synthetic data maintains statistical fidelity to real patient populations without containing identifiable information, and outputs are fully compatible with the OMOP Common Data Model.

For step-by-step instructions on generating synthetic sequences and converting them to OMOP format, please follow the [Synthetic Data Generation Guide](./synthetic_data_generation.md).

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
