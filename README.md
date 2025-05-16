# CEHRGPT

[![PyPI - Version](https://img.shields.io/pypi/v/cehrgpt)](https://pypi.org/project/cehrgpt/)
![Python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)
[![tests](https://github.com/knatarajan-lab/cehrgpt/actions/workflows/tests.yaml/badge.svg)](https://github.com/knatarajan-lab/cehrgpt/actions/workflows/tests.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/knatarajan-lab/cehrgpt-public/blob/main/LICENSE)
[![contributors](https://img.shields.io/github/contributors/knatarajan-lab/cehrgpt.svg)](https://github.com/knatarajan-lab/cehrgpt/graphs/contributors)

## Description
CEHRGPT is a synthetic data generation model developed to handle structured electronic health records (EHR) with enhanced privacy and reliability. It leverages state-of-the-art natural language processing techniques to create realistic, anonymized patient data that can be used for research and development without compromising patient privacy.

## Features
- **Synthetic Patient Data Generation**: Generates comprehensive patient profiles including demographics, medical history, treatment courses, and outcomes.
- **Privacy-Preserving**: Implements techniques to ensure the generated data does not reveal identifiable information.
- **Compatibility with OMOP**: Fully compatible with the OMOP common data model, allowing seamless integration with existing healthcare data systems.
- **Extensible**: Designed to be adaptable to new datasets and different EHR systems.

## Installation
To install CEHRGPT, clone this repository and install the required dependencies.

```bash
git clone https://github.com/knatarajan-lab/cehrgpt.git
cd cehrgpt
pip install .
```

## Pretrain
Pretrain cehrgpt using the Hugging Face trainer, the parameters can be found in the sample configuration yaml
```bash
mkdir test_results
# This is NOT required when streaming is set to true
mkdir test_dataset_prepared
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner sample_configs/cehrgpt_pretrain_sample_config.yaml
```

## Generate synthetic sequences
Generate synthetic sequences using the trained model
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

## Convert synthetic sequences to OMOP
```bash
# omop converter requires the OHDSI vocabulary
export OMOP_VOCAB_DIR = ""
# the omop derived tables need to be built using pyspark
export SPARK_WORKER_INSTANCES="1"
export SPARK_WORKER_CORES="8"
export SPARK_EXECUTOR_CORES="2"
export SPARK_DRIVER_MEMORY="2g"
export SPARK_EXECUTOR_MEMORY="2g"

# Convert the sequences, create the omop derived tables
sh scripts/omop_pipeline.sh \
  test_results/top_p10000/generated_sequences/ \
  test_results/top_p10000/restored_omop/ \
  $OMOP_VOCAB_DIR
```

## Citation
```
@article{cehrgpt2024,
  title={CEHRGPT: Synthetic Data Generation for Electronic Health Records},
  author={Natarajan, K and others},
  journal={arXiv preprint arXiv:2402.04400},
  year={2024}
}
