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
```

## Step 1: Generate Synthetic Sequences

Create synthetic patient sequences using the trained CEHR-GPT model:

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

### Parameter Details

- `--model_folder`: Directory containing the trained CEHR-GPT model
- `--tokenizer_folder`: Directory containing the model tokenizer
- `--output_folder`: Directory where synthetic sequences will be saved
- `--num_of_patients`: Number of synthetic patients to generate (128)
- `--batch_size`: Batch size for generation process (32)
- `--buffer_size`: Buffer size for sequence generation (128)
- `--context_window`: Maximum sequence length for generation (1024)
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
# OMOP vocabulary directory for conversion
export OMOP_VOCAB_DIR="/path/to/omop/"
# Execute conversion pipeline
sh scripts/omop_pipeline.sh \
  test_results/top_p10000/generated_sequences/ \
  test_results/top_p10000/restored_omop/ \
  $OMOP_VOCAB_DIR
```

### Conversion Pipeline Parameters

- **Input Directory**: `test_results/top_p10000/generated_sequences/` - Generated synthetic sequences
- **Output Directory**: `test_results/top_p10000/restored_omop/` - OMOP-formatted output
- **Vocabulary Directory**: `$OMOP_VOCAB_DIR` - OMOP vocabulary for concept mapping

## Privacy and Compliance

The synthetic data generation implements advanced privacy-preserving techniques:

- **De-identification**: No real patient identifiers in generated sequences
- **Statistical Privacy**: Maintains aggregate population statistics without individual privacy risks
- **OMOP Compatibility**: Fully compatible with OMOP Common Data Model standards
- **Extensible Architecture**: Designed to adapt to new datasets and different EHR systems
