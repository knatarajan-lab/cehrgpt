# CEHR-GPT Zero-Shot Prediction

This guide covers performing zero-shot predictions for time-to-event analysis using CEHR-GPT without requiring task-specific training, enabling rapid evaluation in low-label clinical settings.

## Prerequisites

Ensure you have:

1. **Trained CEHR-GPT Model**: Pre-trained model available at `$CEHR_GPT_MODEL_DIR`
2. **Processed Patient Data**: Test sequences prepared from feature extraction pipeline
3. **Environment Setup**: Required environment variables configured

## Required Environment Variables

Set up the necessary directory paths:

```bash
# CEHR-GPT installation directory (auto-detect from git repository)
export CEHRGPT_HOME=$(git rev-parse --show-toplevel)

# Data directories
export CEHR_GPT_DATA_DIR="/path/to/processed/data"
```

## Step 3: Zero-Shot Time-to-Event Prediction

Perform zero-shot predictions for time-to-event analysis using the trained model:

```bash
python -m cehrgpt.time_to_event.time_to_event_prediction \
  --batch_size 8 --context_window 4096 --sampling_strategy TopPStrategy --top_p 1.0 \
  --dataset_folder $CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences/hf_readmission/test \
  --num_return_sequences 50 \
  --task_config $CEHRGPT_HOME/src/cehrgpt/time_to_event/config/30_day_readmission.yaml
```

### Parameter Details

- `--batch_size 8`: Number of patient sequences processed simultaneously
- `--context_window 4096`: Maximum sequence length for model input
- `--sampling_strategy TopPStrategy`: Sampling method for prediction generation
- `--top_p 1.0`: Nucleus sampling parameter for prediction diversity
- `--dataset_folder`: Directory containing test patient sequences
- `--num_return_sequences 50`: Number of prediction sequences to generate per patient
- `--task_config`: YAML configuration file defining the prediction task

### Task Configuration

The task configuration file (`30_day_readmission.yaml`) defines:

- **Prediction Window**: Time horizon for event prediction (30 days)
- **Event Definition**: Clinical criteria for readmission events
- **Output Format**: Structure of prediction results
- **Evaluation Metrics**: Performance measures for zero-shot predictions
