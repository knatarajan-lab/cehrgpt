#!/bin/sh

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --base_dir=DIR                 Base directory containing cohorts (required)"
    echo "  --dataset_prepared_path=PATH   Path to prepared dataset (required)"
    echo "  --model_path=PATH              Path to pre-trained model and tokenizer (required)"
    echo "  --preprocessing_workers=NUM    Number of preprocessing workers (required)"
    echo "  --batch_size=NUM               Batch size for evaluation (required)"
    echo "  --output_dir=DIR               Output directory for results (required)"
    echo "  --model_name=NAME              Name for the model output directory (default: cehrgpt_model)"
    echo "  --max_tokens_per_batch=NUM     Maximum tokens per batch (default: 16384)"
    echo "  --torch_type=TYPE              Torch data type (default: float32)"
    echo "  --disable_sample_packing       Disable sample packing (enabled by default)"
    echo ""
    echo "Example:"
    echo "  $0 --base_dir=/path/to/cohorts --dataset_prepared_path=/path/to/dataset_prepared \\"
    echo "     --model_path=/path/to/model --preprocessing_workers=16 --batch_size=64 \\"
    echo "     --output_dir=/path/to/outputs --model_name=my_model --torch_type=float16"
    exit 1
}

# Default values
MODEL_NAME="cehrgpt_model"
MAX_TOKENS_PER_BATCH="16384"
TORCH_TYPE="bfloat16"
DISABLE_SAMPLE_PACKING="false"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --base_dir=*)
            BASE_DIR="${arg#*=}"
            ;;
        --dataset_prepared_path=*)
            DATASET_PREPARED_PATH="${arg#*=}"
            ;;
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --preprocessing_workers=*)
            PREPROCESSING_WORKERS="${arg#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --model_name=*)
            MODEL_NAME="${arg#*=}"
            ;;
        --max_tokens_per_batch=*)
            MAX_TOKENS_PER_BATCH="${arg#*=}"
            ;;
        --torch_type=*)
            TORCH_TYPE="${arg#*=}"
            ;;
        --disable_sample_packing)
            DISABLE_SAMPLE_PACKING="true"
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Error: Unknown option: $arg"
            usage
            ;;
    esac
done

# Check for required arguments
if [ -z "$BASE_DIR" ] || [ -z "$DATASET_PREPARED_PATH" ] || [ -z "$MODEL_PATH" ] || [ -z "$PREPROCESSING_WORKERS" ] || [ -z "$BATCH_SIZE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Validate arguments
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    exit 1
fi

if [ ! -d "$DATASET_PREPARED_PATH" ]; then
    echo "Error: Dataset prepared path does not exist: $DATASET_PREPARED_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if preprocessing workers is a number
if ! [ "$PREPROCESSING_WORKERS" -eq "$PREPROCESSING_WORKERS" ] 2>/dev/null; then
    echo "Error: Preprocessing workers must be a number: $PREPROCESSING_WORKERS"
    exit 1
fi

# Check if batch size is a number
if ! [ "$BATCH_SIZE" -eq "$BATCH_SIZE" ] 2>/dev/null; then
    echo "Error: Batch size must be a number: $BATCH_SIZE"
    exit 1
fi

# Check if max tokens per batch is a number
if ! [ "$MAX_TOKENS_PER_BATCH" -eq "$MAX_TOKENS_PER_BATCH" ] 2>/dev/null; then
    echo "Error: Max tokens per batch must be a number: $MAX_TOKENS_PER_BATCH"
    exit 1
fi

# Validate torch_type (common PyTorch data types)
case "$TORCH_TYPE" in
    float16|float32|float64|bfloat16|int8|int16|int32|int64)
        ;;
    *)
        echo "Error: Invalid torch_type. Supported types: float16, float32, float64, bfloat16, int8, int16, int32, int64"
        exit 1
        ;;
esac

# Validate disable_sample_packing is boolean-like
if [ "$DISABLE_SAMPLE_PACKING" != "true" ] && [ "$DISABLE_SAMPLE_PACKING" != "false" ]; then
    echo "Error: disable_sample_packing must be 'true' or 'false': $DISABLE_SAMPLE_PACKING"
    exit 1
fi

# Log file setup
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

# Log function
log() {
    message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$MAIN_LOG"
}

# Main execution
log "Starting feature extraction and model training process"
log "Configuration:"
log "  --base_dir=$BASE_DIR"
log "  --dataset_prepared_path=$DATASET_PREPARED_PATH"
log "  --model_path=$MODEL_PATH"
log "  --preprocessing_workers=$PREPROCESSING_WORKERS"
log "  --batch_size=$BATCH_SIZE"
log "  --output_dir=$OUTPUT_DIR"
log "  --model_name=$MODEL_NAME"
log "  --max_tokens_per_batch=$MAX_TOKENS_PER_BATCH"
log "  --torch_type=$TORCH_TYPE"
log "  --disable_sample_packing=$DISABLE_SAMPLE_PACKING"

# Find valid cohorts and write to a temp file
TEMP_COHORT_LIST="$LOG_DIR/cohort_list_${TIMESTAMP}.txt"
> "$TEMP_COHORT_LIST" # Clear the file

# Find all valid cohorts (directories with train and test subdirectories)
for cohort_dir in "$BASE_DIR"/*; do
    if [ -d "$cohort_dir" ] && [ -d "$cohort_dir/train" ] && [ -d "$cohort_dir/test" ]; then
        cohort_name=$(basename "$cohort_dir")
        echo "$cohort_name" >> "$TEMP_COHORT_LIST"
    fi
done

# Check if any valid cohorts were found
if [ ! -s "$TEMP_COHORT_LIST" ]; then
    log "ERROR: No valid cohorts found in $BASE_DIR"
    rm -f "$TEMP_COHORT_LIST"
    exit 1
fi

# Display all cohorts that will be processed
cohort_count=$(wc -l < "$TEMP_COHORT_LIST")
log "Found $cohort_count cohorts to process:"
while read -r cohort; do
    log "- $cohort"
done < "$TEMP_COHORT_LIST"

# Process each cohort sequentially
while read -r cohort_name; do
    cohort_dir="$OUTPUT_DIR/$cohort_name"
    output_dir="$cohort_dir/$MODEL_NAME"
    metrics_file="$output_dir/logistic/metrics.json"

    # Check if metrics file exists to determine if cohort has been processed
    if [ -f "$metrics_file" ]; then
        log "Skipping cohort $cohort_name: metrics file already exists at $metrics_file"
        continue
    fi

    log "===================================================="
    log "Processing cohort: $cohort_name"
    log "===================================================="

    cohort_log="$LOG_DIR/${cohort_name}_${TIMESTAMP}.log"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Prepare command for feature extraction
    FEATURE_CMD="python -u -m cehrgpt.tools.linear_prob.compute_cehrgpt_features \
        --data_folder \"$BASE_DIR/$cohort_name/train/\" \
        --test_data_folder \"$BASE_DIR/$cohort_name/test/\" \
        --dataset_prepared_path \"$DATASET_PREPARED_PATH\" \
        --model_name_or_path \"$MODEL_PATH\" \
        --tokenizer_name_or_path \"$MODEL_PATH\" \
        --output_dir \"$output_dir\" \
        --preprocessing_num_workers \"$PREPROCESSING_WORKERS\" \
        --per_device_eval_batch_size \"$BATCH_SIZE\" \
        --max_tokens_per_batch \"$MAX_TOKENS_PER_BATCH\" \
        --torch_dtype \"$TORCH_TYPE\""

    # Add sample packing flag if not disabled
    if [ "$DISABLE_SAMPLE_PACKING" = "false" ]; then
        FEATURE_CMD="$FEATURE_CMD --sample_packing"
    fi

    # Step 1: Feature extraction
    log "Starting feature extraction for $cohort_name..."
    log "Command: $FEATURE_CMD"

    eval "$FEATURE_CMD > \"$cohort_log\" 2>&1"

    feature_extraction_status=$?
    if [ $feature_extraction_status -ne 0 ]; then
        log "ERROR: Feature extraction failed for $cohort_name. Check $cohort_log for details."
        continue
    fi

    # Step 2: Model training
    log "Starting model training for $cohort_name..."
    log "Command: python -u -m cehrgpt.tools.linear_prob.train_with_cehrgpt_features --features_data_dir $output_dir --output_dir $output_dir"

    python -u -m cehrgpt.tools.linear_prob.train_with_cehrgpt_features \
        --features_data_dir "$output_dir" \
        --output_dir "$output_dir" \
        >> "$cohort_log" 2>&1

    echo "Running meds-evaluation for logistic regression for $TASK_NAME..."
    meds-evaluation-cli predictions_path="$output_dir/logistic/test_predictions" \
      output_dir="$output_dir/logistic/"

        # Check if the second command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Running meds-evaluation failed for logistic regression for task $TASK_NAME"
    fi

    model_training_status=$?
    if [ $model_training_status -ne 0 ]; then
        log "ERROR: Model training failed for $cohort_name. Check $cohort_log for details."
        continue
    fi

    log "Successfully completed processing for $cohort_name"
done < "$TEMP_COHORT_LIST"

# Clean up
rm -f "$TEMP_COHORT_LIST"

log "All processing completed"
