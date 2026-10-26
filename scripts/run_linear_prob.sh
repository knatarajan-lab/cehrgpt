#!/bin/sh

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --base_dir=DIR                 Base directory containing cohorts (required)"
    echo "  --dataset_prepared_path=PATH   Path to prepared dataset (required)"
    echo "  --tokenized_full_dataset_path=PATH  Path to the fully tokenized dataset, used to slice out"
    echo "                                 cohort sequences instead of re-tokenizing per cohort (optional)"
    echo "  --cohort_folder=DIR            Base directory containing per-cohort subdirectories used to"
    echo "                                 resolve person_ids/labels for --tokenized_full_dataset_path"
    echo "                                 (optional, defaults to --base_dir)"
    echo "  --model_path=PATH              Path to pre-trained model (required)"
    echo "  --tokenizer_path=PATH          Path to tokenizer (optional, defaults to model_path)"
    echo "  --preprocessing_workers=NUM    Number of preprocessing workers (required)"
    echo "  --batch_size=NUM               Batch size for evaluation (required)"
    echo "  --output_dir=DIR               Output directory for results (required)"
    echo "  --model_name=NAME              Name for the model output directory (default: cehrgpt_model)"
    echo "  --max_tokens_per_batch=NUM     Maximum tokens per batch (default: 16384)"
    echo "  --torch_type=TYPE              Torch data type (default: float32)"
    echo "  --report_to=none|wandb         Log the ROC-AUC and PR-AUC of every cohort to wandb, one run per"
    echo "                                 cohort named <cohort>/<model_name> and grouped by model_name"
    echo "                                 (default: none)"
    echo "  --wandb_project=NAME           wandb project (default: \$WANDB_PROJECT, else cehrgpt-linear-probing)"
    echo "  --disable_sample_packing       Disable sample packing (enabled by default)"
    echo "  --disable_combine_global_local Disable combining global and local features (enabled by default)."
    echo "                                 The extracted features are then the global features alone"
    echo "  --feature_sets=LIST            The features to train a logistic regression on, comma separated:"
    echo "                                 concatenated (the extracted features, global then local), global (the"
    echo "                                 linear probe state) and local (the last token state), or all (default)."
    echo "                                 Global and local need the features combined (the default). The"
    echo "                                 concatenated features are written to <cohort>/<model_name>/logistic/,"
    echo "                                 the others to logistic_global/ and logistic_local/ next to it"
    echo "  --add_random_token             Enable adding random token (disabled by default)"
    echo "  --preprocessing_batch_size=NUM Batch size for dataset preprocessing (optional, default: 1000)"
    echo "  --refresh_processed_dataset    Delete and rebuild the cached prepared dataset (disabled by default)"
    echo "  --is_data_in_meds              Indicate that the data is in MEDS format (disabled by default)"
    echo "  --gpu_ids=IDS                  Extract features on several GPUs: 'all' or comma separated GPU"
    echo "                                 indices, e.g. 0,1 (optional; default: a single process)"
    echo "  --workers_per_gpu=NUM          Worker processes per GPU in --gpu_ids (default: 1). Several"
    echo "                                 workers keep one GPU busy while the others are on the CPU"
    echo "  --dataloader_num_workers=NUM   DataLoader worker processes for feature extraction (default: 0,"
    echo "                                 i.e. batches are built in the main process while the GPU waits)"
    echo "  --dataloader_prefetch_factor=NUM  Batches prefetched per worker (needs --dataloader_num_workers > 0)"
    echo "  --force_attn_implementation=IMPL  Force the attention implementation: eager, sdpa or"
    echo "                                 flash_attention_2 (optional; default: flash_attention_2 for"
    echo "                                 gpt2 when installed, sdpa for qwen2)"
    echo "  --observation_window=NUM       Observation window in days (optional, default: None)"
    echo "  --allow_missing_tokenized_persons  Skip cohort persons that are missing from the tokenized"
    echo "                                 dataset (with a warning) instead of failing; only used with"
    echo "                                 --tokenized_full_dataset_path (disabled by default)"
    echo ""
    echo "Example:"
    echo "  $0 --base_dir=/path/to/cohorts --dataset_prepared_path=/path/to/dataset_prepared \\"
    echo "     --model_path=/path/to/model --preprocessing_workers=16 --batch_size=64 \\"
    echo "     --output_dir=/path/to/outputs --model_name=my_model --torch_type=float16"
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Default values
MODEL_NAME="cehrgpt_model"
REPORT_TO="none"
MAX_TOKENS_PER_BATCH="16384"
TORCH_TYPE="bfloat16"
DISABLE_SAMPLE_PACKING="false"
DISABLE_COMBINE_GLOBAL_LOCAL="false"
FEATURE_SETS=""
ADD_RANDOM_TOKEN="false"
REFRESH_PROCESSED_DATASET="false"
IS_DATA_IN_MEDS="false"
TOKENIZED_FULL_DATASET_PATH=""
OBSERVATION_WINDOW=""
TOKENIZER_PATH=""
PREPROCESSING_BATCH_SIZE="1000"
ALLOW_MISSING_TOKENIZED_PERSONS="false"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --base_dir=*)
            BASE_DIR="${arg#*=}"
            ;;
        --dataset_prepared_path=*)
            DATASET_PREPARED_PATH="${arg#*=}"
            ;;
        --tokenized_full_dataset_path=*)
            TOKENIZED_FULL_DATASET_PATH="${arg#*=}"
            ;;
        --cohort_folder=*)
            COHORT_FOLDER="${arg#*=}"
            ;;
        --observation_window=*)
            OBSERVATION_WINDOW="${arg#*=}"
            ;;
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --tokenizer_path=*)
            TOKENIZER_PATH="${arg#*=}"
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
        --report_to=*)
            REPORT_TO="${arg#*=}"
            ;;
        --wandb_project=*)
            export WANDB_PROJECT="${arg#*=}"
            ;;
        --disable_sample_packing)
            DISABLE_SAMPLE_PACKING="true"
            ;;
        --disable_combine_global_local)
            DISABLE_COMBINE_GLOBAL_LOCAL="true"
            ;;
        --feature_sets=*)
            FEATURE_SETS="${arg#*=}"
            ;;
        --add_random_token)
            ADD_RANDOM_TOKEN="true"
            ;;
        --preprocessing_batch_size=*)
            PREPROCESSING_BATCH_SIZE="${arg#*=}"
            ;;
        --refresh_processed_dataset)
            REFRESH_PROCESSED_DATASET="true"
            ;;
        --is_data_in_meds)
            IS_DATA_IN_MEDS="true"
            ;;
        --gpu_ids=*)
            GPU_IDS="${arg#*=}"
            ;;
        --workers_per_gpu=*)
            WORKERS_PER_GPU="${arg#*=}"
            ;;
        --dataloader_num_workers=*)
            DATALOADER_NUM_WORKERS="${arg#*=}"
            ;;
        --dataloader_prefetch_factor=*)
            DATALOADER_PREFETCH_FACTOR="${arg#*=}"
            ;;
        --allow_missing_tokenized_persons)
            ALLOW_MISSING_TOKENIZED_PERSONS="true"
            ;;
        --force_attn_implementation=*)
            FORCE_ATTN_IMPLEMENTATION="${arg#*=}"
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

if [ -n "$TOKENIZED_FULL_DATASET_PATH" ] && [ ! -d "$TOKENIZED_FULL_DATASET_PATH" ]; then
    echo "Error: Tokenized full dataset path does not exist: $TOKENIZED_FULL_DATASET_PATH"
    exit 1
fi

# Default cohort_folder to base_dir if not explicitly provided
if [ -z "$COHORT_FOLDER" ]; then
    COHORT_FOLDER="$BASE_DIR"
elif [ ! -d "$COHORT_FOLDER" ]; then
    echo "Error: Cohort folder does not exist: $COHORT_FOLDER"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Default TOKENIZER_PATH to MODEL_PATH if not provided
if [ -z "$TOKENIZER_PATH" ]; then
    TOKENIZER_PATH="$MODEL_PATH"
elif [ ! -d "$TOKENIZER_PATH" ]; then
    echo "Error: Tokenizer path does not exist: $TOKENIZER_PATH"
    exit 1
fi

if [ -n "$TOKENIZED_FULL_DATASET_PATH" ] && [ ! -d "$TOKENIZED_FULL_DATASET_PATH" ]; then
    echo "Error: Tokenized full dataset path does not exist: $TOKENIZED_FULL_DATASET_PATH"
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

# Validate report_to
case "$REPORT_TO" in
    none|wandb)
        ;;
    *)
        echo "Error: Invalid report_to. Supported values: none, wandb"
        usage
        ;;
esac

# Validate torch_type (common PyTorch data types)
case "$TORCH_TYPE" in
    float16|float32|float64|bfloat16|int8|int16|int32|int64)
        ;;
    *)
        echo "Error: Invalid torch_type. Supported types: float16, float32, float64, bfloat16, int8, int16, int32, int64"
        exit 1
        ;;
esac

# Validate disable flags are boolean-like
if [ "$DISABLE_SAMPLE_PACKING" != "true" ] && [ "$DISABLE_SAMPLE_PACKING" != "false" ]; then
    echo "Error: disable_sample_packing must be 'true' or 'false': $DISABLE_SAMPLE_PACKING"
    exit 1
fi

if [ "$DISABLE_COMBINE_GLOBAL_LOCAL" != "true" ] && [ "$DISABLE_COMBINE_GLOBAL_LOCAL" != "false" ]; then
    echo "Error: disable_combine_global_local must be 'true' or 'false': $DISABLE_COMBINE_GLOBAL_LOCAL"
    exit 1
fi

# The feature sets, as the words that train_with_cehrgpt_features takes
FEATURE_SETS_ARGS=$(echo "$FEATURE_SETS" | tr ',' ' ')
for feature_set in $FEATURE_SETS_ARGS; do
    case "$feature_set" in
        all|concatenated|global|local) ;;
        *)
            echo "Error: unknown feature set '$feature_set' in --feature_sets=$FEATURE_SETS (use all, concatenated, global, local)"
            exit 1
            ;;
    esac
    if [ "$DISABLE_COMBINE_GLOBAL_LOCAL" = "true" ] && { [ "$feature_set" = "local" ] || [ "$feature_set" = "concatenated" ]; }; then
        echo "Error: the $feature_set features need the global and local features combined, which --disable_combine_global_local turns off"
        exit 1
    fi
done

# The result folders of the feature sets that were asked for, one per line, in <output_dir>. The extracted features
# (concatenated, or the global features when they were not combined) are written to logistic/, the global and the
# local features of combined features to logistic_global/ and logistic_local/. feature_info.json, which the
# extraction writes, tells whether the features were combined and whether the model has a separate global state.
required_result_dirs() {
    info="$1/feature_info.json"
    combined="false"
    if [ -f "$info" ] && grep -q '"local"' "$info"; then
        combined="true"
    fi
    if [ -z "$FEATURE_SETS_ARGS" ] || [ "$FEATURE_SETS_ARGS" = "all" ]; then
        echo "logistic"
        if [ "$combined" = "true" ]; then
            echo "logistic_global"
            # A model without a MOTOR head has the same global and local features: they are trained once
            if ! grep -q '"global_is_last_hidden_state": true' "$info"; then
                echo "logistic_local"
            fi
        fi
        return
    fi
    for feature_set in $FEATURE_SETS_ARGS; do
        case "$feature_set" in
            concatenated) echo "logistic" ;;
            global)
                if [ "$combined" = "true" ] || [ ! -f "$info" ]; then echo "logistic_global"; else echo "logistic"; fi
                ;;
            local) echo "logistic_local" ;;
            all) echo "logistic" ;;
        esac
    done
}

# Succeeds when every result folder that is required has its metrics
results_complete() {
    for result_dir in $(required_result_dirs "$1"); do
        if [ ! -f "$1/$result_dir/metrics.json" ]; then
            return 1
        fi
    done
    return 0
}

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
log "  --tokenized_full_dataset_path=$TOKENIZED_FULL_DATASET_PATH"
log "  --cohort_folder=$COHORT_FOLDER"
log "  --model_path=$MODEL_PATH"
log "  --tokenizer_path=$TOKENIZER_PATH"
log "  --preprocessing_workers=$PREPROCESSING_WORKERS"
log "  --batch_size=$BATCH_SIZE"
log "  --output_dir=$OUTPUT_DIR"
log "  --model_name=$MODEL_NAME"
log "  --max_tokens_per_batch=$MAX_TOKENS_PER_BATCH"
log "  --torch_type=$TORCH_TYPE"
log "  --report_to=$REPORT_TO"
log "  --disable_sample_packing=$DISABLE_SAMPLE_PACKING"
log "  --disable_combine_global_local=$DISABLE_COMBINE_GLOBAL_LOCAL"
log "  --feature_sets=${FEATURE_SETS:-all}"
log "  --add_random_token=$ADD_RANDOM_TOKEN"
log "  --preprocessing_batch_size=$PREPROCESSING_BATCH_SIZE"
log "  --refresh_processed_dataset=$REFRESH_PROCESSED_DATASET"
log "  --is_data_in_meds=$IS_DATA_IN_MEDS"
log "  --observation_window=$OBSERVATION_WINDOW"
log "  --allow_missing_tokenized_persons=$ALLOW_MISSING_TOKENIZED_PERSONS"

# Find valid cohorts and write to a temp file
TEMP_COHORT_LIST="$LOG_DIR/cohort_list_${TIMESTAMP}.txt"
> "$TEMP_COHORT_LIST" # Clear the file

# Find all valid cohorts.
# When --tokenized_full_dataset_path is set, cohort sequences are sliced out of the
# pre-tokenized dataset (see extract_cohort_sequences), so any subdirectory is a valid
# cohort. Otherwise require train/ and test/ subdirectories.
for cohort_dir in "$BASE_DIR"/*; do
    [ -d "$cohort_dir" ] || continue
    cohort_name=$(basename "$cohort_dir")
    # LOG_DIR lives directly under BASE_DIR, so it would otherwise be picked up as a cohort
    [ "$cohort_name" = "logs" ] && continue
    if [ -n "$TOKENIZED_FULL_DATASET_PATH" ]; then
        echo "$cohort_name" >> "$TEMP_COHORT_LIST"
    elif [ -d "$cohort_dir/train" ] && [ -d "$cohort_dir/test" ]; then
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
    # Remove trailing slash from OUTPUT_DIR if present, then add cohort_name
    cohort_dir="${OUTPUT_DIR%/}/$cohort_name"
    output_dir="$cohort_dir/$MODEL_NAME"

    # Check if the metrics of the feature sets exist to determine if cohort has already been processed
    if results_complete "$output_dir"; then
        log "Skipping cohort $cohort_name: results already exist in $output_dir"
        for result_dir in $(required_result_dirs "$output_dir"); do
            log "$result_dir: $(cat "$output_dir/$result_dir/metrics.json")"
        done
        continue
    fi

    log "===================================================="
    log "Processing cohort: $cohort_name"
    log "===================================================="

    cohort_log="$LOG_DIR/${cohort_name}_${TIMESTAMP}.log"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # --data_folder/--test_data_folder assume a per-cohort train/ and test/ directory
    # layout. That doesn't apply when --tokenized_full_dataset_path is set (the cohort
    # may be a single flat directory, and the train/test split comes from the
    # pre-tokenized dataset instead) -- --data_folder is a required argument, so we still
    # pass it (pointing at the cohort dir itself), but --test_data_folder is dropped since
    # it's unused in that mode.
    if [ -n "$TOKENIZED_FULL_DATASET_PATH" ]; then
        DATA_FOLDER_ARG="--data_folder \"$COHORT_FOLDER/$cohort_name\""
    else
        DATA_FOLDER_ARG="--data_folder \"$BASE_DIR/$cohort_name/train/\" --test_data_folder \"$BASE_DIR/$cohort_name/test/\""
    fi

    # Prepare command for feature extraction
    FEATURE_CMD="python -u -m cehrgpt.tools.linear_prob.compute_cehrgpt_features \
        $DATA_FOLDER_ARG \
        --dataset_prepared_path \"$DATASET_PREPARED_PATH\" \
        --model_name_or_path \"$MODEL_PATH\" \
        --tokenizer_name_or_path \"$TOKENIZER_PATH\" \
        --output_dir \"$output_dir\" \
        --preprocessing_num_workers \"$PREPROCESSING_WORKERS\" \
        --preprocessing_batch_size \"$PREPROCESSING_BATCH_SIZE\" \
        --per_device_eval_batch_size \"$BATCH_SIZE\" \
        --max_tokens_per_batch \"$MAX_TOKENS_PER_BATCH\" \
        --torch_dtype \"$TORCH_TYPE\""

    # Add optional flags based on configuration
    if [ "$DISABLE_SAMPLE_PACKING" = "false" ]; then
        FEATURE_CMD="$FEATURE_CMD --sample_packing"
    fi

    # The argument defaults to true, so the disabled case has to be passed explicitly
    if [ "$DISABLE_COMBINE_GLOBAL_LOCAL" = "true" ]; then
        FEATURE_CMD="$FEATURE_CMD --combine_global_local_features false"
    else
        FEATURE_CMD="$FEATURE_CMD --combine_global_local_features true"
    fi

    if [ "$ADD_RANDOM_TOKEN" = "true" ]; then
        FEATURE_CMD="$FEATURE_CMD --add_random_token"
    fi

    # Add tokenized_full_dataset_path if provided, so the cohort sequences are sliced
    # out of the fully tokenized dataset instead of being re-tokenized per cohort.
    # --cohort_folder points at the per-cohort dir (parquet files may be nested in
    # subdirectories, e.g. train/+test/ or a single data/ folder -- extract_cohort_sequences
    # globs recursively) so it can pull person_ids from all of them at once.
    if [ -n "$TOKENIZED_FULL_DATASET_PATH" ]; then
        FEATURE_CMD="$FEATURE_CMD \
        --tokenized_full_dataset_path \"$TOKENIZED_FULL_DATASET_PATH\" \
        --cohort_folder \"$COHORT_FOLDER/$cohort_name\""
    fi

    if [ -n "$GPU_IDS" ]; then
        FEATURE_CMD="$FEATURE_CMD --gpu_ids \"$GPU_IDS\""
    fi
    if [ -n "$WORKERS_PER_GPU" ]; then
        FEATURE_CMD="$FEATURE_CMD --workers_per_gpu \"$WORKERS_PER_GPU\""
    fi
    if [ -n "$DATALOADER_NUM_WORKERS" ]; then
        FEATURE_CMD="$FEATURE_CMD --dataloader_num_workers \"$DATALOADER_NUM_WORKERS\""
    fi
    if [ -n "$DATALOADER_PREFETCH_FACTOR" ]; then
        FEATURE_CMD="$FEATURE_CMD --dataloader_prefetch_factor \"$DATALOADER_PREFETCH_FACTOR\""
    fi

    # Add observation_window if provided
    if [ -n "$OBSERVATION_WINDOW" ]; then
        FEATURE_CMD="$FEATURE_CMD --observation_window \"$OBSERVATION_WINDOW\""
    fi

    if [ "$REFRESH_PROCESSED_DATASET" = "true" ]; then
        FEATURE_CMD="$FEATURE_CMD --refresh_processed_dataset"
    fi

    if [ "$IS_DATA_IN_MEDS" = "true" ]; then
        FEATURE_CMD="$FEATURE_CMD --is_data_in_meds"
    fi

    # Skip cohort persons that are missing from the tokenized dataset instead of failing
    if [ "$ALLOW_MISSING_TOKENIZED_PERSONS" = "true" ]; then
        FEATURE_CMD="$FEATURE_CMD --allow_missing_tokenized_persons"
    fi

    # Add force_attn_implementation if provided
    if [ -n "$FORCE_ATTN_IMPLEMENTATION" ]; then
        FEATURE_CMD="$FEATURE_CMD --force_attn_implementation \"$FORCE_ATTN_IMPLEMENTATION\""
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
    log "Command: python -u -m cehrgpt.tools.linear_prob.train_with_cehrgpt_features --features_data_dir $output_dir --output_dir $output_dir --report_to $REPORT_TO --feature_sets ${FEATURE_SETS_ARGS:-all}"

    # shellcheck disable=SC2086 -- the feature sets are separate words
    python -u -m cehrgpt.tools.linear_prob.train_with_cehrgpt_features \
        --features_data_dir "$output_dir" \
        --output_dir "$output_dir" \
        --report_to "$REPORT_TO" \
        --feature_sets ${FEATURE_SETS_ARGS:-all} \
        >> "$cohort_log" 2>&1

    model_training_status=$?
    if [ $model_training_status -ne 0 ]; then
        log "ERROR: Model training failed for $cohort_name. Check $cohort_log for details."
        continue
    fi

    # Step 3: meds-evaluation of every feature set that was trained
    evaluation_failed="false"
    for result_dir in $(required_result_dirs "$output_dir"); do
        log "Running meds-evaluation for logistic regression ($result_dir) for $cohort_name..."
        evaluation_output=$("$SCRIPT_DIR/run_meds_evaluation.sh" \
            --output_dir="$output_dir/$result_dir/" \
            "$output_dir/$result_dir/test_predictions" 2>&1)
        evaluation_status=$?
        echo "$evaluation_output" >> "$cohort_log"
        if [ $evaluation_status -ne 0 ]; then
            log "ERROR: Running meds-evaluation failed for logistic regression ($result_dir) for cohort $cohort_name. Check $cohort_log for details."
            evaluation_failed="true"
            continue
        fi
        log "Results for $cohort_name ($result_dir):"
        log "$evaluation_output"
    done
    if [ "$evaluation_failed" = "true" ]; then
        continue
    fi

    log "Successfully completed processing for $cohort_name"
done < "$TEMP_COHORT_LIST"

# Clean up
rm -f "$TEMP_COHORT_LIST"

log "All processing completed"
