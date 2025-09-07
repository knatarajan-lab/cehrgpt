#!/usr/bin/env bash
#
# cohort_processor.sh - Process multiple cohort directories using a data extraction script
#
# This script processes each cohort directory within a specified folder
# using a Python script for data extraction.

set -e  # Exit immediately if a command exits with a non-zero status

# Display help documentation
function display_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script processes each cohort directory found within a specified cohort folder"
    echo "using a Python script for data extraction."
    echo ""
    echo "Options:"
    echo "  -h, --help                        Display this help message and exit"
    echo "  -v, --verbose                     Enable verbose output"
    echo "  -c, --cohort-folder DIR           Specify the cohort folder (default: \$COHORT_FOLDER env variable)"
    echo "  -i, --input-dir DIR               Specify the input directory (default: \$INPUT_DIR env variable)"
    echo "  -o, --output-dir DIR              Specify the output directory (default: \$OUTPUT_DIR env variable)"
    echo "  -p, --patient-splits-folder DIR   Specify the patient splits folder (default: \$PATIENT_SPLITS_FOLDER env variable)"
    echo "  -e, --ehr-tables LIST             Specify the EHR tables as a space-separated list in quotes"
    echo "                                    (default: \$EHR_TABLES env variable)"
    echo "  -ov, --observation-window DAYS    Specify the observation window in days (default: 0)"
    echo ""
    echo "Environment Variables (used as defaults if parameters not provided):"
    echo "  COHORT_FOLDER                     Base directory for cohorts"
    echo "  INPUT_DIR                         Base input directory"
    echo "  OUTPUT_DIR                        Base output directory"
    echo "  PATIENT_SPLITS_FOLDER             Patient splits folder"
    echo "  EHR_TABLES                        Space-separated list of EHR tables to process"
    echo ""
    echo "Example:"
    echo "  $0 -c /path/to/cohorts -i /path/to/input -o /path/to/output"
    echo ""
    echo "Note: This script must be run from the root of the project directory."
}

# Check if required environment variables are set or provided via command line args
# (We'll check after parsing command line arguments)

# Parse command line arguments
VERBOSE=false
OBSERVATION_WINDOW=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            display_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--cohort-folder)
            COHORT_FOLDER="$2"
            shift 2
            ;;
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--patient-splits-folder)
            PATIENT_SPLITS_FOLDER="$2"
            shift 2
            ;;
        -e|--ehr-tables)
            EHR_TABLES="$2"
            shift 2
            ;;
        -ov|--observation-window)
            OBSERVATION_WINDOW="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown parameter: $1"
            display_help
            exit 1
            ;;
    esac
done

# Verify that all required parameters are provided
missing_params=""

if [ -z "$COHORT_FOLDER" ]; then
    missing_params="$missing_params COHORT_FOLDER (--cohort-folder)"
fi

if [ -z "$INPUT_DIR" ]; then
    missing_params="$missing_params INPUT_DIR (--input-dir)"
fi

if [ -z "$OUTPUT_DIR" ]; then
    missing_params="$missing_params OUTPUT_DIR (--output-dir)"
fi

if [ -z "$PATIENT_SPLITS_FOLDER" ]; then
    missing_params="$missing_params PATIENT_SPLITS_FOLDER (--patient-splits-folder)"
fi

if [ -z "$EHR_TABLES" ]; then
    missing_params="$missing_params EHR_TABLES (--ehr-tables)"
fi

# If any required parameters are missing, show error and exit
if [ -n "$missing_params" ]; then
    echo "Error: Missing required parameters:"
    for param in $missing_params; do
        echo "  - $param"
    done
    echo ""
    echo "Please provide these values either as environment variables or command-line parameters."
    echo "Run '$0 --help' for usage information."
    exit 1
fi

# Display configured directories
if [ "$VERBOSE" = true ]; then
    echo "Using the following directories:"
    echo "  COHORT_FOLDER: $COHORT_FOLDER"
    echo "  INPUT_DIR: $INPUT_DIR"
    echo "  OUTPUT_DIR: $OUTPUT_DIR"
    echo "  PATIENT_SPLITS_FOLDER: $PATIENT_SPLITS_FOLDER"
    echo "  EHR_TABLES: $EHR_TABLES"
    echo "  Observation Window: $OBSERVATION_WINDOW days"
fi

# Check if required directories exist
if [ ! -d "$COHORT_FOLDER" ]; then
    echo "Error: Cohort folder does not exist: $COHORT_FOLDER"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d "$PATIENT_SPLITS_FOLDER" ]; then
    echo "Error: Patient splits folder does not exist: $PATIENT_SPLITS_FOLDER"
    exit 1
fi

# Count the number of cohort directories
COHORT_COUNT=$(find "$COHORT_FOLDER" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$COHORT_COUNT" -eq 0 ]; then
    echo "Warning: No cohort directories found in $COHORT_FOLDER"
    exit 0
fi

echo "Found $COHORT_COUNT cohort directories to process"
echo "Starting processing at $(date)"

# Process counter
PROCESSED=0
FAILED=0

# Loop through each sub-directory in the cohort folder
for cohort_dir in "$COHORT_FOLDER"/*; do
    if [ -d "$cohort_dir" ]; then
        COHORT_NAME=$(basename "$cohort_dir")
        echo "[$((PROCESSED+1))/$COHORT_COUNT] Processing cohort: $COHORT_NAME"

        # Display verbose information if enabled
        if [ "$VERBOSE" = true ]; then
            echo "  Input directory: $INPUT_DIR"
            echo "  Output directory: $OUTPUT_DIR"
            echo "  Cohort directory: $cohort_dir"
            echo "  Observation Window: $OBSERVATION_WINDOW days"
        fi

        # Run the Python script with the directory-specific arguments
        if python -u -m cehrbert_data.tools.extract_features \
            -c "$COHORT_NAME" \
            -i "$INPUT_DIR" \
            -o "$OUTPUT_DIR" \
            -dl 1985-01-01 \
            -du 2023-12-31 \
            --cohort_dir "$cohort_dir" \
            --person_id_column subject_id \
            --index_date_column prediction_time \
            --label_column boolean_value \
            -ip \
            --gpt_patient_sequence \
            --att_type day \
            --inpatient_att_type day \
            -iv \
            --ehr_table_list $EHR_TABLES \
            --patient_splits_folder "$PATIENT_SPLITS_FOLDER" \
            --cache_events \
            --should_construct_artificial_visits \
            --disconnect_problem_list_records \
            --include_concept_list \
            --keep_samples_with_no_features \
            --observation_window "$OBSERVATION_WINDOW"; then

            echo "✅ Successfully processed cohort: $COHORT_NAME"
            PROCESSED=$((PROCESSED+1))
        else
            echo "❌ Failed to process cohort: $COHORT_NAME"
            FAILED=$((FAILED+1))
        fi

        echo "--------------------------------------"
    fi
done

# Print summary
echo "Processing complete at $(date)"
echo "Summary: $PROCESSED cohorts processed successfully, $FAILED failed"

if [ $FAILED -gt 0 ]; then
    echo "Warning: Some cohorts failed to process"
    exit 1
else
    echo "All cohorts have been processed successfully"
    exit 0
fi
