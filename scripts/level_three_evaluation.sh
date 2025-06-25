#!/bin/bash

# Medical Cohort Generation and Evaluation Script
# This script generates various medical prediction cohorts and runs evaluations on them
# using the CEHR-BERT framework with OMOP Common Data Model

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script metadata
SCRIPT_NAME=$(basename "$0")
VERSION="1.0.0"

# Default values
DEFAULT_DATE_LOWER="1985-01-01"
DEFAULT_DATE_UPPER="2023-12-31"
DEFAULT_AGE_LOWER=18
DEFAULT_AGE_UPPER=100
VERBOSE=false
DRY_RUN=false

# Available cohorts
AVAILABLE_COHORTS=("cad_cabg" "hf_readmission" "copd_readmission" "hospitalization" "afib_ischemic_stroke")

# Function to display usage information
show_help() {
    cat << EOF
$SCRIPT_NAME - Medical Cohort Generation and Evaluation Script

DESCRIPTION:
    This script generates medical prediction cohorts using OMOP Common Data Model
    and runs baseline model evaluations using the CEHR-BERT framework.

USAGE:
    $SCRIPT_NAME [OPTIONS] -o OMOP_FOLDER

REQUIRED ARGUMENTS:
    -o, --omop-folder PATH          Path to OMOP data folder

OPTIONAL ARGUMENTS:
    -p, --patient-splits PATH       Path to patient splits folder (optional)
    -c, --cohorts LIST              Comma-separated list of cohorts to generate
                                   Available: ${AVAILABLE_COHORTS[*]}
                                   Default: all cohorts
    -dl, --date-lower DATE          Lower date bound (default: $DEFAULT_DATE_LOWER)
    -du, --date-upper DATE          Upper date bound (default: $DEFAULT_DATE_UPPER)
    -l, --age-lower AGE             Lower age limit (default: $DEFAULT_AGE_LOWER)
    -u, --age-upper AGE             Upper age limit (default: $DEFAULT_AGE_UPPER)
    -v, --verbose                   Enable verbose output
    -n, --dry-run                   Show what would be executed without running
    -h, --help                      Show this help message
    --version                       Show version information

COHORT DESCRIPTIONS:
    cad_cabg                Coronary Artery Disease CABG prediction
    hf_readmission          Heart Failure readmission prediction
    copd_readmission        COPD readmission prediction
    hospitalization         General hospitalization prediction
    afib_ischemic_stroke    Atrial Fibrillation ischemic stroke prediction

EXAMPLES:
    # Generate all cohorts with default settings
    $SCRIPT_NAME -o /path/to/omop/data

    # Generate specific cohorts with patient splits
    $SCRIPT_NAME -o /path/to/omop/data -p /path/to/splits -c "cad_cabg,hf_readmission"

    # Dry run to see what would be executed
    $SCRIPT_NAME -o /path/to/omop/data --dry-run

    # Verbose output with custom date range
    $SCRIPT_NAME -o /path/to/omop/data -v -dl 2010-01-01 -du 2022-12-31

EXIT CODES:
    0    Success
    1    General error
    2    Invalid arguments
    3    Missing dependencies

AUTHOR:
    Generated from existing medical cohort processing script

VERSION:
    $VERSION
EOF
}

# Function to display version
show_version() {
    echo "$SCRIPT_NAME version $VERSION"
}

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        "INFO")
            echo "[$timestamp] INFO: $message"
            ;;
        "WARN")
            echo "[$timestamp] WARN: $message" >&2
            ;;
        "ERROR")
            echo "[$timestamp] ERROR: $message" >&2
            ;;
        "DEBUG")
            if [[ "$VERBOSE" == true ]]; then
                echo "[$timestamp] DEBUG: $message"
            fi
            ;;
    esac
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()

    # Check for Python
    if ! command -v python &> /dev/null; then
        missing_deps+=("python")
    fi

    # Check for required Python modules (basic check)
    if ! python -c "import sys" &> /dev/null; then
        missing_deps+=("python environment issue")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        log "ERROR" "Please install missing dependencies before running this script"
        exit 3
    fi
}

# Function to validate arguments
validate_arguments() {
    # Check if OMOP folder exists
    if [[ ! -d "$OMOP_FOLDER" ]]; then
        log "ERROR" "OMOP folder does not exist: $OMOP_FOLDER"
        exit 2
    fi

    # Check patient splits folder if provided
    if [[ -n "$PATIENT_SPLITS_FOLDER" && ! -d "$PATIENT_SPLITS_FOLDER" ]]; then
        log "ERROR" "Patient splits folder does not exist: $PATIENT_SPLITS_FOLDER"
        exit 2
    fi

    # Validate date format (basic check)
    if ! date -d "$DATE_LOWER" &> /dev/null; then
        log "ERROR" "Invalid date format for lower bound: $DATE_LOWER"
        exit 2
    fi

    if ! date -d "$DATE_UPPER" &> /dev/null; then
        log "ERROR" "Invalid date format for upper bound: $DATE_UPPER"
        exit 2
    fi

    # Validate age ranges
    if [[ $AGE_LOWER -lt 0 || $AGE_UPPER -lt 0 || $AGE_LOWER -gt $AGE_UPPER ]]; then
        log "ERROR" "Invalid age range: $AGE_LOWER to $AGE_UPPER"
        exit 2
    fi
}

# Function to create directory if it doesn't exist
create_directory_if_not_exists() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log "INFO" "Creating directory: $dir"
        if [[ "$DRY_RUN" == false ]]; then
            mkdir -p "$dir"
        fi
    fi
}

# Function to execute command with dry-run support
execute_command() {
    local cmd="$*"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would execute: $cmd"
    else
        log "DEBUG" "Executing: $cmd"
        eval "$cmd"
    fi
}

# Function to generate CAD CABG cohort
generate_cad_cabg() {
    log "INFO" "Generating CAD CABG cohort"

    local cohort_dir="$OMOP_FOLDER/cohorts/cad_cabg"
    local eval_dir="$OMOP_FOLDER/evaluation_gpt/cad_cabg"

    create_directory_if_not_exists "$cohort_dir"
    create_directory_if_not_exists "$eval_dir"

    # Generate cohort
    local cmd="python -u -m cehrbert_data.prediction_cohorts.cad_cabg_cohort"
    cmd+=" -c cad_cabg_bow"
    cmd+=" -i '$OMOP_FOLDER'"
    cmd+=" -o '$cohort_dir/'"
    cmd+=" -dl $DATE_LOWER -du $DATE_UPPER"
    cmd+=" -l $AGE_LOWER -u $AGE_UPPER -ow 360 -ps 0 -pw 360 -f"
    cmd+=" --att_type cehr_bert"
    cmd+=" --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv"
    cmd+=" --is_remove_index_prediction_starts"

    execute_command "$cmd"

    # Run evaluation
    log "INFO" "Running predictions on CAD CABG cohort"
    cmd="python -m cehrbert.evaluations.evaluation"
    cmd+=" -a baseline_model"
    cmd+=" -d '$cohort_dir/cad_cabg_bow/'"
    cmd+=" -ef '$eval_dir/'"

    if [[ -n "$PATIENT_SPLITS_FOLDER" ]]; then
        cmd+=" --patient_splits_folder '$PATIENT_SPLITS_FOLDER'"
    fi

    execute_command "$cmd"
}

# Function to generate HF readmission cohort
generate_hf_readmission() {
    log "INFO" "Generating HF readmission cohort"

    local cohort_dir="$OMOP_FOLDER/cohorts/hf_readmission"
    local eval_dir="$OMOP_FOLDER/evaluation_gpt/hf_readmission"

    create_directory_if_not_exists "$cohort_dir"
    create_directory_if_not_exists "$eval_dir"

    # Generate cohort
    local cmd="python -u -m cehrbert_data.prediction_cohorts.hf_readmission"
    cmd+=" -c hf_readmission_bow"
    cmd+=" -i '$OMOP_FOLDER'"
    cmd+=" -o '$cohort_dir'"
    cmd+=" -dl $DATE_LOWER -du $DATE_UPPER"
    cmd+=" -l $AGE_LOWER -u $AGE_UPPER -ow 360 -ps 1 -pw 30 -f"
    cmd+=" --att_type cehr_bert"
    cmd+=" --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv"
    cmd+=" --is_remove_index_prediction_starts"

    execute_command "$cmd"

    # Run evaluation
    log "INFO" "Running predictions on HF readmission cohort"
    cmd="python -m cehrbert.evaluations.evaluation"
    cmd+=" -a baseline_model"
    cmd+=" -d '$cohort_dir/hf_readmission_bow/'"
    cmd+=" -ef '$eval_dir/'"

    if [[ -n "$PATIENT_SPLITS_FOLDER" ]]; then
        cmd+=" --patient_splits_folder '$PATIENT_SPLITS_FOLDER'"
    fi

    execute_command "$cmd"
}

# Function to generate COPD readmission cohort
generate_copd_readmission() {
    log "INFO" "Generating COPD readmission cohort"

    local cohort_dir="$OMOP_FOLDER/cohorts/copd_readmission"
    local eval_dir="$OMOP_FOLDER/evaluation_gpt/copd_readmission"

    create_directory_if_not_exists "$cohort_dir"
    create_directory_if_not_exists "$eval_dir"

    # Generate cohort
    local cmd="python -u -m cehrbert_data.prediction_cohorts.copd_readmission"
    cmd+=" -c copd_readmission_bow"
    cmd+=" -i '$OMOP_FOLDER'"
    cmd+=" -o '$cohort_dir'"
    cmd+=" -dl $DATE_LOWER -du $DATE_UPPER"
    cmd+=" -l $AGE_LOWER -u $AGE_UPPER -ow 360 -ps 1 -pw 30 -f"
    cmd+=" --att_type cehr_bert"
    cmd+=" --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv"
    cmd+=" --is_remove_index_prediction_starts"

    execute_command "$cmd"

    # Run evaluation
    log "INFO" "Running predictions on COPD readmission cohort"
    cmd="python -m cehrbert.evaluations.evaluation"
    cmd+=" -a baseline_model"
    cmd+=" -d '$cohort_dir/copd_readmission_bow/'"
    cmd+=" -ef '$eval_dir/'"

    if [[ -n "$PATIENT_SPLITS_FOLDER" ]]; then
        cmd+=" --patient_splits_folder '$PATIENT_SPLITS_FOLDER'"
    fi

    execute_command "$cmd"
}

# Function to generate hospitalization cohort
generate_hospitalization() {
    log "INFO" "Generating hospitalization cohort"

    local cohort_dir="$OMOP_FOLDER/cohorts/hospitalization"
    local eval_dir="$OMOP_FOLDER/evaluation_gpt/hospitalization"

    create_directory_if_not_exists "$cohort_dir"
    create_directory_if_not_exists "$eval_dir"

    # Generate cohort
    local cmd="python -u -m cehrbert_data.prediction_cohorts.hospitalization"
    cmd+=" -c hospitalization_bow"
    cmd+=" -i '$OMOP_FOLDER'"
    cmd+=" -o '$cohort_dir'"
    cmd+=" -dl $DATE_LOWER -du $DATE_UPPER"
    cmd+=" -l $AGE_LOWER -u $AGE_UPPER -ow 540 -hw 180 -ps 0 -pw 360 -f -iw"
    cmd+=" --att_type cehr_bert"
    cmd+=" --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv"

    execute_command "$cmd"

    # Run evaluation
    log "INFO" "Running predictions on hospitalization cohort"
    cmd="python -m cehrbert.evaluations.evaluation"
    cmd+=" -a baseline_model"
    cmd+=" -d '$cohort_dir/hospitalization_bow/'"
    cmd+=" -ef '$eval_dir/'"

    if [[ -n "$PATIENT_SPLITS_FOLDER" ]]; then
        cmd+=" --patient_splits_folder '$PATIENT_SPLITS_FOLDER'"
    fi

    execute_command "$cmd"
}

# Function to generate AFIB ischemic stroke cohort
generate_afib_ischemic_stroke() {
    log "INFO" "Generating AFIB ischemic stroke cohort"

    local cohort_dir="$OMOP_FOLDER/cohorts/afib_ischemic_stroke"
    local eval_dir="$OMOP_FOLDER/evaluation_gpt/afib_ischemic_stroke"

    create_directory_if_not_exists "$cohort_dir"
    create_directory_if_not_exists "$eval_dir"

    # Generate cohort
    local cmd="python -u -m cehrbert_data.prediction_cohorts.afib_ischemic_stroke"
    cmd+=" -c afib_ischemic_stroke_bow"
    cmd+=" -i '$OMOP_FOLDER'"
    cmd+=" -o '$cohort_dir'"
    cmd+=" -dl $DATE_LOWER -du $DATE_UPPER"
    cmd+=" -l $AGE_LOWER -u $AGE_UPPER -ow 720 -ps 0 -pw 360 -f"
    cmd+=" --att_type cehr_bert"
    cmd+=" --ehr_table_list condition_occurrence procedure_occurrence drug_exposure -iv"
    cmd+=" --is_remove_index_prediction_starts"

    execute_command "$cmd"

    # Run evaluation
    log "INFO" "Running predictions on AFIB ischemic stroke cohort"
    cmd="python -m cehrbert.evaluations.evaluation"
    cmd+=" -a baseline_model"
    cmd+=" -d '$cohort_dir/afib_ischemic_stroke_bow/'"
    cmd+=" -ef '$eval_dir/'"

    if [[ -n "$PATIENT_SPLITS_FOLDER" ]]; then
        cmd+=" --patient_splits_folder '$PATIENT_SPLITS_FOLDER'"
    fi

    execute_command "$cmd"
}

# Function to process cohorts
process_cohorts() {
    local cohorts_to_process=("${COHORTS[@]}")

    log "INFO" "Processing cohorts: ${cohorts_to_process[*]}"
    log "INFO" "OMOP Folder: $OMOP_FOLDER"
    if [[ -n "$PATIENT_SPLITS_FOLDER" ]]; then
        log "INFO" "Patient Splits Folder: $PATIENT_SPLITS_FOLDER"
    fi
    log "INFO" "Date Range: $DATE_LOWER to $DATE_UPPER"
    log "INFO" "Age Range: $AGE_LOWER to $AGE_UPPER"

    for cohort in "${cohorts_to_process[@]}"; do
        case "$cohort" in
            "cad_cabg")
                generate_cad_cabg
                ;;
            "hf_readmission")
                generate_hf_readmission
                ;;
            "copd_readmission")
                generate_copd_readmission
                ;;
            "hospitalization")
                generate_hospitalization
                ;;
            "afib_ischemic_stroke")
                generate_afib_ischemic_stroke
                ;;
            *)
                log "WARN" "Unknown cohort: $cohort"
                ;;
        esac
    done
}

# Initialize variables
OMOP_FOLDER=""
PATIENT_SPLITS_FOLDER=""
COHORTS=("${AVAILABLE_COHORTS[@]}")  # Default to all cohorts
DATE_LOWER="$DEFAULT_DATE_LOWER"
DATE_UPPER="$DEFAULT_DATE_UPPER"
AGE_LOWER=$DEFAULT_AGE_LOWER
AGE_UPPER=$DEFAULT_AGE_UPPER

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --version)
            show_version
            exit 0
            ;;
        -o|--omop-folder)
            OMOP_FOLDER="$2"
            shift 2
            ;;
        -p|--patient-splits)
            PATIENT_SPLITS_FOLDER="$2"
            shift 2
            ;;
        -c|--cohorts)
            IFS=',' read -ra COHORTS <<< "$2"
            shift 2
            ;;
        -dl|--date-lower)
            DATE_LOWER="$2"
            shift 2
            ;;
        -du|--date-upper)
            DATE_UPPER="$2"
            shift 2
            ;;
        -l|--age-lower)
            AGE_LOWER="$2"
            shift 2
            ;;
        -u|--age-upper)
            AGE_UPPER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            log "ERROR" "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 2
            ;;
        *)
            log "ERROR" "Unexpected positional argument: $1"
            echo "Use -h or --help for usage information"
            exit 2
            ;;
    esac
done

# Check required arguments
if [[ -z "$OMOP_FOLDER" ]]; then
    log "ERROR" "OMOP folder is required (-o/--omop-folder)"
    echo "Use -h or --help for usage information"
    exit 2
fi

# Export environment variables for child processes
export OMOP_FOLDER
export PATIENT_SPLITS_FOLDER

# Main execution
main() {
    log "INFO" "Starting Medical Cohort Generation and Evaluation Script v$VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "DRY RUN MODE - No actual commands will be executed"
    fi

    # Check dependencies
    check_dependencies

    # Validate arguments
    validate_arguments

    # Process cohorts
    process_cohorts

    log "INFO" "Script completed successfully"
}

# Run main function
main "$@"
