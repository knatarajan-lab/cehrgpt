#!/bin/sh

# OMOP Pipeline Script
# Converts patient sequences to OMOP format and reconstructs OMOP tables

set -e  # Exit on any error

# Script information
SCRIPT_NAME=$(basename "$0")
VERSION="1.0.0"

# Default values
BUFFER_SIZE=1280
CPU_CORES=10
DOMAIN_TABLES="condition_occurrence drug_exposure procedure_occurrence measurement"

# Function to display help
show_help() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Convert patient sequences to OMOP format and reconstruct OMOP tables.

REQUIRED OPTIONS:
    -p, --patient-sequence-folder PATH    Directory containing patient sequence files
    -o, --omop-folder PATH               Output directory for OMOP-formatted data
    -s, --source-omop-folder PATH        Source directory containing OMOP vocabulary tables

OPTIONAL SETTINGS:
    -b, --buffer-size SIZE               Buffer size for processing (default: $BUFFER_SIZE)
    -c, --cpu-cores CORES               Number of CPU cores to use (default: $CPU_CORES)
    -d, --domain-tables LIST            Space-separated list of domain tables (default: "$DOMAIN_TABLES")
    -h, --help                          Show this help message and exit
    -v, --version                       Show version information and exit

EXAMPLES:
    # Basic usage
    $SCRIPT_NAME --patient-sequence-folder /path/to/sequences --omop-folder /path/to/output --source-omop-folder /path/to/source

    # With custom buffer size and CPU cores
    $SCRIPT_NAME -p /path/to/sequences -o /path/to/output -s /path/to/source -b 2560 -c 16

    # With custom domain tables
    $SCRIPT_NAME --patient-sequence-folder /path/to/sequences --omop-folder /path/to/output --source-omop-folder /path/to/source --domain-tables "condition_occurrence procedure_occurrence"

DESCRIPTION:
    This script performs the following operations:
    1. Creates output directory if it doesn't exist
    2. Removes existing OMOP tables from output directory
    3. Copies OMOP concept tables from source directory
    4. Converts patient sequences to OMOP format
    5. Reconstructs observation_period table
    6. Reconstructs condition_era table

EOF
}

# Function to display version
show_version() {
    echo "$SCRIPT_NAME version $VERSION"
}

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if string starts with dash
starts_with_dash() {
    case "$1" in
        -*) return 0 ;;
        *) return 1 ;;
    esac
}

# Function to validate arguments
validate_args() {
    if [ -z "$PATIENT_SEQUENCE_FOLDER" ]; then
        echo "Error: --patient-sequence-folder is required" >&2
        show_help
        exit 1
    fi

    if [ -z "$OMOP_FOLDER" ]; then
        echo "Error: --omop-folder is required" >&2
        show_help
        exit 1
    fi

    if [ -z "$SOURCE_OMOP_FOLDER" ]; then
        echo "Error: --source-omop-folder is required" >&2
        show_help
        exit 1
    fi

    if [ ! -d "$PATIENT_SEQUENCE_FOLDER" ]; then
        echo "Error: Patient sequence folder '$PATIENT_SEQUENCE_FOLDER' does not exist" >&2
        exit 1
    fi

    if [ ! -d "$SOURCE_OMOP_FOLDER" ]; then
        echo "Error: Source OMOP folder '$SOURCE_OMOP_FOLDER' does not exist" >&2
        exit 1
    fi

    # Validate numeric arguments
    case "$BUFFER_SIZE" in
        ''|*[!0-9]*)
            echo "Error: Buffer size must be a positive integer" >&2
            exit 1
            ;;
        *)
            if [ "$BUFFER_SIZE" -le 0 ]; then
                echo "Error: Buffer size must be a positive integer" >&2
                exit 1
            fi
            ;;
    esac

    case "$CPU_CORES" in
        ''|*[!0-9]*)
            echo "Error: CPU cores must be a positive integer" >&2
            exit 1
            ;;
        *)
            if [ "$CPU_CORES" -le 0 ]; then
                echo "Error: CPU cores must be a positive integer" >&2
                exit 1
            fi
            ;;
    esac
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -p|--patient-sequence-folder)
            if [ -z "$2" ] || starts_with_dash "$2"; then
                echo "Error: --patient-sequence-folder requires a value" >&2
                exit 1
            fi
            PATIENT_SEQUENCE_FOLDER="$2"
            shift 2
            ;;
        --patient-sequence-folder=*)
            PATIENT_SEQUENCE_FOLDER="${1#*=}"
            shift
            ;;
        -o|--omop-folder)
            if [ -z "$2" ] || starts_with_dash "$2"; then
                echo "Error: --omop-folder requires a value" >&2
                exit 1
            fi
            OMOP_FOLDER="$2"
            shift 2
            ;;
        --omop-folder=*)
            OMOP_FOLDER="${1#*=}"
            shift
            ;;
        -s|--source-omop-folder)
            if [ -z "$2" ] || starts_with_dash "$2"; then
                echo "Error: --source-omop-folder requires a value" >&2
                exit 1
            fi
            SOURCE_OMOP_FOLDER="$2"
            shift 2
            ;;
        --source-omop-folder=*)
            SOURCE_OMOP_FOLDER="${1#*=}"
            shift
            ;;
        -b|--buffer-size)
            if [ -z "$2" ] || starts_with_dash "$2"; then
                echo "Error: --buffer-size requires a value" >&2
                exit 1
            fi
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --buffer-size=*)
            BUFFER_SIZE="${1#*=}"
            shift
            ;;
        -c|--cpu-cores)
            if [ -z "$2" ] || starts_with_dash "$2"; then
                echo "Error: --cpu-cores requires a value" >&2
                exit 1
            fi
            CPU_CORES="$2"
            shift 2
            ;;
        --cpu-cores=*)
            CPU_CORES="${1#*=}"
            shift
            ;;
        -d|--domain-tables)
            if [ -z "$2" ] || starts_with_dash "$2"; then
                echo "Error: --domain-tables requires a value" >&2
                exit 1
            fi
            DOMAIN_TABLES="$2"
            shift 2
            ;;
        --domain-tables=*)
            DOMAIN_TABLES="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            show_version
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            echo "Use --help to see available options" >&2
            exit 1
            ;;
    esac
done

# Validate arguments
validate_args

# Set derived environment variables
export PATIENT_SEQUENCE_FOLDER
export OMOP_FOLDER
export SOURCE_OMOP_FOLDER
export PATIENT_SPLITS_FOLDER="$SOURCE_OMOP_FOLDER/patient_splits"

# Display configuration
log "Starting OMOP pipeline conversion"
log "Configuration:"
log "  Patient Sequence Folder: $PATIENT_SEQUENCE_FOLDER"
log "  OMOP Output Folder: $OMOP_FOLDER"
log "  Source OMOP Folder: $SOURCE_OMOP_FOLDER"
log "  Patient Splits Folder: $PATIENT_SPLITS_FOLDER"
log "  Buffer Size: $BUFFER_SIZE"
log "  CPU Cores: $CPU_CORES"
log "  Domain Tables: $DOMAIN_TABLES"

# Create output directory if it doesn't exist
if [ ! -d "$OMOP_FOLDER" ]; then
    log "Creating output directory: $OMOP_FOLDER"
    mkdir -p "$OMOP_FOLDER"
fi

# Remove existing OMOP tables
log "Removing existing OMOP tables"
rm -rf $OMOP_FOLDER/{person,visit_occurrence,condition_occurrence,procedure_occurrence,drug_exposure,death,measurement,observation_period,condition_era}

# Remove existing OMOP concept tables
log "Removing existing OMOP concept tables"
rm -rf $OMOP_FOLDER/{concept,concept_ancestor,concept_relationship}

# Copy OMOP concept tables if they don't already exist
log "Copying OMOP concept tables"
for table in concept concept_relationship concept_ancestor; do
    if [ ! -d "$OMOP_FOLDER/$table" ]; then
        log "Copying $table table"
        if [ -d "$SOURCE_OMOP_FOLDER/$table" ]; then
            cp -r "$SOURCE_OMOP_FOLDER/$table" "$OMOP_FOLDER/$table"
        else
            echo "Warning: Source table '$SOURCE_OMOP_FOLDER/$table' not found" >&2
        fi
    else
        log "Table $table already exists, skipping"
    fi
done

# Reconstruct OMOP instance from patient sequences
log "Reconstructing OMOP instance from patient sequences"
python -m cehrgpt.generation.omop_converter_batch \
  --patient_sequence_path "$PATIENT_SEQUENCE_FOLDER" \
  --output_folder "$OMOP_FOLDER" \
  --concept_path "$OMOP_FOLDER/concept" \
  --buffer_size "$BUFFER_SIZE" \
  --cpu_cores "$CPU_CORES"

# Create observation_period
log "Reconstructing observation_period table"
python -u -m cehrgpt.omop.observation_period \
  --input_folder "$OMOP_FOLDER" \
  --output_folder "$OMOP_FOLDER" \
  --domain_table_list "$DOMAIN_TABLES"

# Create condition_era
log "Reconstructing condition_era table"
python -u -m cehrgpt.omop.condition_era \
  --input_folder "$OMOP_FOLDER" \
  --output_folder "$OMOP_FOLDER" \
  --domain_table_list "condition_occurrence"

log "OMOP pipeline conversion completed successfully"
