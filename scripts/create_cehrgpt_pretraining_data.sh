#!/bin/sh

# Function to display usage
usage() {
    echo "Usage: $0 --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER --start_date START_DATE"
    echo ""
    echo "Required Arguments:"
    echo "  --input_folder PATH      Input folder path"
    echo "  --output_folder PATH     Output folder path"
    echo "  --start_date DATE        Start date"
    echo ""
    echo "Example:"
    echo "  $0 --input_folder /path/to/input --output_folder /path/to/output --start_date 1985-01-01"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize variables
INPUT_FOLDER=""
OUTPUT_FOLDER=""
START_DATE=""

# Domain tables (sh-compatible - space-separated string)
DOMAIN_TABLES="condition_occurrence procedure_occurrence drug_exposure"

# Parse command line arguments without getopt
while [ $# -gt 0 ]; do
    case $1 in
        --input_folder)
            if [ -z "$2" ]; then
                echo "Error: --input_folder requires a value"
                usage
            fi
            # Check if next argument starts with --
            case "$2" in
                --*)
                    echo "Error: --input_folder requires a value"
                    usage
                    ;;
            esac
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output_folder)
            if [ -z "$2" ]; then
                echo "Error: --output_folder requires a value"
                usage
            fi
            # Check if next argument starts with --
            case "$2" in
                --*)
                    echo "Error: --output_folder requires a value"
                    usage
                    ;;
            esac
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --start_date)
            if [ -z "$2" ]; then
                echo "Error: --start_date requires a value"
                usage
            fi
            # Check if next argument starts with --
            case "$2" in
                --*)
                    echo "Error: --start_date requires a value"
                    usage
                    ;;
            esac
            START_DATE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            ;;
        *)
            echo "Error: Unexpected argument $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ] || [ -z "$START_DATE" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

echo "Starting CEHR-GPT preprocessing with:"
echo "  Input folder: $INPUT_FOLDER"
echo "  Output folder: $OUTPUT_FOLDER"
echo "  Start date: $START_DATE"
echo ""

export CEHRBERT_DATA_HOME=$(python -c "import cehrbert_data; print(cehrbert_data.__file__.rsplit('/', 1)[0])")

# Check if SPARK_SUBMIT_OPTIONS is set and not empty
if [ -n "$SPARK_SUBMIT_OPTIONS" ]; then
    SPARK_OPTIONS="$SPARK_SUBMIT_OPTIONS"
    echo "Using Spark options: $SPARK_OPTIONS"
else
    SPARK_OPTIONS=""
    echo "No Spark options specified"
fi

# Step 1: Generate included concept list
CONCEPT_LIST_CMD="spark-submit $SPARK_OPTIONS $CEHRBERT_DATA_HOME/apps/generate_included_concept_list.py \
-i \"$INPUT_FOLDER\" \
-o \"$OUTPUT_FOLDER\" \
--min_num_of_patients 100 \
--ehr_table_list $DOMAIN_TABLES"

echo "Running concept list generation:"
echo "$CONCEPT_LIST_CMD"
eval "$CONCEPT_LIST_CMD"

if [ $? -ne 0 ]; then
    echo "Error: Concept list generation failed"
    exit 1
fi

# Step 2: Generate training data
TRAINING_DATA_CMD="spark-submit $SPARK_OPTIONS $CEHRBERT_DATA_HOME/apps/generate_training_data.py \
--input_folder \"$INPUT_FOLDER\" \
--output_folder \"$OUTPUT_FOLDER\" \
-d $START_DATE \
--att_type day \
--inpatient_att_type day \
-iv \
-ip \
--include_concept_list \
--gpt_patient_sequence \
--should_construct_artificial_visits \
--disconnect_problem_list_records \
--domain_table_list $DOMAIN_TABLES"

echo "Running training data generation:"
echo "$TRAINING_DATA_CMD"
eval "$TRAINING_DATA_CMD"

if [ $? -ne 0 ]; then
    echo "Error: Training data generation failed"
    exit 1
fi

echo "CEHR-GPT preprocessing completed successfully!"
