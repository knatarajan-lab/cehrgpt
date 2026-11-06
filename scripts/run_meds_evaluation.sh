#!/bin/sh

# Run meds-evaluation on any number of folders of predictions.

usage() {
    echo "Usage: $0 [options] PATH [PATH ...]"
    echo ""
    echo "Runs meds-evaluation-cli on each predictions folder: a folder with the prediction parquet files"
    echo "(e.g. <cohort>/<model>/logistic/test_predictions, or the output of zero_shot_motor_predictions),"
    echo "or a folder that has a test_predictions folder in it (e.g. <cohort>/<model>/logistic)."
    echo ""
    echo "Options:"
    echo "  --recursive        Also evaluate every predictions folder found below each PATH: a folder named"
    echo "                     test_predictions, or one with a predictions.parquet in it"
    echo "  --output_dir=DIR   Where to write the results (only with a single predictions folder)."
    echo "                     Default: next to the predictions: the parent of a folder named test_predictions,"
    echo "                     else <folder>_evaluation"
    echo ""
    echo "Exits with an error if any evaluation failed; the others are still run."
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/cohort/model/logistic_global"
    echo "  $0 --recursive /path/to/outputs"
    exit 1
}

RECURSIVE="false"
OUTPUT_DIR=""
PATHS=""

for arg in "$@"; do
    case "$arg" in
        --recursive) RECURSIVE="true" ;;
        --output_dir=*) OUTPUT_DIR="${arg#*=}" ;;
        -h|--help) usage ;;
        --*)
            echo "Unknown option: $arg"
            usage
            ;;
        *) PATHS="$PATHS
$arg" ;;
    esac
done

if [ -z "$PATHS" ]; then
    usage
fi

# A predictions folder has the parquet files of the predictions in it
has_predictions() {
    [ -n "$(find "$1" -maxdepth 1 -name '*.parquet' 2>/dev/null | head -n 1)" ]
}

# The predictions folders that a path stands for, one per line
predictions_dirs() {
    path="${1%/}"
    if [ "$RECURSIVE" = "true" ]; then
        find "$path" -type d \( -name test_predictions -o -exec sh -c 'test -f "$1/predictions.parquet"' sh {} \; \) -print | sort
    elif has_predictions "$path"; then
        echo "$path"
    elif [ -d "$path/test_predictions" ]; then
        echo "$path/test_predictions"
    fi
}

# Where the results of a predictions folder go
output_dir_for() {
    dir="${1%/}"
    if [ -n "$OUTPUT_DIR" ]; then
        echo "${OUTPUT_DIR%/}"
    elif [ "$(basename "$dir")" = "test_predictions" ]; then
        dirname "$dir"
    else
        echo "${dir}_evaluation"
    fi
}

FOUND=0
FAILED=0
LIST=$(mktemp)
echo "$PATHS" | while IFS= read -r path; do
    [ -z "$path" ] && continue
    if [ ! -d "$path" ]; then
        echo "ERROR: $path is not a folder" >&2
        echo "MISSING" >> "$LIST"
        continue
    fi
    found=$(predictions_dirs "$path")
    if [ -z "$found" ]; then
        echo "ERROR: no predictions in $path (no parquet files, and no test_predictions folder; try --recursive)" >&2
        echo "MISSING" >> "$LIST"
        continue
    fi
    echo "$found" >> "$LIST"
done

while IFS= read -r predictions_dir; do
    [ -z "$predictions_dir" ] && continue
    if [ "$predictions_dir" = "MISSING" ]; then
        FAILED=$((FAILED + 1))
        continue
    fi
    FOUND=$((FOUND + 1))
done < "$LIST"

if [ -n "$OUTPUT_DIR" ] && [ "$FOUND" -gt 1 ]; then
    echo "ERROR: --output_dir needs a single predictions folder, but $FOUND were found" >&2
    rm -f "$LIST"
    exit 1
fi

while IFS= read -r predictions_dir; do
    [ -z "$predictions_dir" ] && continue
    [ "$predictions_dir" = "MISSING" ] && continue
    results_dir=$(output_dir_for "$predictions_dir")
    mkdir -p "$results_dir"
    echo "meds-evaluation: $predictions_dir -> $results_dir"
    if meds-evaluation-cli predictions_path="$predictions_dir" output_dir="$results_dir/"; then
        if [ -f "$results_dir/results.json" ]; then
            cat "$results_dir/results.json"
            echo ""
        fi
    else
        echo "ERROR: meds-evaluation failed for $predictions_dir" >&2
        FAILED=$((FAILED + 1))
    fi
done < "$LIST"

rm -f "$LIST"
if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED evaluation(s) failed" >&2
    exit 1
fi
