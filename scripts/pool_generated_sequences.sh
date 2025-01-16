#!/bin/bash

# Check if sufficient arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_root> <destination_folder>"
    exit 1
fi

# Assigning the first and second argument to variables
SOURCE_ROOT=$1
DESTINATION_FOLDER=$2

# Find and copy parquet files from directories named 'generated_sequences'
find "$SOURCE_ROOT" -path '*/generated_sequences/*.parquet' -exec cp {} "$DESTINATION_FOLDER" \;

echo "Parquet files have been copied successfully."
