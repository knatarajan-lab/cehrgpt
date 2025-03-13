#!/bin/bash

# This script runs various Python simulations and generates plots
# It accepts three parameters: output directory, number of steps, and number of samples

# Check if all arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <n_steps> <n_samples>"
    exit 1
fi

# Assigning command line arguments to variables
OUTPUT_DIR="$1"
N_STEPS="$2"
N_SAMPLES="$3"

# Run time token simulation
python -u -m cehrgpt.simulations.time_token_simulation --output_dir "$OUTPUT_DIR" --n_steps "$N_STEPS" --n_samples "$N_SAMPLES"

# Run time embedding simulation
python -u -m cehrgpt.simulations.time_embedding_simulation --output_dir "$OUTPUT_DIR" --n_steps "$N_STEPS" --n_samples "$N_SAMPLES"

# Generate plots
python -u -m cehrgpt.simulations.generate_plots "$OUTPUT_DIR"
