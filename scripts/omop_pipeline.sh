#!/bin/bash

# Exporting input arguments as environment variables
export PATIENT_SEQUENCE_FOLDER="$1"
export OMOP_FOLDER="$2"
export SOURCE_OMOP_FOLDER="$3"
export PATIENT_SPLITS_FOLDER="$SOURCE_OMOP_FOLDER/patient_splits"

# Echoing the values of the environment variables
echo "PATIENT_SEQUENCE_FOLDER=$PATIENT_SEQUENCE_FOLDER"
echo "OMOP_FOLDER=$OMOP_FOLDER"
echo "SOURCE_OMOP_FOLDER=$SOURCE_OMOP_FOLDER"

# Ensure OMOP_FOLDER exists
if [ ! -d "$OMOP_FOLDER" ]; then
    echo "Creating $OMOP_FOLDER"
    mkdir -p "$OMOP_FOLDER"
fi

# Removing existing OMOP tables
rm -rf $OMOP_FOLDER/{person,visit_occurrence,condition_occurrence,procedure_occurrence,drug_exposure,death,measurement,observation_period,condition_era}

# Removing existing OMOP concept tables
rm -rf $OMOP_FOLDER/{concept,concept_ancestor,concept_relationship}

# Copying OMOP concept tables if they don't already exist
for table in concept concept_relationship concept_ancestor; do
    if [ ! -d "$OMOP_FOLDER/$table" ]; then
        echo "Creating $OMOP_FOLDER/$table"
        cp -r "$SOURCE_OMOP_FOLDER/$table" "$OMOP_FOLDER/$table"
    fi
done

# Reconstructing the OMOP instance from patient sequences
echo "Reconstructing the OMOP instance from patient sequences in $OMOP_FOLDER"
python -m cehrgpt.generation.omop_converter_batch \
  --patient_sequence_path "$PATIENT_SEQUENCE_FOLDER" \
  --output_folder "$OMOP_FOLDER" \
  --concept_path "$OMOP_FOLDER/concept" \
  --buffer_size 1280 \
  --cpu_cores 10

# Create observation_period
echo "Reconstructing observation_period in $OMOP_FOLDER"
python -u -m cehrgpt.omop.observation_period \
  --input_folder "$OMOP_FOLDER" \
  --output_folder "$OMOP_FOLDER" \
  --domain_table_list "condition_occurrence drug_exposure procedure_occurrence measurement"

# Create condition_era
echo "Reconstructing condition_era in $OMOP_FOLDER"
python -u -m cehrgpt.omop.condition_era \
  --input_folder "$OMOP_FOLDER" \
  --output_folder "$OMOP_FOLDER" \
  --domain_table_list "condition_occurrence"
