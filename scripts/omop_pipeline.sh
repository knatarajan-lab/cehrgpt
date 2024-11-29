#!/bin/bash

# Exporting input arguments as environment variables
export PATIENT_SEQUENCE_FOLDER=$1
export OMOP_FOLDER=$2
export SOURCE_OMOP_FOLDER=$3
export PATIENT_SPLITS_FOLDER=$SOURCE_OMOP_FOLDER/patient_splits

# Echoing the values of the environment variables
echo "PATIENT_SEQUENCE_FOLDER=$PATIENT_SEQUENCE_FOLDER"
echo "OMOP_FOLDER=$OMOP_FOLDER"
echo "SOURCE_OMOP_FOLDER=$SOURCE_OMOP_FOLDER"

# Removing existing OMOP tables
rm -rf $OMOP_FOLDER/person/
rm -rf $OMOP_FOLDER/visit_occurrence/
rm -rf $OMOP_FOLDER/condition_occurrence/
rm -rf $OMOP_FOLDER/procedure_occurrence/
rm -rf $OMOP_FOLDER/drug_exposure/
rm -rf $OMOP_FOLDER/death/
rm -rf $OMOP_FOLDER/measurement/
rm -rf $OMOP_FOLDER/observation_period/
rm -rf $OMOP_FOLDER/condition_era/

# Removing existing OMOP concept tables
rm -rf $OMOP_FOLDER/concept
rm -rf $OMOP_FOLDER/concept_ancestor
rm -rf $OMOP_FOLDER/concept_relationship

# Copying OMOP concept tables if they don't already exist
if [ ! -d $OMOP_FOLDER/concept ]; then
    echo "Creating $OMOP_FOLDER/concept"
    cp -r $SOURCE_OMOP_FOLDER/concept $OMOP_FOLDER/concept
fi

if [ ! -d $OMOP_FOLDER/concept_relationship ]; then
    echo "Creating $OMOP_FOLDER/concept_relationship"
    cp -r $SOURCE_OMOP_FOLDER/concept_relationship $OMOP_FOLDER/concept_relationship
fi

if [ ! -d $OMOP_FOLDER/concept_ancestor ]; then
    echo "Creating $OMOP_FOLDER/concept_ancestor"
    cp -r $SOURCE_OMOP_FOLDER/concept_ancestor $OMOP_FOLDER/concept_ancestor
fi

# Reconstructing the OMOP instance from patient sequences
python -m cehrgpt.generation.omop_converter_batch \
  --patient_sequence_path $PATIENT_SEQUENCE_FOLDER \
  --output_folder $OMOP_FOLDER \
  --concept_path $OMOP_FOLDER/concept \
  --buffer_size 1280 \
  --cpu_cores 10

# Create observation_period
python -u -m cehrgpt.omop.observation_period \
  --input_folder $OMOP_FOLDER \
  --output_folder $OMOP_FOLDER \
  --domain_table_list condition_occurrence drug_exposure procedure_occurrence measurement

# Create condition_era
python -u -m cehrgpt.omop.condition_era \
  --input_folder $OMOP_FOLDER \
  --output_folder $OMOP_FOLDER \
  --domain_table_list condition_occurrence
