#!/usr/bin/env python
"""
Diabetes Treatment Pathways Study Script.

Specialized script for analyzing diabetes treatment pathways using OMOP CDM data.
Uses the generalized treatment_pathway module with diabetes-specific parameters.
"""

import argparse
import sys

from cehrgpt.analysis.treatment_pathway.treatment_pathway import main as treatment_main

# Diabetes-specific configuration
DIABETES_CONFIG = {
    "study_name": "DIABETES",
    "drug_concepts": [21600712, 21500148],
    "target_conditions": [201820],
    "exclusion_conditions": [444094, 35506621],
}


def parse_diabetes_arguments():
    """Parse command line arguments specific to diabetes study."""
    parser = argparse.ArgumentParser(
        description="Analyze diabetes treatment pathways using OMOP CDM data"
    )

    parser.add_argument(
        "--omop-folder",
        required=True,
        help="Path to OMOP CDM data folder containing parquet files",
    )

    parser.add_argument(
        "--output-folder", required=True, help="Output folder for results"
    )

    parser.add_argument(
        "--app-name",
        default="Diabetes_Treatment_Pathways",
        help="Spark application name (default: Diabetes_Treatment_Pathways)",
    )

    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )

    return parser.parse_args()


def main():
    """Main function that overwrites sys.argv and calls the generalized treatment pathway main."""
    diabetes_args = parse_diabetes_arguments()

    print("Diabetes Treatment Pathways Analysis")
    print("=" * 50)
    print(f"Study Configuration:")
    print(f"  - Study Name: {DIABETES_CONFIG['study_name']}")
    print(f"  - Drug Concepts: {DIABETES_CONFIG['drug_concepts']}")
    print(f"  - Target Conditions: {DIABETES_CONFIG['target_conditions']}")
    print(f"  - Exclusion Conditions: {DIABETES_CONFIG['exclusion_conditions']}")
    print(f"  - OMOP Folder: {diabetes_args.omop_folder}")
    print(f"  - Output Folder: {diabetes_args.output_folder}")
    print("=" * 50)

    # Overwrite sys.argv with the generalized treatment pathway arguments
    sys.argv = [
        "treatment_pathway.py",
        "--omop-folder",
        diabetes_args.omop_folder,
        "--output-folder",
        diabetes_args.output_folder,
        "--drug-concepts",
        ",".join(map(str, DIABETES_CONFIG["drug_concepts"])),
        "--target-conditions",
        ",".join(map(str, DIABETES_CONFIG["target_conditions"])),
        "--exclusion-conditions",
        ",".join(map(str, DIABETES_CONFIG["exclusion_conditions"])),
        "--app-name",
        diabetes_args.app_name,
        "--spark-master",
        diabetes_args.spark_master,
        "--study-name",
        DIABETES_CONFIG["study_name"],
    ]
    # Parse arguments using the generalized script's parser and call its main function
    treatment_main()


if __name__ == "__main__":
    main()
