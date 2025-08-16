#!/usr/bin/env python
"""
HTN Treatment Pathways Study Script.

Specialized script for analyzing hypertension treatment pathways using OMOP CDM data.
Uses the generalized treatment_pathway module with HTN-specific parameters.
"""

import argparse
import sys

from cehrgpt.analysis.treatment_pathway.treatment_pathway import main as treatment_main

# HTN-specific configuration
HTN_CONFIG = {
    "study_name": "HTN",
    "drug_concepts": [21600381, 21601461, 21601560, 21601664, 21601744, 21601782],
    "target_conditions": [316866],
    "exclusion_conditions": [444094],
}


def parse_htn_arguments():
    """Parse command line arguments specific to HTN study."""
    parser = argparse.ArgumentParser(
        description="Analyze hypertension treatment pathways using OMOP CDM data"
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
        default="HTN_Treatment_Pathways",
        help="Spark application name (default: HTN_Treatment_Pathways)",
    )

    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )

    return parser.parse_args()


def main():
    """Main function that overwrites sys.argv and calls the generalized treatment pathway main."""
    htn_args = parse_htn_arguments()

    print("HTN Treatment Pathways Analysis")
    print("=" * 50)
    print(f"Study Configuration:")
    print(f"  - Study Name: {HTN_CONFIG['study_name']}")
    print(f"  - Drug Concepts: {HTN_CONFIG['drug_concepts']}")
    print(f"  - Target Conditions: {HTN_CONFIG['target_conditions']}")
    print(f"  - Exclusion Conditions: {HTN_CONFIG['exclusion_conditions']}")
    print(f"  - OMOP Folder: {htn_args.omop_folder}")
    print(f"  - Output Folder: {htn_args.output_folder}")
    print("=" * 50)

    # Overwrite sys.argv with the generalized treatment pathway arguments
    sys.argv = [
        "treatment_pathway.py",
        "--omop-folder",
        htn_args.omop_folder,
        "--output-folder",
        htn_args.output_folder,
        "--drug-concepts",
        ",".join(map(str, HTN_CONFIG["drug_concepts"])),
        "--target-conditions",
        ",".join(map(str, HTN_CONFIG["target_conditions"])),
        "--exclusion-conditions",
        ",".join(map(str, HTN_CONFIG["exclusion_conditions"])),
        "--app-name",
        htn_args.app_name,
        "--spark-master",
        htn_args.spark_master,
        "--study-name",
        HTN_CONFIG["study_name"],
    ]
    # Parse arguments using the generalized script's parser and call its main function
    treatment_main()


if __name__ == "__main__":
    main()
