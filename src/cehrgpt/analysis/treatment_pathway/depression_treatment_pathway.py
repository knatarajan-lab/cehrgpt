#!/usr/bin/env python
"""
Depression Treatment Pathways Study Script.

Specialized script for analyzing depression treatment pathways using OMOP CDM data.
Uses the generalized treatment_pathway module with depression-specific parameters.
"""

import argparse
import sys

from cehrgpt.analysis.treatment_pathway.treatment_pathway import main as treatment_main

# Depression-specific configuration
DEPRESSION_CONFIG = {
    "study_name": "DEPRESSION",
    "drug_concepts": [21604686, 21500526],
    "target_conditions": [440383],
    "exclusion_conditions": [444094, 432876, 435783],
}


def parse_depression_arguments():
    """Parse command line arguments specific to depression study."""
    parser = argparse.ArgumentParser(
        description="Analyze depression treatment pathways using OMOP CDM data"
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
        default="Depression_Treatment_Pathways",
        help="Spark application name (default: Depression_Treatment_Pathways)",
    )

    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL (default: local[*])",
    )

    return parser.parse_args()


def main():
    """Main function that overwrites sys.argv and calls the generalized treatment pathway main."""
    depression_args = parse_depression_arguments()

    print("Depression Treatment Pathways Analysis")
    print("=" * 50)
    print(f"Study Configuration:")
    print(f"  - Study Name: {DEPRESSION_CONFIG['study_name']}")
    print(f"  - Drug Concepts: {DEPRESSION_CONFIG['drug_concepts']}")
    print(f"  - Target Conditions: {DEPRESSION_CONFIG['target_conditions']}")
    print(f"  - Exclusion Conditions: {DEPRESSION_CONFIG['exclusion_conditions']}")
    print(f"  - OMOP Folder: {depression_args.omop_folder}")
    print(f"  - Output Folder: {depression_args.output_folder}")
    print("=" * 50)

    # Overwrite sys.argv with the generalized treatment pathway arguments
    sys.argv = [
        "treatment_pathway.py",
        "--omop-folder",
        depression_args.omop_folder,
        "--output-folder",
        depression_args.output_folder,
        "--drug-concepts",
        ",".join(map(str, DEPRESSION_CONFIG["drug_concepts"])),
        "--target-conditions",
        ",".join(map(str, DEPRESSION_CONFIG["target_conditions"])),
        "--exclusion-conditions",
        ",".join(map(str, DEPRESSION_CONFIG["exclusion_conditions"])),
        "--app-name",
        depression_args.app_name,
        "--spark-master",
        depression_args.spark_master,
        "--study-name",
        DEPRESSION_CONFIG["study_name"],
    ]
    # Parse arguments using the generalized script's parser and call its main function
    treatment_main()


if __name__ == "__main__":
    main()
