import argparse

from cehrbert_data.utils.spark_utils import validate_table_names


def create_omop_argparse():
    parser = argparse.ArgumentParser(
        description="Spark application for creating OMOP table"
    )
    parser.add_argument(
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the raw data is",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--continue_job",
        dest="continue_job",
        action="store_true",
    )
    parser.add_argument(
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to download",
        type=validate_table_names,
        required=True,
    )
    return parser
