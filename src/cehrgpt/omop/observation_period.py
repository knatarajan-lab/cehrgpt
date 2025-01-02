import glob
import os.path

from cehrgpt.omop.omop_argparse import create_omop_argparse
from cehrgpt.omop.omop_table_builder import OmopTableBuilder
from cehrgpt.omop.queries.observation_period import (
    OBSERVATION_PERIOD_QUERY,
    OBSERVATION_PERIOD_WITH_MEASUREMENT_QUERY,
)

OBSERVATION_PERIOD = "observation_period"


def main(args):
    include_measurement = (
        len(glob.glob(os.path.join(args.input_folder, "measurement", "*.parquet"))) > 0
    )
    dependency_list = [
        "person",
        "visit_occurrence",
        "condition_occurrence",
        "procedure_occurrence",
        "drug_exposure",
    ]
    if include_measurement:
        dependency_list.append("measurement")

    OmopTableBuilder.create_omop_query_builder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        continue_job=args.continue_job,
        table_name=OBSERVATION_PERIOD,
        query_template=(
            OBSERVATION_PERIOD_WITH_MEASUREMENT_QUERY
            if include_measurement
            else OBSERVATION_PERIOD_QUERY
        ),
        dependency_list=dependency_list,
    ).build()


if __name__ == "__main__":
    main(create_omop_argparse().parse_args())
