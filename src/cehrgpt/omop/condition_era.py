from cehrgpt.omop.omop_argparse import create_omop_argparse
from cehrgpt.omop.omop_table_builder import OmopTableBuilder
from cehrgpt.omop.queries.condition_era import CONDITION_ERA_QUERY

CONDITION_ERA = "condition_era"


def main(args):
    OmopTableBuilder.create_omop_query_builder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        continue_job=args.continue_job,
        table_name=CONDITION_ERA,
        query_template=CONDITION_ERA_QUERY,
        dependency_list=["condition_occurrence"],
    ).build()


if __name__ == "__main__":
    main(create_omop_argparse().parse_args())
