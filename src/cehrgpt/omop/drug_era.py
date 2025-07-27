from cehrgpt.omop.omop_argparse import create_omop_argparse
from cehrgpt.omop.omop_table_builder import OmopTableBuilder
from cehrgpt.omop.queries.drug_era import DRUG_ERA_QUERY

DRUG_ERA = "drug_era"


def main(args):
    OmopTableBuilder.create_omop_query_builder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        continue_job=args.continue_job,
        table_name=DRUG_ERA,
        query_template=DRUG_ERA_QUERY,
        dependency_list=["condition_occurrence", "concept_ancestor", "concept"],
    ).build()


if __name__ == "__main__":
    main(create_omop_argparse().parse_args())
