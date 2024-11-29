from datetime import date
from typing import Any, Dict, List, Optional

import cehrbert_data.cohorts.spark_app_base
from cehrbert_data.cohorts.query_builder import QueryBuilder, QuerySpec
from cehrbert_data.cohorts.spark_app_base import BaseCohortBuilder
from cehrbert_data.utils.spark_utils import preprocess_domain_table


def custom_instantiate_dependencies(spark, input_folder, dependency_list):
    dependency_dict = dict()
    for domain_table_name in dependency_list:
        table = preprocess_domain_table(spark, input_folder, domain_table_name)
        table.createOrReplaceTempView(domain_table_name)
        dependency_dict[domain_table_name] = table
    return dependency_dict


# Monkeypatch the function
cehrbert_data.cohorts.spark_app_base.instantiate_dependencies = (
    custom_instantiate_dependencies
)


class OmopTableBuilder(BaseCohortBuilder):
    cohort_required_columns = []

    def __init__(
        self,
        query_builder: QueryBuilder,
        input_folder: str,
        output_folder: str,
        continue_job: bool = False,
    ):
        super().__init__(
            query_builder=query_builder,
            input_folder=input_folder,
            output_folder=output_folder,
            date_lower_bound="1900-01-01",
            date_upper_bound=date.today().strftime("%Y-%m-%d"),
            age_lower_bound=0,
            age_upper_bound=200,
            prior_observation_period=0,
            post_observation_period=0,
            continue_job=continue_job,
        )

    def build(self):
        # Check whether the cohort has been generated
        if self._continue_job and self.cohort_exists():
            return self
        cohort = self.create_cohort()
        cohort.write.mode("overwrite").parquet(self._output_data_folder)
        return self

    @staticmethod
    def create_omop_query_builder(
        input_folder: str,
        output_folder: str,
        table_name: str,
        query_template: str,
        dependency_list: List[str],
        query_parameters: Optional[Dict[str, Any]] = None,
        continue_job: bool = False,
    ):

        if query_parameters is None:
            query_parameters = dict()
        query = QuerySpec(
            table_name=table_name,
            query_template=query_template,
            parameters=query_parameters,
        )
        table_query_builder = QueryBuilder(
            cohort_name=table_name,
            dependency_list=dependency_list,
            query=query,
            ancestor_table_specs=[],
        )

        return OmopTableBuilder(
            query_builder=table_query_builder,
            input_folder=input_folder,
            output_folder=output_folder,
            continue_job=continue_job,
        )
