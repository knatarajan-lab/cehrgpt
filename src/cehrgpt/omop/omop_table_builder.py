import logging
import os
import re
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession

from cehrgpt.omop.query_builder import QueryBuilder, QuerySpec
from cehrgpt.omop.spark_utils import preprocess_domain_table


def validate_folder(folder_path: str):
    if not os.path.exists(folder_path):
        raise FileExistsError(f"{folder_path} does not exist")


def validate_domain_tables(input_folder: str, table_list: List[str]):
    for domain_table_name in table_list:
        parquet_file_path = os.path.join(input_folder, domain_table_name)
        if not os.path.exists(parquet_file_path):
            raise FileExistsError(f"{parquet_file_path} does not exist")


class OmopTableBuilder:
    """
    Builds an OMOP table (parquet) with a Spark SQL query on the tables it depends on.

    Adapted from cehrbert_data.cohorts.spark_app_base.BaseCohortBuilder (MIT License, same copyright holder
    as cehrgpt), without the parts that are about cohorts: the observation period, the demographics and the
    ancestor tables.
    """

    def __init__(
        self,
        query_builder: QueryBuilder,
        input_folder: str,
        output_folder: str,
        continue_job: bool = False,
    ):
        self._query_builder = query_builder
        self._input_folder = input_folder
        self._output_folder = output_folder
        cohort_name = re.sub(
            "[^a-z0-9]+", "_", self._query_builder.get_cohort_name().lower()
        )
        self._output_data_folder = os.path.join(self._output_folder, cohort_name)
        self._continue_job = continue_job

        self.get_logger().info(
            f"query_builder: {query_builder}\n"
            f"input_folder: {input_folder}\n"
            f"output_folder: {output_folder}\n"
            f"continue_job: {continue_job}\n"
        )
        if self._query_builder.get_ancestor_table_specs():
            raise NotImplementedError("Ancestor tables are not supported")

        validate_folder(self._input_folder)
        validate_folder(self._output_folder)
        validate_domain_tables(
            self._input_folder, self._query_builder.get_dependency_list()
        )

        self.spark = SparkSession.builder.appName(
            f"Generate {self._query_builder.get_cohort_name()}"
        ).getOrCreate()
        self._dependency_dict = self._instantiate_dependencies()

    def _instantiate_dependencies(self) -> Dict[str, DataFrame]:
        dependency_dict = dict()
        for domain_table_name in self._query_builder.get_dependency_list():
            table = preprocess_domain_table(
                self.spark, self._input_folder, domain_table_name
            )
            table.createOrReplaceTempView(domain_table_name)
            dependency_dict[domain_table_name] = table
        return dependency_dict

    def create_table(self) -> DataFrame:
        """Run the queries of the query builder, the main query last, and return its result."""

        def run(query: QuerySpec) -> DataFrame:
            table = self.spark.sql(query.query_template.format(**query.parameters))
            table.createOrReplaceGlobalTempView(query.table_name)
            return table

        # Build the dependencies for the main query to use if the dependency_queries are available
        for dependency_query in self._query_builder.get_dependency_queries() or []:
            run(dependency_query)
        if self._query_builder.get_entry_cohort_query():
            run(self._query_builder.get_entry_cohort_query())
        if self._query_builder.get_negative_query():
            run(self._query_builder.get_negative_query())

        main_query = self._query_builder.get_query()
        table = run(main_query)

        # Post process the table if the post_process_queries are available
        for post_query in self._query_builder.get_post_process_queries() or []:
            table = self.spark.sql(
                post_query.query_template.format(**post_query.parameters)
            )
            table.createOrReplaceGlobalTempView(main_query.table_name)
        return table

    def build(self):
        # Check whether the table has been generated
        if self._continue_job and self.table_exists():
            return self
        table = self.create_table()
        table.write.mode("overwrite").parquet(self._output_data_folder)
        return self

    def table_exists(self) -> bool:
        try:
            self.spark.read.parquet(self._output_data_folder)
            return True
        except Exception:
            return False

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

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)
