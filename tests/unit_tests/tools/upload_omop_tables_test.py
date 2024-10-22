import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cehrgpt.tools.upload_omop_tables import upload_omop_tables


class TestOMOPUpload(unittest.TestCase):

    @patch("src.cehrgpt.tools.upload_omop_tables.f")  # Mock Spark SQL functions
    @patch("src.cehrgpt.tools.upload_omop_tables.SparkSession")  # Mock SparkSession
    def test_upload_omop_tables(self, mock_spark, mock_f):
        # Mock the Spark session and DataFrame
        mock_df = MagicMock()
        mock_spark.read.format.return_value.load.return_value = mock_df

        # Mock columns for casting
        mock_df.columns = ["concept_id", "measurement_date"]

        # Mock `f.col()` and `f.lit()` to return other mocks
        mock_col = MagicMock()
        mock_lit = MagicMock()

        # Mock comparison operations on these mock objects
        mock_col.__gt__.return_value = (
            MagicMock()
        )  # For `f.col(timestamp_column) > f.lit(...)`
        mock_col.__lt__.return_value = (
            MagicMock()
        )  # For `f.col(timestamp_column) < f.lit(...)`

        mock_f.col.return_value = mock_col
        mock_f.lit.return_value = mock_lit

        # Mock the filter and withColumn methods to return the DataFrame itself
        mock_df.filter.return_value = mock_df
        mock_df.withColumn.return_value = mock_df

        # Call the upload_omop_tables function (not main)
        upload_omop_tables(
            mock_spark,
            Path("measurement"),
            {
                "base_url": "jdbc:postgresql://localhost:5432/testdb",
                "user": "test_user",
                "password": "test_password",
            },
        )

        # Verify that the DataFrame was read correctly and filtered
        mock_spark.read.format.assert_called_with("parquet")
        mock_spark.read.format().load.assert_called_with("measurement/")
        mock_df.filter.assert_any_call(
            mock_f.col("measurement_date") > mock_f.lit("1900-01-01").cast("date")
        )
        mock_df.filter.assert_any_call(
            mock_f.col("measurement_date") < mock_f.lit("9999-01-01").cast("date")
        )

        # Verify that the columns were cast correctly
        mock_df.withColumn.assert_any_call(
            "concept_id", mock_f.col("concept_id").cast("integer")
        )
        mock_df.withColumn.assert_any_call(
            "measurement_date", mock_f.col("measurement_date").cast("date")
        )

        # Verify that the DataFrame was written to the database
        mock_df.repartition.assert_called_with(10)
        mock_df.repartition().write.format.assert_called_with("jdbc")
        mock_df.repartition().write.format().options.assert_called_with(
            url="jdbc:postgresql://localhost:5432/testdb",
            dbtable="measurement",
            user="test_user",
            password="test_password",
            batchsize=200000,
            queryTimeout=500,
        )
        mock_df.repartition().write.format().options().mode.assert_called_with(
            "overwrite"
        )
        mock_df.repartition().write.format().options().mode().save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
