import os

import polars as pl

from cehrgpt.gpt_utils import extract_time_interval_in_days, is_att_token


def main(args):
    dataset = pl.read_parquet(os.path.join(args.input_dir, "*.parquet"))
    time_token_frequency_df = (
        dataset.select(pl.col("concept_ids").explode().alias("concept_id"))
        .filter(pl.col("concept_id").map_elements(is_att_token))
        .with_columns(
            pl.col("concept_id")
            .map_elements(extract_time_interval_in_days)
            .alias("time_interval")
        )
    )
    results = time_token_frequency_df.select(
        pl.mean("time_interval").alias("mean"), pl.std("time_interval").alias("std")
    ).to_dicts()[0]
    print(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EHR Irregularity analysis")
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        action="store",
        help="The path for where the input data folder",
        required=True,
    )
    main(parser.parse_args())
