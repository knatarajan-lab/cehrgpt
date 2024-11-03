import argparse
import os
import subprocess
from pathlib import Path

import yaml


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for benchmarking CEHRGPT on ehrshot cohorts"
    )
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--base_yaml_file", required=True)
    parser.add_argument("--output_folder", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = create_arg_parser()

    with open(args.base_yaml_file, "rb") as stream:
        base_config = yaml.safe_load(stream)

    for individual_cohort in os.listdir(args.cohort_dir):
        if individual_cohort.endswith("/"):
            individual_cohort = individual_cohort[:-1]
        cohort_name = os.path.basename(individual_cohort)
        individual_output = os.path.join(args.output_folder, cohort_name)
        if os.path.exists(individual_output):
            continue
        Path(individual_output).mkdir(parents=True, exist_ok=True)
        base_config.data_folder = os.path.join(individual_cohort, "train")
        base_config.test_data_folder = os.path.join(individual_cohort, "test")
        base_config.output_dir = individual_output

        # Write YAML data to a file
        config_path = os.path.join(individual_output, "config.yaml")
        with open(config_path, "w") as yaml_file:
            yaml.dump(base_config, yaml_file, default_flow_style=False)

        command = [
            "python",
            "-u",
            "-m",
            "cehrgpt.runners.hf_cehrgpt_finetune_runner",
            config_path,
        ]

        # Start the subprocess
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as process:
            # Read stdout and stderr line-by-line in real time
            for line in process.stdout:
                print(line, end="")  # Print stdout line as it comes
            for error_line in process.stderr:
                print(error_line, end="")  # Print stderr line as it comes
