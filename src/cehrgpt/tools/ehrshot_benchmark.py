import argparse
import os
import select  # Import select for monitoring stdout and stderr
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

    for cohort_name in os.listdir(args.cohort_dir):
        if cohort_name.endswith("/"):
            cohort_name = cohort_name[:-1]
        individual_output = os.path.join(args.output_folder, cohort_name)
        if os.path.exists(individual_output):
            continue
        Path(individual_output).mkdir(parents=True, exist_ok=True)
        base_config["data_folder"] = os.path.join(args.cohort_dir, cohort_name, "train")
        base_config["test_data_folder"] = os.path.join(
            args.cohort_dir, cohort_name, "test"
        )
        base_config["output_dir"] = individual_output

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
            while True:
                # Use select to wait for either stdout or stderr to have data
                reads = [process.stdout.fileno(), process.stderr.fileno()]
                ret = select.select(reads, [], [])

                # Read data from stdout and stderr as it becomes available
                for fd in ret[0]:
                    if fd == process.stdout.fileno():
                        line = process.stdout.readline()
                        if line:
                            print(line, end="")
                    elif fd == process.stderr.fileno():
                        line = process.stderr.readline()
                        if line:
                            print(line, end="")

                # Break loop when process finishes
                if process.poll() is not None:
                    break
