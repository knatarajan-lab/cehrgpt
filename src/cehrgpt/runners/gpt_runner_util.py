import dataclasses
import os
import sys
from typing import Tuple

from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    ModelArguments,
)
from transformers import HfArgumentParser, TrainingArguments
from transformers.utils import logging

from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

LOG = logging.get_logger("transformers")


def parse_dynamic_arguments(
    argument_classes: Tuple[dataclasses.dataclass, ...] = (
        DataTrainingArguments,
        ModelArguments,
        TrainingArguments,
    )
) -> Tuple:
    """
    Parses command-line arguments with extended flexibility, allowing for the inclusion of custom argument classes.

    This function utilizes `HfArgumentParser` to parse arguments from command line input, JSON, or YAML files.
    By default, it expects `ModelArguments`, `DataTrainingArguments`, and `TrainingArguments`, but it can be extended
    with additional argument classes through the `argument_classes` parameter, making it suitable
    for various custom setups.

    Parameters:
        argument_classes (Tuple[Type]): A tuple of argument classes to be parsed. Defaults to
        `(ModelArguments, DataTrainingArguments, TrainingArguments)`. Additional argument classes can be specified
        for greater flexibility in configuration.

    Returns:
        Tuple: A tuple of parsed arguments, one for each argument class provided. The order of the returned tuple
        matches the order of the `argument_classes` parameter.

    Raises:
        FileNotFoundError: If the specified JSON or YAML file does not exist.
        json.JSONDecodeError: If there is an error parsing a JSON file.
        yaml.YAMLError: If there is an error parsing a YAML file.
        Exception: For other issues that occur during argument parsing.

    Example usage:
        - Command-line: `python training_script.py --model_name_or_path bert-base-uncased --do_train`
        - JSON file: `python training_script.py config.json`
        - YAML file: `python training_script.py config.yaml`

    Flexibility:
        The function can be customized to include new argument classes as needed:

        Example with a custom argument class:
            ```python
            class CustomArguments:
                # Define custom arguments here
                pass


            custom_args = parse_extended_args(
                (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
            )
            ```
        This example demonstrates how to include additional argument classes
        beyond the defaults for a more tailored setup.
    """
    parser = HfArgumentParser(argument_classes)

    # Check if input is a JSON or YAML file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.expanduser(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.expanduser(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()

    return tuple(args)


def parse_runner_args() -> (
    Tuple[CehrGPTArguments, DataTrainingArguments, ModelArguments, TrainingArguments]
):
    cehrgpt_args, data_args, model_args, training_args = parse_dynamic_arguments(
        (CehrGPTArguments, DataTrainingArguments, ModelArguments, TrainingArguments)
    )
    return cehrgpt_args, data_args, model_args, training_args
