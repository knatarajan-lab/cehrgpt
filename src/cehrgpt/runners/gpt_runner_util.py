import dataclasses
import os
import sys
from typing import Optional, Tuple

from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    ModelArguments,
)
from transformers import HfArgumentParser, TrainingArguments
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

LOG = logging.get_logger("transformers")


def resolve_attn_implementation(
    backbone: str = "gpt2", requested: Optional[str] = None
) -> str:
    """
    Choose the attention implementation for a backbone.

    An explicit `requested` value always wins. Otherwise:

      * gpt2 uses flash attention 2 when it is installed, else eager. Only these two are
        implemented for that backbone - its attention module ignores `_attn_implementation`
        apart from the flash check, so asking for sdpa there would silently run eager math.
      * qwen2 uses sdpa. Eager attention materialises a `(heads, L, L)` score matrix, which
        is untenable when sample packing pushes L to `max_tokens_per_batch` (~108 GiB of
        activations at L=16384 for a 6-layer model). Its flash path exists but has not been
        validated on real hardware.

    Args:
        backbone: The `backbone` field of the model config ("gpt2" or "qwen2").
        requested: An explicit override, typically
            `CehrGPTArguments.force_attn_implementation`.

    Returns:
        One of "flash_attention_2", "sdpa" or "eager".
    """
    if requested:
        if requested == "flash_attention_2" and not is_flash_attn_2_available():
            raise ValueError(
                "flash_attention_2 was requested but flash attention 2 is not available "
                "in this environment."
            )
        if backbone == "gpt2" and requested == "sdpa":
            raise ValueError(
                "The gpt2 backbone does not implement sdpa; its attention always runs "
                "eager. Use 'eager' or 'flash_attention_2', or switch to backbone=qwen2."
            )
        if backbone == "qwen2" and requested == "flash_attention_2":
            LOG.warning(
                "Using flash attention 2 with the qwen2 backbone. This path is not yet "
                "validated; verify that packed sequences remain isolated before trusting "
                "the results."
            )
        if backbone == "qwen2" and requested == "eager":
            LOG.warning(
                "Using eager attention with the qwen2 backbone. Memory grows with the "
                "square of the packed sequence length; prefer sdpa when sample packing is "
                "enabled."
            )
        return requested
    if backbone == "qwen2":
        LOG.info("Using sdpa attention for the qwen2 backbone.")
        return "sdpa"
    return "flash_attention_2" if is_flash_attn_2_available() else "eager"


def read_backbone(model_name_or_path: str) -> str:
    """
    Read the `backbone` field from a saved CEHR-GPT checkpoint.

    Used by the fine-tuning, linear-probing and generation entry points, which get their
    architecture from a checkpoint rather than from command-line arguments. Falls back to
    "gpt2" when the config cannot be read, which matches the config default and the
    behaviour of every checkpoint saved before the qwen2 backbone existed.
    """
    try:
        return getattr(
            CEHRGPTConfig.from_pretrained(os.path.expanduser(model_name_or_path)),
            "backbone",
            "gpt2",
        )
    except Exception as error:
        LOG.warning(
            "Could not read the backbone from %s (%s); assuming gpt2.",
            model_name_or_path,
            error,
        )
        return "gpt2"


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
