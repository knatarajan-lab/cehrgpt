# import os
# from collections import defaultdict
# from functools import partial
# from typing import Dict, Tuple
#
# import polars as pl
# from transformers import TrainingArguments
# from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
# from cehrgpt.runners.gpt_runner_util import parse_dynamic_arguments
# from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTGRPOArguments
#
#
# def main():
#     cehrgpt_grpo_args, model_args, training_args = parse_dynamic_arguments(
#         (CehrGPTGRPOArguments, ModelArguments, TrainingArguments)
#     )
#     cehrgpt_grpo_args.real_data_dir
#
#
# if __name__ == "__main__":
#     pass
