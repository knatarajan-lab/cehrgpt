import argparse
from enum import Enum


class SamplingStrategy(Enum):
    TopKStrategy = "TopKStrategy"
    TopPStrategy = "TopPStrategy"
    TopMixStrategy = "TopMixStrategy"


def create_inference_base_arg_parser(
    description: str = "Base arguments for cehr-gpt inference",
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tokenizer_folder",
        dest="tokenizer_folder",
        action="store",
        help="The path for your model_folder",
        required=True,
    )
    parser.add_argument(
        "--model_folder",
        dest="model_folder",
        action="store",
        help="The path for your model_folder",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your generated data",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        action="store",
        type=int,
        help="batch_size",
        required=True,
    )
    parser.add_argument(
        "--buffer_size",
        dest="buffer_size",
        action="store",
        type=int,
        default=100,
        help="buffer_size",
        required=False,
    )
    parser.add_argument(
        "--context_window",
        dest="context_window",
        action="store",
        type=int,
        help="The context window of the gpt model",
        required=True,
    )
    parser.add_argument(
        "--min_num_of_concepts",
        dest="min_num_of_concepts",
        action="store",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--sampling_strategy",
        dest="sampling_strategy",
        action="store",
        choices=[e.value for e in SamplingStrategy],
        help="Pick the sampling strategy from the three options top_k, top_p and top_mix",
        required=True,
    )
    parser.add_argument(
        "--top_k",
        dest="top_k",
        action="store",
        default=100,
        type=int,
        help="The number of top concepts to sample",
        required=False,
    )
    parser.add_argument(
        "--top_p",
        dest="top_p",
        action="store",
        default=1.0,
        type=float,
        help="The accumulative probability of top concepts to sample",
        required=False,
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        action="store",
        default=1.0,
        type=float,
        help="The temperature parameter for softmax",
        required=False,
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        action="store",
        default=1.0,
        type=float,
        help="The repetition penalty during decoding",
        required=False,
    )
    parser.add_argument(
        "--num_beams",
        dest="num_beams",
        action="store",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--num_beam_groups",
        dest="num_beam_groups",
        action="store",
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--epsilon_cutoff",
        dest="epsilon_cutoff",
        action="store",
        default=0.0,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--use_bfloat16",
        dest="use_bfloat16",
        action="store_true",
    )
    return parser
