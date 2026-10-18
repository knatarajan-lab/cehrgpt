# Adapted from cehrbert.runners.hf_runner_argument_dataclass (MIT License, same copyright holder as cehrgpt).
"""The model arguments of the runners. The data arguments are in meds_to_cehr.arguments."""

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class FineTuneModelType(Enum):
    POOLING = "pooling"
    LSTM = "lstm"


@dataclasses.dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune,.

    or train from scratch.
    """

    model_name_or_path: Optional[str] = dataclasses.field(
        metadata={
            "help": (
                "The model checkpoint for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = dataclasses.field(
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    early_stopping_patience: Optional[int] = dataclasses.field(
        default=1,
        metadata={
            "help": "stop training when the specified metric worsens " "for `early_stopping_patience` evaluation calls."
        },
    )
    cache_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded " "from huggingface.co"},
    )
    use_auth_token: bool = dataclasses.field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. "
            "Please use `token` instead."
        },
    )
    trust_remote_code: bool = dataclasses.field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    hidden_size: Optional[int] = dataclasses.field(
        default=128,
        metadata={"help": "The embedding and hidden size for the transformer block"},
    )
    num_hidden_layers: Optional[int] = dataclasses.field(
        default=6,
        metadata={"help": "The number of layers used in the transformer model"},
    )
    n_head: Optional[int] = dataclasses.field(
        default=8, metadata={"help": "The number of heads in Multi-Head Attention"}
    )
    max_position_embeddings: Optional[int] = dataclasses.field(
        default=512,
        metadata={"help": "The maximum length of the sequence allowed for the transformer model"},
    )
    finetune_model_type: Literal[FineTuneModelType.POOLING.value, FineTuneModelType.LSTM.value] = dataclasses.field(
        default=FineTuneModelType.POOLING.value,
        metadata={
            "help": "The finetune model type to choose from",
            "choices": f"choices={[e.value for e in FineTuneModelType]}",
        },
    )
    use_lora: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "The flag to indicate whether or not to use the Lora adapter for finetuning"},
    )
    lora_rank: Optional[int] = dataclasses.field(
        default=16, metadata={"help": "Lora attention dimension (the “rank”)."}
    )
    lora_alpha: Optional[int] = dataclasses.field(
        default=16, metadata={"help": "The alpha parameter for Lora scaling."}
    )
    target_modules: Optional[List[str]] = dataclasses.field(
        default_factory=lambda: ["query", "value"],
        metadata={
            "help": "The names of the modules to apply the adapter to. If this is specified, only the modules with the "
            "specified names will be replaced. When passing a string, a regex match will be performed. When "
            "passing a list of strings, either an exact match will be performed or it is checked if the name "
            "of the module ends with any of the passed strings. If this is specified as ‘all-linear’, "
            "then all linear/Conv1D modules are chosen, excluding the output layer. If this is not specified, "
            "modules will be chosen according to the model architecture. If the architecture is not known, "
            "an error will be raised — in this case, you should specify the target modules manually."
        },
    )
    lora_dropout: Optional[float] = dataclasses.field(
        default=0.1, metadata={"help": "The dropout probability for Lora layers"}
    )
    exclude_position_ids: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "Whether or not to exclude position ids from the transformer model"},
    )
    include_values: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "Whether or not to include values into the model"},
    )
    use_sub_time_tokenization: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "Whether or not to decompose the time interval into year/month/day"},
    )
    include_value_prediction: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "Whether or not to include value prediction head for cehrbert"},
    )
    include_ttv_prediction: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "Whether or not to include the time to visit prediction"},
    )
    time_token_loss_weight: Optional[float] = dataclasses.field(
        default=1.0, metadata={"help": "The weight of the time token loss"}
    )
    time_to_visit_loss_weight: Optional[float] = dataclasses.field(
        default=1.0, metadata={"help": "The weight of the time to visit loss"}
    )
    attn_implementation: Literal["eager", "sdpa"] = dataclasses.field(
        default="eager",
        metadata={
            "help": "The type of attention implementation to use",
            "choices": f"choices={['eager', 'sdpa']}",
        },
    )

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
