import dataclasses
from typing import Optional


@dataclasses.dataclass
class CehrGPTArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    expand_tokenizer: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to expand the tokenizer for fine-tuning."
        },
    )
