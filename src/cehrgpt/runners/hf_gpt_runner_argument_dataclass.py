import dataclasses
from typing import Optional


@dataclasses.dataclass
class CehrGPTArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    expand_tokenizer: Optional[bool] = (
        dataclasses.field(
            default=False,
            metadata={
                "help": "A flag to indicate whether we want to expand the tokenizer for fine-tuning."
            },
        ),
    )
    hyperparameter_tuning_percentage: Optional[float] = (
        dataclasses.field(
            default=0.1,
            metadata={
                "help": "The percentage of the train/val will be use for hyperparameter tuning."
            },
        ),
    )
    n_trials: Optional[int] = (
        dataclasses.field(
            default=10,
            metadata={
                "help": "The number of trails will be use for hyperparameter tuning."
            },
        ),
    )
    early_stopping_patience: Optional[int] = (
        dataclasses.field(
            default=1,
            metadata={"help": "The early_stopping_patience for overfitting"},
        ),
    )
