import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class CehrGPTArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    continue_pretrain: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to continue to pretrain cehrgpt on the new dataset"
        },
    )
    retrain_with_full: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to retrain the model on the full set after early stopping"
        },
    )
    expand_tokenizer: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to expand the tokenizer for fine-tuning."
        },
    )
    hyperparameter_tuning_percentage: Optional[float] = dataclasses.field(
        default=0.1,
        metadata={
            "help": "The percentage of the train/val will be use for hyperparameter tuning."
        },
    )
    n_trials: Optional[int] = dataclasses.field(
        default=10,
        metadata={
            "help": "The number of trails will be use for hyperparameter tuning."
        },
    )
    hyperparameter_tuning: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "A flag to indicate if we want to do hyperparameter tuning."},
    )
    hyperparameter_batch_sizes: Optional[List[int]] = dataclasses.field(
        default_factory=lambda: [4, 8, 16],
        metadata={"help": "Hyperparameter search batch sizes"},
    )
    hyperparameter_num_train_epochs: Optional[List[int]] = dataclasses.field(
        default_factory=lambda: [10],
        metadata={"help": "Hyperparameter search num_train_epochs"},
    )
    lr_low: Optional[float] = dataclasses.field(
        default=1e-5,
        metadata={
            "help": "The lower bound of the learning rate range for hyperparameter tuning."
        },
    )
    lr_high: Optional[float] = dataclasses.field(
        default=5e-5,
        metadata={
            "help": "The upper bound of the learning rate range for hyperparameter tuning."
        },
    )
    weight_decays_low: Optional[float] = dataclasses.field(
        default=1e-3,
        metadata={
            "help": "The lower bound of the weight decays range for hyperparameter tuning."
        },
    )
    weight_decays_high: Optional[float] = dataclasses.field(
        default=1e-2,
        metadata={
            "help": "The upper bound of the weight decays range for hyperparameter tuning."
        },
    )
    causal_sfm: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether the GPT conforms to the causal Standard Fairness Model"
        },
    )
    demographics_size: Optional[int] = dataclasses.field(
        default=4,
        metadata={
            "help": "The number of demographics tokens in the patient sequence "
            "It defaults to 4, assuming the demographics tokens follow this pattern [Year][Age][Gender][Race]"
        },
    )
    drop_long_sequences: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "The lower bound of the learning rate range for hyperparameter tuning."
        },
    )
    lab_token_penalty: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to use lab token loss penalty."
        },
    )
    lab_token_loss_weight: Optional[float] = dataclasses.field(
        default=1.0,
        metadata={"help": "lab_token_loss_weight penalty co-efficient"},
    )
    entropy_penalty: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "A flag to indicate whether we want to use entropy penalty."},
    )
    entropy_penalty_alpha: Optional[float] = dataclasses.field(
        default=0.01,
        metadata={"help": "Entropy penalty co-efficient"},
    )
