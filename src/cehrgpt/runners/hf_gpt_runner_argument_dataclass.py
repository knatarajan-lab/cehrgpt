import dataclasses
from typing import List, Literal, Optional

from cehrgpt.models.gpt2 import ACT2FN


@dataclasses.dataclass
class CehrGPTArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    tokenized_full_dataset_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "The path to the tokenized dataset created for the full population"
        },
    )
    activation_function: Literal[tuple(ACT2FN.keys())] = dataclasses.field(
        default="gelu_new",
        metadata={"help": "The activation function to use"},
    )
    decoder_mlp: Literal["GPT2MLP", "LlamaMLP"] = dataclasses.field(
        default="GPT2MLP",
        metadata={"help": "The decoder MLP architecture"},
    )
    include_inpatient_hour_token: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "Include inpatient hour token"},
    )
    include_demographics: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to always include the demographics for the long sequences that are longer than the model context window."
        },
    )
    continue_pretrain: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to continue to pretrain cehrgpt on the new dataset"
        },
    )
    pretrained_embedding_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "The path to the concept pretrained embeddings"},
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
    few_shot_predict: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to use a few shots to train the model"
        },
    )
    n_shots: Optional[int] = dataclasses.field(
        default=128,
        metadata={"help": "The number of examples from the training set."},
    )
    hyperparameter_tuning: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "A flag to indicate if we want to do hyperparameter tuning."},
    )
    hyperparameter_tuning_is_grid: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "A flag to indicate if we want to do hyperparameter tuning."},
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
    hyperparameter_batch_sizes: Optional[List[int]] = dataclasses.field(
        default_factory=lambda: [4, 8, 16],
        metadata={"help": "Hyperparameter search batch sizes"},
    )
    hyperparameter_num_train_epochs: Optional[List[int]] = dataclasses.field(
        default_factory=lambda: [10],
        metadata={"help": "Hyperparameter search num_train_epochs"},
    )
    hyperparameter_learning_rates: Optional[List[int]] = dataclasses.field(
        default_factory=lambda: [1e-5],
        metadata={"help": "Hyperparameter search learning rates"},
    )
    hyperparameter_weight_decays: Optional[List[int]] = dataclasses.field(
        default_factory=lambda: [1e-2],
        metadata={"help": "Hyperparameter search learning rates"},
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
    next_token_prediction_loss_weight: float = dataclasses.field(
        default=1.0, metadata={"help": "The weight of the next token prediction loss"}
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
    value_prediction_loss_weight: Optional[float] = dataclasses.field(
        default=1.0,
        metadata={"help": "The weight of the value prediction loss"},
    )
    entropy_penalty: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "A flag to indicate whether we want to use entropy penalty."},
    )
    entropy_penalty_alpha: Optional[float] = dataclasses.field(
        default=0.01,
        metadata={"help": "Entropy penalty co-efficient"},
    )
    n_pretrained_embeddings_layers: Optional[int] = dataclasses.field(
        default=2,
        metadata={
            "help": "The number of feed forward layers for transforming pretrained embeddings to internal embeddings"
        },
    )
    meds_repartition: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to repartition the meds train tune sets"
        },
    )
    use_early_stopping: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "A flag to indicate whether we want to use early stopping."},
    )
    early_stopping_threshold: Optional[float] = dataclasses.field(
        default=0.01,
        metadata={
            "help": "A threshold to denote how much the specified metric must improve to satisfy early stopping conditions."
        },
    )
    inner_dim: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "The dimensionality of the hidden layer"},
    )
    apply_rotary: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to use rotary encoder layers"
        },
    )
    sample_packing: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to use sample packing for efficient training."
        },
    )
    max_tokens_per_batch: int = dataclasses.field(
        default=16384, metadata={"help": "Maximum number of tokens in each batch"}
    )
    include_motor_time_to_event: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to include the motor time to events"
        },
    )
    num_motor_tasks: Optional[int] = dataclasses.field(
        default=10000,
        metadata={"help": "The number of max MOTOR tasks"},
    )
    motor_time_to_event_weight: Optional[float] = dataclasses.field(
        default=1.0,
        metadata={"help": "The MOTOR time to event loss weight"},
    )
    motor_num_time_pieces: Optional[int] = dataclasses.field(
        default=8,
        metadata={
            "help": "The number of times each motor_num_time_pieces piece has to be"
        },
    )
    motor_use_ontology: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to use motor_use_ontology"
        },
    )
    motor_sampling_probability: Optional[float] = dataclasses.field(
        default=0.0,
        metadata={"help": "A flag to indicate whether we want to use sample packing"},
    )
    vocab_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "The directory where the concept data is stored."},
    )
    average_over_sequence: bool = dataclasses.field(
        default=False,
        metadata={"help": "Whether or not to average tokens per sequence"},
    )
    apply_entropy_filter: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "A flag to indicate whether we want to use entropy filter."},
    )
    min_prevalence: Optional[float] = dataclasses.field(
        default=1 / 1000,
        metadata={"help": "The min_prevalence to keep the concepts in the tokenizer"},
    )
    class_weights: Optional[List[int]] = dataclasses.field(
        default=None,
        metadata={"help": "The class weights for training"},
    )
    negative_sampling_probability: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "The probability of negative samples will be included in the training data"
        },
    )
    num_of_trajectories_per_sample: Optional[int] = dataclasses.field(
        default=1,
        metadata={"help": "The number of trajectories per sample"},
    )
    generation_input_length: Optional[int] = dataclasses.field(
        default=1024,
        metadata={"help": "The length of the input sequence"},
    )
    generation_max_new_tokens: Optional[int] = dataclasses.field(
        default=1024,
        metadata={"help": "The maximum number of tokens in the generation sequence"},
    )
