import dataclasses
from typing import List, Literal, Optional

from cehrgpt.models.gpt2 import ACT2FN


@dataclasses.dataclass
class CehrGPTArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    patient_splits_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "Path to the patient splits file"},
    )
    tokenized_dataset_name: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "The name of the dataset to use for tokenization."},
    )
    tokenized_full_dataset_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "The path to the tokenized dataset created for the full population"
        },
    )
    allow_missing_tokenized_persons: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "When extracting cohort sequences from the tokenized dataset, skip the "
            "cohort persons that are missing from it (with a warning) instead of raising an "
            "error."
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
    backbone: Literal["gpt2", "qwen2"] = dataclasses.field(
        default="gpt2",
        metadata={
            "help": "Which decoder stack to train. 'gpt2' is the original CEHR-GPT block "
            "(LayerNorm, Conv1D projections, optional age-based rotary). 'qwen2' is the "
            "Qwen2-style block (RMSNorm, Linear q/k/v/o projections, per-head rotary over "
            "sequential positions)."
        },
    )
    rms_norm_eps: Optional[float] = dataclasses.field(
        default=1e-6,
        metadata={"help": "Epsilon for the RMSNorm layers; only used by the qwen2 backbone"},
    )
    instability_probe: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "Attach InstabilityProbe, which logs gradient norms (total and "
            "per decoder layer), max |logit|, embedding norm and final-norm weight norm, "
            "and stops the run on the first non-finite gradient. Diagnostic only; it "
            "reads gradients and does not modify training."
        },
    )
    profile_gpu_steps: int = dataclasses.field(
        default=0,
        metadata={
            "help": "Profile this many training steps (after --profile_gpu_start_step) and log "
            "the seconds per step and the most expensive GPU operations. 0 disables it."
        },
    )
    profile_gpu_start_step: int = dataclasses.field(
        default=20,
        metadata={
            "help": "Training steps to run before profiling starts, so that compilation and "
            "allocator warm-up are not in the profile."
        },
    )
    instability_probe_every: Optional[int] = dataclasses.field(
        default=25,
        metadata={"help": "Step interval for InstabilityProbe logging"},
    )
    resid_pdrop: Optional[float] = dataclasses.field(
        default=0.1,
        metadata={
            "help": "Dropout on the residual path (attention output and MLP output). "
            "Applies to both backbones."
        },
    )
    embd_pdrop: Optional[float] = dataclasses.field(
        default=0.1,
        metadata={"help": "Dropout on the embeddings. Applies to both backbones."},
    )
    attn_pdrop: Optional[float] = dataclasses.field(
        default=0.1,
        metadata={
            "help": "Dropout on the attention probabilities. Applies to both backbones."
        },
    )
    num_key_value_heads: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "Number of key/value heads for grouped-query attention. Defaults to "
            "num_attention_heads, i.e. standard multi-head attention. Only used by the "
            "qwen2 backbone."
        },
    )
    rope_theta: Optional[float] = dataclasses.field(
        default=10000.0,
        metadata={
            "help": "Base period of the rotary embeddings; only used by the qwen2 backbone"
        },
    )
    use_qk_norm: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "Apply RMSNorm to each head's query and key before the rotary "
            "embedding (Qwen3's QK-norm). Qwen2 leaves the attention logit scale "
            "unbounded, which destabilises from-scratch training. Adds parameters, so a "
            "checkpoint trained with this cannot be reloaded without it. Only used by the "
            "qwen2 backbone."
        },
    )
    # NOTE: deliberately not named `attn_implementation`; cehrbert's ModelArguments
    # already defines that field and HfArgumentParser raises on duplicate field names
    # across the dataclasses it parses together.
    force_attn_implementation: Optional[
        Literal["eager", "sdpa", "flash_attention_2"]
    ] = dataclasses.field(
        default=None,
        metadata={
            "help": "Force a specific attention implementation. When unset, the gpt2 "
            "backbone uses flash attention 2 if available (else eager) and the qwen2 "
            "backbone uses sdpa. 'sdpa' is only implemented for the qwen2 backbone."
        },
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
    freeze_cehrgpt_generation_model: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to freeze the CEHR-GPT generation model weights"
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
    linear_prob_n_layer: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "The number of linear prob layers to use."},
    )
    vocab_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "The directory where the concept data is stored."},
    )
    include_open_vocab: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Train the patient timeline QA head (open vocabulary head): it answers when any "
            "question about the future of a patient happens, at the prediction points of the MOTOR "
            "labels, so include_motor_time_to_event is required. Needs open_vocab_hierarchy_path and "
            "open_vocab_embeddings_dir. Combine with continue_pretrain to add the head to a model whose "
            "MOTOR and CEHR-GPT training is done, see open_vocab_start_step."
        },
    )
    open_vocab_start_step: int = dataclasses.field(
        default=0,
        metadata={
            "help": "The number of optimizer steps CEHR-GPT and MOTOR train alone before the patient timeline "
            "QA loss is added: the head is skipped (and costs no compute) while the global step is below it "
            "and trained from then on. 0 trains it from the first step. The step counts from the beginning "
            "of the run, also when it resumes from a checkpoint."
        },
    )
    open_vocab_hierarchy_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Parquet file with the concept to question roll-up, written by "
            "python -m cehrgpt.open_vocab.assets."
        },
    )
    open_vocab_embeddings_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Folder with the embeddings of the question names, written by "
            "python -m cehrgpt.open_vocab.question_embeddings."
        },
    )
    open_vocab_num_questions: int = dataclasses.field(
        default=512,
        metadata={"help": "The number of questions asked of each batch."},
    )
    open_vocab_positive_fraction: float = dataclasses.field(
        default=0.5,
        metadata={
            "help": "The share of the questions of a batch drawn from those that happen to some patient of "
            "the batch; the rest is drawn from all the questions."
        },
    )
    open_vocab_loss_weight: float = dataclasses.field(
        default=1.0,
        metadata={"help": "The weight of the patient timeline QA loss in the total loss."},
    )
    open_vocab_residual_dim: int = dataclasses.field(
        default=256,
        metadata={"help": "The width of the hidden layer that refines the linear map of a question."},
    )
    gpu_ids: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Linear probing only: run feature extraction on several GPUs. 'all' or "
            "comma separated GPU indices (relative to CUDA_VISIBLE_DEVICES). The dataset is "
            "tokenized once, split into disjoint shards, and each worker process extracts "
            "the features of one shard on its own GPU."
        },
    )
    workers_per_gpu: int = dataclasses.field(
        default=1,
        metadata={
            "help": "Linear probing only: number of worker processes per GPU in --gpu_ids "
            "(or on the single GPU given). More than one worker keeps a GPU busy while the "
            "others are on the CPU, at the cost of one model copy per worker."
        },
    )
    shard_dataset_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Internal: path to the dataset shard a --gpu_ids worker processes."
        },
    )
    apply_entropy_filter: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "A flag to indicate whether we want to use entropy filter."},
    )
    keep_unused_parquet_columns: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "Load every column of the parquet patient sequences. By default the per-token "
            "columns that tokenization and pretraining never read (orders, record_ranks, priorities, "
            "...) are left out, which roughly halves the datasets cache and the tokenization map."
        },
    )
    min_patient_count: Optional[int] = dataclasses.field(
        default=0,
        metadata={
            "help": "Keep a clinical concept in the tokenizer only if at least this many patients "
            "of the tokenizer training data have it. With an ontology (motor_use_ontology) the "
            "descendants count towards their ancestors, once per patient. 0 keeps every concept. "
            "Time and visit markers and the demographic tokens (year, age, gender, race) are always kept"
        },
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
    combine_global_local_features: Optional[bool] = dataclasses.field(
        default=True,
        metadata={
            "help": "A flag to indicate whether we want to combine global (linear prob) "
            "and local (last token hidden states) features"
        },
    )
    add_random_token: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to indicate whether we want to use linear prob tokens"
        },
    )
    include_age_at_vs_prediction: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to include the age-at-VS reconstruction learning objective, "
            "which predicts patient age at each [VS] token using cross-entropy loss. "
            "The age vocab size is derived automatically from the tokenizer vocabulary."
        },
    )
    age_at_vs_prediction_loss_weight: Optional[float] = dataclasses.field(
        default=1.0,
        metadata={"help": "Loss weight for the age-at-VS prediction objective."},
    )
    include_year_at_vs_prediction: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": "A flag to include the year-at-VS reconstruction learning objective, "
            "which predicts the calendar year at each [VS] token using cross-entropy loss. "
            "The year vocab size is derived automatically from the tokenizer vocabulary."
        },
    )
    year_at_vs_prediction_loss_weight: Optional[float] = dataclasses.field(
        default=1.0,
        metadata={"help": "Loss weight for the year-at-VS prediction objective."},
    )
    aux_loss_warmup_steps: int = dataclasses.field(
        default=0,
        metadata={
            "help": "Number of training steps over which auxiliary losses (all losses except "
            "token_loss) are linearly ramped from 0 to their full weight. "
            "Set to 0 (default) to disable warmup and use full weights from the start. "
            "Ignored when aux_loss_start_step > 0."
        },
    )
    aux_loss_start_step: int = dataclasses.field(
        default=0,
        metadata={
            "help": "Step at which auxiliary losses are switched on (step function). "
            "Before this step the weight is 0; at and after it the weight is 1. "
            "When set (> 0) this overrides aux_loss_warmup_steps."
        },
    )
    refresh_processed_dataset: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "If set, delete the cached prepared dataset at prepared_ds_path and "
            "rebuild it from scratch, ignoring any previously saved version."
        },
    )
    use_time_embedding: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Add trainable Time2Vec-style embeddings for epoch_times and ages "
            "(sin(t/scaling_factor * w + phi)), projected back to hidden_size via a linear layer. "
            "Mirrors the CehrBert embedding approach."
        },
    )
    n_time_embd: int = dataclasses.field(
        default=16,
        metadata={"help": "Dimensionality of the time/age embedding vectors (Time2Vec size)."},
    )
    time_embedding_scaling_factor: float = dataclasses.field(
        default=86400.0,
        metadata={
            "help": "Scaling factor for epoch_times before Time2Vec encoding. "
            "Defaults to 86400 (seconds per day), normalising epoch timestamps to days."
        },
    )
    age_embedding_scaling_factor: float = dataclasses.field(
        default=100.0,
        metadata={
            "help": "Scaling factor for ages before Time2Vec encoding. "
            "Defaults to 100, normalising typical age-in-years values to [0, ~1]."
        },
    )
    use_local_attention: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Enable visit-local attention: each token attends only to the current visit "
            "and the previous local_attention_n_prev_visits visits. Only applies to eager "
            "(non-Flash) attention; no-op during KV-cache generation."
        },
    )
    local_attention_n_prev_visits: int = dataclasses.field(
        default=5,
        metadata={"help": "Number of previous visits visible to each token when use_local_attention is True."},
    )
