from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import optuna
from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
from datasets import Dataset, DatasetDict
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from transformers.utils import logging

from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

LOG = logging.get_logger("transformers")


class OptunaMetricCallback(TrainerCallback):
    """
    A custom callback to store the best metric in the evaluation metrics dictionary during training.

    This callback monitors the training state and updates the metrics dictionary with the `best_metric`
    (e.g., the lowest `eval_loss` or highest accuracy) observed during training. It ensures that the
    best metric value is preserved in the final evaluation results, even if early stopping occurs.

    Attributes:
        None

    Methods:
        on_evaluate(args, state, control, **kwargs):
            Called during evaluation. Adds `state.best_metric` to `metrics` if it exists.

    Example Usage:
        ```
        store_best_metric_callback = StoreBestMetricCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[store_best_metric_callback]
        )
        ```
    """

    def on_evaluate(self, args, state, control, **kwargs):
        """
        During evaluation, adds the best metric value to the metrics dictionary if it exists.

        Args:
            args: Training arguments.
            state: Trainer state object that holds information about training progress.
            control: Trainer control object to modify training behavior.
            **kwargs: Additional keyword arguments, including `metrics`, which holds evaluation metrics.

        Updates:
            `metrics["best_metric"]`: Sets this to `state.best_metric` if available.
        """
        # Check if best metric is available and add it to metrics if it exists
        metrics = kwargs.get("metrics", {})
        if state.best_metric is not None:
            metrics.update(
                {"optuna_best_metric": min(state.best_metric, metrics["eval_loss"])}
            )
        else:
            metrics.update({"optuna_best_metric": metrics["eval_loss"]})


def get_suggestion(
    trial,
    hyperparameter_name: str,
    hyperparameters: List[Union[float, int]],
    is_grid: bool = False,
) -> Union[float, int]:
    """
    Get hyperparameter suggestion based on search mode.

    Args:
        trial: Optuna trial object
        hyperparameter_name: Name of the hyperparameter
        hyperparameters: List of hyperparameter values
        is_grid: Whether to use grid search mode

    Returns:
        Suggested hyperparameter value

    Raises:
        RuntimeError: If Bayesian mode is used with incorrect number of bounds
    """
    if is_grid:
        return trial.suggest_categorical(hyperparameter_name, hyperparameters)

    # For Bayesian optimization, we need exactly 2 values (lower and upper bounds)
    if len(hyperparameters) != 2:
        raise RuntimeError(
            f"{hyperparameter_name} must contain exactly two values (lower and upper bound) "
            f"for Bayesian Optimization, but {len(hyperparameters)} values were provided: {hyperparameters}"
        )

    # Ensure bounds are sorted
    lower, upper = sorted(hyperparameters)
    return trial.suggest_float(hyperparameter_name, lower, upper, log=True)


def hp_space(trial: optuna.Trial, cehrgpt_args: CehrGPTArguments):
    """
    Define the hyperparameter search space.

    Args:
        trial: Optuna trial object
        cehrgpt_args: CehrGPTArguments
    Returns:
        Dictionary of hyperparameter suggestions
    """

    is_grid = cehrgpt_args.hyperparameter_tuning_is_grid
    learning_rates = cehrgpt_args.hyperparameter_learning_rates
    weight_decays = cehrgpt_args.hyperparameter_weight_decays
    batch_sizes = cehrgpt_args.hyperparameter_batch_sizes
    num_train_epochs = cehrgpt_args.hyperparameter_num_train_epochs

    return {
        "learning_rate": get_suggestion(
            trial, "learning_rate", learning_rates, is_grid
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", batch_sizes
        ),
        "weight_decay": get_suggestion(trial, "weight_decay", weight_decays, is_grid),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs", num_train_epochs
        ),
    }


def create_grid_search_space(cehrgpt_args: CehrGPTArguments):
    """
    Create the search space dictionary for GridSampler.

    Args:
        cehrgpt_args: CehrGPTArguments

    Returns:
        Dictionary defining the grid search space
    """
    return {
        "learning_rate": cehrgpt_args.hyperparameter_learning_rates,
        "weight_decay": cehrgpt_args.hyperparameter_weight_decays,
        "per_device_train_batch_size": cehrgpt_args.hyperparameter_batch_sizes,
        "num_train_epochs": cehrgpt_args.hyperparameter_num_train_epochs,
    }


def calculate_total_combinations(search_space: dict) -> int:
    """Calculate total number of combinations in grid search."""
    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


def sample_dataset(data: Dataset, percentage: float, seed: int) -> Dataset:
    """
    Samples a subset of the given dataset based on a specified percentage.

    This function uses a random train-test split to select a subset of the dataset, returning a sample
    that is approximately `percentage` of the total dataset size. It is useful for creating smaller
    datasets for tasks such as hyperparameter tuning or quick testing.

    Args:
        data (Dataset): The input dataset to sample from.
        percentage (float): The fraction of the dataset to sample, represented as a decimal
                            (e.g., 0.1 for 10%).
        seed (int): A random seed for reproducibility in the sampling process.

    Returns:
        Dataset: A sampled subset of the input dataset containing `percentage` of the original data.

    Example:
        ```
        sampled_data = sample_dataset(my_dataset, percentage=0.1, seed=42)
        ```

    Notes:
        - The `train_test_split` method splits the dataset into "train" and "test" portions. This function
          returns the "test" portion, which is the specified percentage of the dataset.
        - Ensure that `percentage` is between 0 and 1 to avoid errors.
    """
    if percentage >= 1.0:
        return data

    return data.train_test_split(
        test_size=percentage,
        seed=seed,
    )["test"]


def perform_hyperparameter_search(
    trainer_class,
    model_init: Callable,
    dataset: DatasetDict,
    data_collator: CehrGptDataCollator,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
) -> Tuple[TrainingArguments, Optional[str]]:
    """
    Perform hyperparameter tuning for the CehrGPT model using Optuna with the Hugging Face Trainer.

    This function supports two modes:
    1. Bayesian Optimization (TPE): Intelligently explores hyperparameter space using bounds
    2. Grid Search: Exhaustively tests all combinations of discrete values

    Args:
        trainer_class: A Trainer or its subclass
        model_init (Callable): A function to initialize the model, used for each hyperparameter trial.
        dataset (DatasetDict): A Hugging Face DatasetDict containing "train" and "validation" datasets.
        data_collator (CehrGptDataCollator): A data collator for processing batches.
        training_args (TrainingArguments): Configuration for training parameters (e.g., epochs, evaluation strategy).
        model_args (ModelArguments): Model configuration arguments, including early stopping parameters.
        cehrgpt_args (CehrGPTArguments): Additional arguments specific to CehrGPT, including hyperparameter
                                         tuning options and search mode configuration.

    Returns:
        Tuple[TrainingArguments, Optional[str]]: Updated TrainingArguments with best hyperparameters
                                               and optional run_id of the best trial.

    Example:
        ```
        best_training_args, run_id = perform_hyperparameter_search(
            trainer_class=Trainer,
            model_init=my_model_init,
            dataset=my_dataset_dict,
            data_collator=my_data_collator,
            training_args=initial_training_args,
            model_args=model_args,
            cehrgpt_args=cehrgpt_args
        )
        ```

    Notes:
        - If `cehrgpt_args.hyperparameter_tuning` is set to `True`, this function samples a portion of the
          training and validation datasets for efficient tuning.
        - `EarlyStoppingCallback` is added to the Trainer if early stopping is enabled in `model_args`.
        - Optuna's `hyperparameter_search` is configured with the specified number of trials (`n_trials`)
          and learning rate and batch size ranges provided in `cehrgpt_args`.

    Logging:
        Logs the best hyperparameters found at the end of the search.
    """
    if not cehrgpt_args.hyperparameter_tuning:
        return training_args, None

    # Prepare hyperparameters based on mode
    if (
        cehrgpt_args.hyperparameter_tuning_is_grid
        and cehrgpt_args.hyperparameter_tuning_is_grid
    ):
        search_space = create_grid_search_space(cehrgpt_args)
        total_combinations = calculate_total_combinations(search_space)

        LOG.info(f"Grid search mode: Testing {total_combinations} combinations")
        LOG.info(f"Search space: {search_space}")

        # Adjust n_trials for grid search if not set appropriately
        if cehrgpt_args.n_trials < total_combinations:
            LOG.warning(
                f"n_trials ({cehrgpt_args.n_trials}) is less than total combinations ({total_combinations}). "
                f"Setting n_trials to {total_combinations} to test all combinations."
            )
            cehrgpt_args.n_trials = total_combinations

        # Configure sampler based on search mode
        sampler = optuna.samplers.GridSampler(search_space, seed=training_args.seed)
    else:
        LOG.info("Bayesian optimization mode (TPE)")
        LOG.info(f"Learning rate bounds: {cehrgpt_args.hyperparameter_learning_rates}")
        LOG.info(f"Weight decay bounds: {cehrgpt_args.hyperparameter_weight_decays}")
        LOG.info(f"Batch sizes: {cehrgpt_args.hyperparameter_batch_sizes}")
        LOG.info(f"Epochs: {cehrgpt_args.hyperparameter_num_train_epochs}")
        # Configure the TPE sampler
        sampler = optuna.samplers.TPESampler(seed=training_args.seed)

    # Prepare datasets
    save_total_limit_original = training_args.save_total_limit
    training_args.save_total_limit = 1

    sampled_train = sample_dataset(
        dataset["train"],
        cehrgpt_args.hyperparameter_tuning_percentage,
        training_args.seed,
    )
    sampled_val = sample_dataset(
        dataset["validation"],
        cehrgpt_args.hyperparameter_tuning_percentage,
        training_args.seed,
    )
    # Create trainer
    hyperparam_trainer = trainer_class(
        model_init=model_init,
        data_collator=data_collator,
        train_dataset=sampled_train,
        eval_dataset=sampled_val,
        callbacks=[
            EarlyStoppingCallback(model_args.early_stopping_patience),
            OptunaMetricCallback(),
        ],
        args=training_args,
    )

    best_trial = hyperparam_trainer.hyperparameter_search(
        direction="minimize",
        hp_space=partial(
            hp_space,
            cehrgpt_args=cehrgpt_args,
        ),
        backend="optuna",
        n_trials=cehrgpt_args.n_trials,
        compute_objective=lambda m: m["optuna_best_metric"],
        sampler=sampler,
    )

    # Log results
    LOG.info("=" * 50)
    LOG.info("HYPERPARAMETER SEARCH COMPLETED")
    LOG.info("=" * 50)
    LOG.info(f"Best hyperparameters: {best_trial.hyperparameters}")
    LOG.info(f"Best metric (eval_loss): {best_trial.objective}")
    LOG.info(f"Best run_id: {best_trial.run_id}")
    LOG.info("=" * 50)

    # Restore original settings and update with best hyperparameters
    training_args.save_total_limit = save_total_limit_original
    for k, v in best_trial.hyperparameters.items():
        setattr(training_args, k, v)
        LOG.info(f"Updated training_args.{k} = {v}")

    return training_args, best_trial.run_id
