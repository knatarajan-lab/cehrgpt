import copy
import json
import os
from typing import Callable, Union

import optuna
from cehrbert.runners.hf_runner_argument_dataclass import ModelArguments
from cehrbert.runners.runner_util import get_last_hf_checkpoint
from datasets import Dataset, DatasetDict, IterableDataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
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
    if percentage == 1.0:
        return data

    if isinstance(data, Dataset):
        return data.train_test_split(
            test_size=percentage,
            seed=seed,
        )["test"]
    else:
        raise RuntimeError(
            "IterableDataset is not supported for hyperparameter search."
        )


def create_objective(
    model_init: Callable,
    train_dataset: Union[Dataset, IterableDataset],
    eval_dataset: Union[Dataset, IterableDataset],
    data_collator: CehrGptDataCollator,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
):
    def objective(trial):
        args = copy.deepcopy(training_args)
        args.output_dir = os.path.join(args.output_dir, f"runs/{trial.number}")
        os.makedirs(args.output_dir, exist_ok=True)
        args.learning_rate = trial.suggest_float(
            "learning_rate", cehrgpt_args.lr_low, cehrgpt_args.lr_high
        )
        args.weight_decay = trial.suggest_float(
            "weight_decays",
            cehrgpt_args.weight_decays_low,
            cehrgpt_args.weight_decays_high,
        )
        args.per_device_train_batch_size = trial.suggest_categorical(
            "per_device_train_batch_size", cehrgpt_args.hyperparameter_batch_sizes
        )
        LOG.info(
            "Trial %s: learning_rate %s, weight_decay %s, per_device_train_batch_size %s",
            trial.number,
            args.learning_rate,
            args.weight_decay,
            args.per_device_train_batch_size,
        )
        checkpoint = get_last_hf_checkpoint(args)

        trainer = Trainer(
            model_init=model_init,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(model_args.early_stopping_patience)],
            args=args,
        )

        # Train the model
        trainer.train(resume_from_checkpoint=checkpoint)

        # Save actual training epochs to a JSON file in the trial's output directory
        epochs_info = {"actual_epochs": trainer.state.epoch}
        epochs_file_path = os.path.join(args.output_dir, "epochs_info.json")
        with open(epochs_file_path, "w") as f:
            json.dump(epochs_info, f)
        # Get the number of epochs the model actually trained
        # Save the actual_epochs into the trial's user attributes for later retrieval
        trial.set_user_attr("actual_epochs", trainer.state.epoch)

        # Return the evaluation metric to guide the hyperparameter search
        return trainer.evaluate()["eval_loss"]

    return objective


def perform_hyperparameter_search(
    model_init: Callable,
    dataset: DatasetDict,
    data_collator: CehrGptDataCollator,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
) -> TrainingArguments:
    """
    Perform hyperparameter tuning for the CehrGPT model using Optuna with the Hugging Face Trainer.

    This function initializes a Trainer with sampled training and validation sets, and performs
    a hyperparameter search using Optuna. The search tunes learning rate, batch size, and weight decay
    to optimize model performance based on a specified objective metric (e.g., validation loss).
    After the search, it updates the provided `TrainingArguments` with the best hyperparameters found.

    Args:
        model_init (Callable): A function to initialize the model, used for each hyperparameter trial.
        dataset (DatasetDict): A Hugging Face DatasetDict containing "train" and "validation" datasets.
        data_collator (CehrGptDataCollator): A data collator for processing batches.
        training_args (TrainingArguments): Configuration for training parameters (e.g., epochs, evaluation strategy).
        model_args (ModelArguments): Model configuration arguments, including early stopping parameters.
        cehrgpt_args (CehrGPTArguments): Additional arguments specific to CehrGPT, including hyperparameter
                                         tuning options such as learning rate range, batch sizes, and tuning percentage.

    Returns:
        TrainingArguments: Updated `TrainingArguments` instance containing the best hyperparameters found
                           from the search.

    Example:
        ```
        best_training_args = perform_hyperparameter_search(
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
    if cehrgpt_args.hyperparameter_tuning:
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

        objective = create_objective(
            model_init=model_init,
            train_dataset=sampled_train,
            eval_dataset=sampled_val,
            data_collator=data_collator,
            training_args=training_args,
            model_args=model_args,
            cehrgpt_args=cehrgpt_args,
        )

        # Now my_objective can be passed directly to an optimizer
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=cehrgpt_args.n_trials)
        # Retrieve the best trial's actual epochs
        best_trial = study.best_trial
        LOG.info(
            "The number of epochs run for the best trial %s",
            best_trial.user_attrs["actual_epochs"],
        )
        best_trial_epochs = best_trial.user_attrs["actual_epochs"]
        best_trial_epochs -= model_args.early_stopping_patience
        LOG.info(
            "The total number of epochs after subtracting from best_trial_epochs is %s",
            best_trial_epochs,
        )
        training_args.num_train_epochs = best_trial_epochs
        # Update training arguments with best hyperparameters and set epochs based on adjusted effective epochs
        for k, v in best_trial.hyperparameters.items():
            LOG.info("The best parameter %s in the best trial is %s", k, v)
            setattr(training_args, k, v)

    return training_args
