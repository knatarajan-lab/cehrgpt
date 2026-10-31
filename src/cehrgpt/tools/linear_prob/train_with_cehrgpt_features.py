import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as pa_dataset
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cehrgpt.tools.linear_prob.feature_io import (
    FEATURE_SETS,
    read_feature_info,
    resolve_feature_sets,
)


DEFAULT_SEED = 42


def order_samples(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    The samples ordered by subject_id and prediction_time, and then shuffled with the seed.

    The feature files have random names, so the order in which the files of a folder are read, and the samples in it,
    differs from one run to the next. This puts the samples in an order that depends on the samples and the seed only.
    Samples of the same subject and time (the same sample extracted twice) are ordered by their label and their first
    feature, which does not depend on the files either.
    """
    first_feature = np.fromiter(
        (np.asarray(features).flat[0] if len(features) else 0.0 for features in df["features"]),
        dtype=float,
        count=len(df),
    )
    # np.lexsort sorts by the last key first
    order = np.lexsort(
        (
            first_feature,
            df["boolean_value"].to_numpy().astype(int),
            pd.to_datetime(df["prediction_time"]).to_numpy(),
            df["subject_id"].to_numpy(),
        )
    )
    return df.iloc[order].sample(frac=1.0, random_state=seed, replace=False)


def stack_features(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["features"].apply(lambda x: np.array(x).flatten()))


def prepare_dataset(
    df: pd.DataFrame,
    feature_processor: Dict[str, Union[StandardScaler, OneHotEncoder]],
    features: Optional[np.ndarray] = None,
    columns: Optional[slice] = None,
) -> Dict[str, Any]:
    """
    The scaled features and the labels of df.

    features are the stacked feature vectors of df when the caller has them already, and columns are the
    part of each vector to use (the global or the local features), all of it when None.
    """
    age_scaler = feature_processor["age_scaler"]
    gender_encoder = feature_processor["gender_encoder"]
    race_encoder = feature_processor["race_encoder"]
    feature_scaler = feature_processor["feature_scaler"]

    age_scaler.transform(df[["age_at_index"]].to_numpy())

    one_hot_gender = gender_encoder.transform(
        np.expand_dims(df.gender_concept_id.to_numpy(), axis=1)
    )
    one_hot_race = race_encoder.transform(
        np.expand_dims(df.race_concept_id.to_numpy(), axis=1)
    )

    if features is None:
        features = stack_features(df)
    if columns is not None:
        features = features[:, columns]
    features = feature_scaler.transform(features)
    # features = np.hstack(
    #     [scaled_age, one_hot_gender.toarray(), one_hot_race.toarray(), features]
    # )
    return {
        "subject_id": df["subject_id"].to_numpy(),
        "prediction_time": df["prediction_time"].tolist(),
        "features": features,
        "boolean_value": df["boolean_value"].to_numpy(),
    }


DEFAULT_WANDB_PROJECT = "cehrgpt-linear-probing"


def init_wandb(
    args, output_dir: Path, config: Dict[str, Any], variant: Optional[str] = None
) -> Optional[Any]:
    """
    Start the wandb run of this linear probing result, or return None unless --report_to wandb.

    The run is identified by the output directory, so running the same cohort again resumes and
    updates its run instead of creating a new one. By default the run is named <cohort>/<model>
    and grouped by <model>, which is the layout run_linear_prob.sh writes (<output>/<cohort>/<model>).
    A feature set other than the vectors as saved (global or local) is a run of its own: <cohort>/<model>/<set>.
    """
    if args.report_to != "wandb":
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "--report_to wandb needs the wandb package, install it with `pip install wandb`"
        ) from e
    resolved = output_dir.resolve()
    return wandb.init(
        project=args.wandb_project
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_WANDB_PROJECT,
        group=args.wandb_group or resolved.name,
        name=(args.wandb_run_name or f"{resolved.parent.name}/{resolved.name}")
        + (f"/{variant}" if variant else ""),
        id=hashlib.sha1((str(resolved) + (f"/{variant}" if variant else "")).encode()).hexdigest()[:16],
        resume="allow",
        dir=str(resolved),
        config=config,
    )


def report_to_wandb(
    args,
    output_dir: Path,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    variant: Optional[str] = None,
) -> None:
    run = init_wandb(args, output_dir, config, variant)
    if run is None:
        return
    run.log(metrics)
    run.summary.update(metrics)
    run.finish()


def peek_num_features(train_dir: Path) -> int:
    """The width of the feature vectors, from one row (the whole dataset is not read)."""
    row = pa_dataset.dataset(str(train_dir), format="parquet").head(1, columns=["features"])
    return len(row.column("features")[0].as_py())


def variant_paths(output_dir: Path, name: str, as_saved: bool):
    """
    Where the processor and the results of a feature set go.

    The vectors as they were saved keep the paths of a single model (logistic/), which meds-evaluation and the
    earlier results use; the global and local features get logistic_global/ and logistic_local/ next to it.
    """
    suffix = "" if as_saved else f"_{name}"
    return output_dir / f"feature_processor{suffix}.pickle", output_dir / f"logistic{suffix}"


def train_feature_set(
    args,
    name: str,
    columns: Optional[slice],
    data: Dict[str, Any],
    features_data_dir: Path,
    output_dir: Path,
) -> None:
    """
    Train and test the logistic regression on one feature set: columns of each feature vector, or all of it (None)
    for the vectors as they were saved, which keep the paths of a single model.
    """
    as_saved = columns is None
    feature_train, feature_test = data["feature_train"], data["feature_test"]
    feature_processor_path, logistic_dir = variant_paths(output_dir, name, as_saved)
    logistic_dir.mkdir(exist_ok=True, parents=True)
    wandb_variant = None if as_saved else name

    def vectors(split_features: np.ndarray) -> np.ndarray:
        return split_features if columns is None else split_features[:, columns]

    if feature_processor_path.exists():
        with open(feature_processor_path, "rb") as f:
            feature_processor = pickle.load(f)
    else:
        feature_scaler, age_scaler, gender_encoder, race_encoder = (
            StandardScaler(),
            StandardScaler(),
            OneHotEncoder(handle_unknown="ignore"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        feature_scaler = feature_scaler.fit(vectors(data["train_vectors"]))
        age_scaler = age_scaler.fit(feature_train[["age_at_index"]].to_numpy())
        gender_encoder = gender_encoder.fit(
            feature_train[["gender_concept_id"]].to_numpy()
        )
        race_encoder = race_encoder.fit(feature_train[["race_concept_id"]].to_numpy())
        feature_processor = {
            "feature_scaler": feature_scaler,
            "age_scaler": age_scaler,
            "gender_encoder": gender_encoder,
            "race_encoder": race_encoder,
        }
        with open(feature_processor_path, "wb") as f:
            pickle.dump(feature_processor, f)

    logistic_test_result_file = logistic_dir / "metrics.json"
    logistic_model_file = logistic_dir / "model.pickle"
    if logistic_model_file.exists():
        print(
            f"The logistic regression model already exist, loading it from {logistic_model_file}"
        )
        with open(logistic_model_file, "rb") as f:
            model = pickle.load(f)
    else:
        train_dataset = prepare_dataset(
            feature_train, feature_processor, data["train_vectors"], columns
        )
        # Train logistic regression
        model = LogisticRegressionCV(
            scoring="roc_auc",
            random_state=getattr(args, "seed", DEFAULT_SEED),
            max_iter=1000,
            n_jobs=8,
        )
        model.fit(train_dataset["features"], train_dataset["boolean_value"])
        with open(logistic_model_file, "wb") as f:
            pickle.dump(model, f)

    test_dataset = prepare_dataset(
        feature_test, feature_processor, data["test_vectors"], columns
    )

    y_pred = model.predict_proba(test_dataset["features"])[:, 1]
    logistic_predictions = pl.DataFrame(
        {
            "subject_id": test_dataset["subject_id"].tolist(),
            "prediction_time": test_dataset["prediction_time"],
            "predicted_boolean_probability": y_pred.tolist(),
            "predicted_boolean_value": None,
            "boolean_value": test_dataset["boolean_value"].astype(bool).tolist(),
        }
    )
    logistic_predictions = logistic_predictions.with_columns(
        pl.col("predicted_boolean_value").cast(pl.Boolean())
    )
    logistic_test_predictions = logistic_dir / "test_predictions"
    logistic_test_predictions.mkdir(exist_ok=True, parents=True)
    logistic_predictions.write_parquet(logistic_test_predictions / "predictions.parquet")

    roc_auc = roc_auc_score(test_dataset["boolean_value"], y_pred)
    precision, recall, _ = precision_recall_curve(test_dataset["boolean_value"], y_pred)
    pr_auc = auc(recall, precision)

    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    print(f"Logistic ({name}):", features_data_dir.name, metrics)
    with open(logistic_test_result_file, "w") as f:
        json.dump(metrics, f, indent=4)

    report_to_wandb(
        args,
        output_dir,
        metrics,
        {
            "features_data_dir": str(features_data_dir),
            "feature_set": name,
            "num_train_samples": len(feature_train),
            "num_test_samples": len(feature_test),
            "num_features": int(test_dataset["features"].shape[-1]),
            "test_positive_rate": float(np.mean(test_dataset["boolean_value"])),
        },
        wandb_variant,
    )


def main(args):
    features_data_dir = Path(args.features_data_dir)
    output_dir = Path(args.output_dir)
    train_dir = features_data_dir / "features_with_label" / "train_features"
    test_dir = features_data_dir / "features_with_label" / "test_features"

    # The feature sets that the saved vectors allow: the vectors as saved, and when they were saved combined
    # (global | local, twice the hidden size) also the global and the local features on their own
    info = read_feature_info(features_data_dir)
    num_features = info["num_features"] if info else peek_num_features(train_dir)
    feature_sets = resolve_feature_sets(
        getattr(args, "feature_sets", None),
        num_features,
        info,
        getattr(args, "hidden_size", None),
    )
    print(f"Training the feature sets {list(feature_sets)} of {num_features}-wide features")

    pending = {}
    for name, columns in feature_sets.items():
        _, logistic_dir = variant_paths(output_dir, name, columns is None)
        logistic_test_result_file = logistic_dir / "metrics.json"
        if logistic_test_result_file.exists():
            print(f"The {name} models have been trained, and skip ...")
            # Report the existing results too, so that turning wandb on for a finished cohort fills it in
            with open(logistic_test_result_file) as f:
                report_to_wandb(
                    args,
                    output_dir,
                    json.load(f),
                    {"features_data_dir": str(features_data_dir), "feature_set": name},
                    None if columns is None else name,
                )
        else:
            pending[name] = columns
    if not pending:
        return

    feature_train = pd.read_parquet(train_dir)
    feature_test = pd.read_parquet(test_dir)

    # The order of the samples does not depend on the files that they were read from
    seed = getattr(args, "seed", DEFAULT_SEED)
    feature_train = order_samples(feature_train, seed)
    feature_test = order_samples(feature_test, seed)
    data = {
        "feature_train": feature_train,
        "feature_test": feature_test,
        # stacked once for all the feature sets, which only differ in the columns that they use
        "train_vectors": stack_features(feature_train),
        "test_vectors": stack_features(feature_test),
    }
    for name, columns in pending.items():
        train_feature_set(args, name, columns, data, features_data_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train logistic regression model with cehrgpt features"
    )
    parser.add_argument(
        "--features_data_dir",
        required=True,
        help="Directory containing training and test feature files",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the output results"
    )
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        choices=["all", *FEATURE_SETS],
        default=None,
        help="The features to train a logistic regression on: concatenated (the saved vectors, global then local), "
        "global (the linear probe state) and local (the last token state). The global and the local features "
        "can only be told apart in features that were extracted with --combine_global_local_features true "
        "(twice the hidden size). Default: all of them when the features allow, else the saved vectors. "
        "The saved vectors are written to logistic/, the others to logistic_global/ and logistic_local/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="The seed of the shuffle of the samples (they are first ordered by subject_id and prediction_time, "
        "so the order does not depend on the feature files) and of the logistic regression (default: 42)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="The hidden size of the model, for features extracted before feature_info.json was written "
        "(the file normally tells it)",
    )
    parser.add_argument(
        "--report_to",
        choices=["none", "wandb"],
        default="none",
        help="Log the ROC-AUC and PR-AUC to wandb (default: none)",
    )
    parser.add_argument(
        "--wandb_project",
        help=f"wandb project (default: $WANDB_PROJECT, else {DEFAULT_WANDB_PROJECT})",
    )
    parser.add_argument(
        "--wandb_group",
        help="wandb run group (default: the name of the output directory, i.e. the model)",
    )
    parser.add_argument(
        "--wandb_run_name",
        help="wandb run name (default: <parent directory>/<output directory>, i.e. <cohort>/<model>)",
    )
    main(parser.parse_args())
