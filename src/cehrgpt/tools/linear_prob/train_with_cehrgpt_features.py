import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as pa_dataset
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cehrgpt.tools.linear_prob.feature_io import (
    FEATURE_SETS,
    LOCAL,
    read_feature_info,
    resolve_feature_sets,
)


DEFAULT_SEED = 42
DEFAULT_CV_FOLDS = 5
# The feature set that weighs the local half against the global half (see train_feature_set)
BLOCKWISE = "blockwise"


def cv_folds(
    labels: np.ndarray, subject_ids: np.ndarray, args
) -> Union[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    The folds that LogisticRegressionCV picks C with.

    By default the number of folds, which LogisticRegressionCV turns into an unshuffled StratifiedKFold, as before.
    With --group_cv_by_subject, the samples of a subject are all in the same fold: a cohort can have several
    prediction times of the same subject, and folds that split them score C on subjects the model was fit on, which
    favours a weak penalty.
    """
    n_folds = getattr(args, "cv_folds", None) or DEFAULT_CV_FOLDS
    if not getattr(args, "group_cv_by_subject", False):
        return n_folds
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=getattr(args, "seed", DEFAULT_SEED)
    )
    return list(splitter.split(np.zeros(len(labels)), labels, groups=subject_ids))


def best_cv_roc_auc(model: LogisticRegressionCV) -> float:
    """The cross-validated ROC-AUC of the C that the model picked (the mean over the folds)."""
    scores = next(iter(model.scores_.values()))  # (folds, Cs), one entry for a binary label
    return float(scores.mean(axis=0).max())


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


def scale_local(
    features: np.ndarray, local_columns: Optional[slice], scale: Optional[float]
) -> np.ndarray:
    """The features with the (standardized) local columns multiplied by scale; unchanged when scale is None."""
    if scale is None or local_columns is None:
        return features
    features = features.copy()
    features[:, local_columns] *= scale
    return features


def train_feature_set(
    args,
    name: str,
    columns: Optional[slice],
    data: Dict[str, Any],
    features_data_dir: Path,
    output_dir: Path,
    local_columns: Optional[slice] = None,
    local_scales: Optional[Sequence[float]] = None,
) -> None:
    """
    Train and test the logistic regression on one feature set: columns of each feature vector, or all of it (None)
    for the vectors as they were saved, which keep the paths of a single model.

    With local_scales (the blockwise set), the local columns of the standardized vectors are multiplied by each scale
    in turn and the scale whose best C has the highest cross-validated ROC-AUC is kept. Multiplying the local columns
    by a is the same as an L2 penalty 1/a**2 times stronger on them than on the global ones, so this gives the two
    halves their own penalty; a scale of 0 is the global features alone.
    """
    as_saved = columns is None and not local_scales
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
        folds = cv_folds(
            train_dataset["boolean_value"], train_dataset["subject_id"], args
        )
        model, local_scale, cv_by_scale = None, None, {}
        for scale in local_scales or [None]:
            candidate = LogisticRegressionCV(
                Cs=getattr(args, "num_cs", None) or 10,
                cv=folds,
                scoring="roc_auc",
                random_state=getattr(args, "seed", DEFAULT_SEED),
                max_iter=1000,
                n_jobs=8,
            )
            candidate.fit(
                scale_local(train_dataset["features"], local_columns, scale),
                train_dataset["boolean_value"],
            )
            if scale is not None:
                cv_by_scale[scale] = best_cv_roc_auc(candidate)
                print(f"Logistic ({name}): local scale {scale}, CV ROC-AUC {cv_by_scale[scale]:.4f}")
            if model is None or best_cv_roc_auc(candidate) > best_cv_roc_auc(model):
                model, local_scale = candidate, scale
        if local_scale is not None:
            # the scale is part of the model: the test features are scaled the same way
            model.local_scale_ = local_scale
            model.cv_roc_auc_by_local_scale_ = cv_by_scale
        with open(logistic_model_file, "wb") as f:
            pickle.dump(model, f)

    test_dataset = prepare_dataset(
        feature_test, feature_processor, data["test_vectors"], columns
    )
    test_dataset["features"] = scale_local(
        test_dataset["features"], local_columns, getattr(model, "local_scale_", None)
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

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        # how C was picked, to compare the cross-validated ROC-AUC with the test one
        "cv_roc_auc": best_cv_roc_auc(model),
        "C": float(model.C_[0]),
        "cv": "grouped by subject" if getattr(args, "group_cv_by_subject", False) else "stratified",
    }
    if getattr(model, "local_scale_", None) is not None:
        metrics["local_scale"] = float(model.local_scale_)
        metrics["cv_roc_auc_by_local_scale"] = {
            str(k): v for k, v in model.cv_roc_auc_by_local_scale_.items()
        }
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
    local_scales = getattr(args, "local_block_scales", None)
    local_columns = None
    if local_scales:
        # the local columns of the combined vectors; an error when the vectors were not saved combined
        local_columns = resolve_feature_sets([LOCAL], num_features, info, getattr(args, "hidden_size", None))[LOCAL]
    blockwise_dir = variant_paths(output_dir, BLOCKWISE, False)[1]
    train_blockwise = bool(local_scales) and not (blockwise_dir / "metrics.json").exists()
    if not pending and not train_blockwise:
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
    if train_blockwise:
        train_feature_set(
            args, BLOCKWISE, None, data, features_data_dir, output_dir, local_columns, local_scales
        )


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
        "--group_cv_by_subject",
        action="store_true",
        help="Keep all the samples of a subject in the same fold when C is picked by cross-validation "
        "(StratifiedGroupKFold on subject_id). Without it the folds are an unshuffled StratifiedKFold, as before",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help=f"The number of cross-validation folds that C is picked with (default: {DEFAULT_CV_FOLDS})",
    )
    parser.add_argument(
        "--num_cs",
        type=int,
        default=10,
        help="The number of values of C, log-spaced from 1e-4 to 1e4, that are tried (default: 10)",
    )
    parser.add_argument(
        "--local_block_scales",
        nargs="+",
        type=float,
        help="Also train the blockwise model on combined features, written to logistic_blockwise/: the "
        "standardized local columns are multiplied by each of these scales (e.g. 0 0.25 0.5 1) and the scale with "
        "the best cross-validated ROC-AUC is kept, which gives the local features their own L2 penalty. "
        "0 is the global features alone and 1 the concatenated ones",
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
