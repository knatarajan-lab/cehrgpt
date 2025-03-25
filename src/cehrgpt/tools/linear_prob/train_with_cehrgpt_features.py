import argparse
import functools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def lightgbm_objective(trial, *, train_data, dev_data, num_trees=None):
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    dtrain = lgb.Dataset(train_data["features"], label=train_data["boolean_value"])
    ddev = lgb.Dataset(dev_data["features"], label=dev_data["boolean_value"])

    if num_trees is None:
        callbacks = [lgb.early_stopping(10)]
        gbm = lgb.train(
            param, dtrain, num_boost_round=1000, valid_sets=(ddev,), callbacks=callbacks
        )
    else:
        gbm = lgb.train(param, dtrain, num_boost_round=num_trees)

    y_pred = gbm.predict(dev_data["features"], raw_score=True)

    error = -roc_auc_score(dev_data["boolean_value"], y_pred)

    if num_trees is None:
        trial.set_user_attr("num_trees", gbm.best_iteration + 1)

    return error


def prepare_dataset(
    df: pd.DataFrame, feature_processor: Dict[str, Union[StandardScaler, OneHotEncoder]]
) -> Dict[str, Any]:
    age_scaler = feature_processor["age_scaler"]
    gender_encoder = feature_processor["gender_encoder"]
    race_encoder = feature_processor["race_encoder"]
    scaled_age = age_scaler.transform(df[["age_at_index"]].to_numpy())
    one_hot_gender = gender_encoder.transform(df[["gender_concept_id"]].to_numpy())
    one_hot_race = race_encoder.transform(df[["gender_concept_id"]].to_numpy())

    features = np.stack(df["features"].apply(lambda x: np.array(x).flatten()))
    concatenated_features = np.hstack(
        [scaled_age, one_hot_gender.toarray(), one_hot_race.toarray(), features]
    )
    return {
        "subject_id": df["subject_id"].tolist(),
        "prediction_time": df["prediction_time"].tolist(),
        "features": concatenated_features,
        "boolean_value": df["boolean_value"].to_numpy(),
    }


def main(args):
    features_data_dir = Path(args.features_data_dir)
    output_dir = Path(args.output_dir)
    feature_processor_path = output_dir / "feature_processor.pickle"
    logistic_dir = output_dir / "logistic"
    logistic_dir.mkdir(exist_ok=True, parents=True)
    gbm_dir = output_dir / "gbm"
    gbm_dir.mkdir(exist_ok=True, parents=True)
    logistic_test_result_file = logistic_dir / "metrics.json"
    gbm_test_result_file = gbm_dir / "metrics.json"
    if logistic_test_result_file.exists() and gbm_test_result_file.exists():
        print("The models have been trained, and skip ...")
        exit(0)

    feature_train = pd.read_parquet(features_data_dir / "train" / "features")
    feature_test = pd.read_parquet(features_data_dir / "test" / "features")

    if feature_processor_path.exists():
        with open(feature_processor_path, "rb") as f:
            feature_processor = pickle.load(f)
    else:
        age_scaler, gender_encoder, race_encoder = (
            StandardScaler(),
            OneHotEncoder(),
            OneHotEncoder(),
        )
        age_scaler = age_scaler.fit(feature_train[["age_at_index"]].to_numpy())
        gender_encoder = gender_encoder.fit(
            feature_train[["gender_concept_id"]].to_numpy()
        )
        race_encoder = race_encoder.fit(feature_train[["gender_concept_id"]].to_numpy())
        feature_processor = {
            "age_scaler": age_scaler,
            "gender_encoder": gender_encoder,
            "race_encoder": race_encoder,
        }
        with open(feature_processor_path, "wb") as f:
            pickle.dump(feature_processor, f)

    if logistic_test_result_file.exists():
        print(
            f"The results for logistic regression already exist at {logistic_test_result_file}"
        )
    else:
        train_dataset = prepare_dataset(feature_train, feature_processor)
        test_dataset = prepare_dataset(feature_test, feature_processor)
        # Train logistic regression
        model = LogisticRegressionCV(scoring="roc_auc")
        model.fit(train_dataset["features"], train_dataset["boolean_value"])
        y_pred = model.predict_log_proba(test_dataset["features"])[:, 1]

        logistic_predictions = pd.DataFrame(
            {
                "subject_id": test_dataset["subject_id"].tolist(),
                "prediction_time": test_dataset["prediction_time"].tolist(),
                "predicted_boolean_probability": y_pred.tolist(),
                "predicted_boolean_value": None,
                "boolean_value": test_dataset["boolean_value"].astype(bool).tolist(),
            }
        )
        logistic_test_predictions = logistic_dir / "test_predictions"
        logistic_test_predictions.mkdir(exist_ok=True, parents=True)
        logistic_predictions.to_parquet(
            logistic_test_predictions / "test_gbm_predictions.parquet"
        )

        roc_auc = roc_auc_score(test_dataset["boolean_value"], y_pred)
        precision, recall, _ = precision_recall_curve(
            test_dataset["boolean_value"], y_pred
        )
        pr_auc = auc(recall, precision)

        metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}
        print("Logistic:", features_data_dir.name, metrics)
        with open(logistic_test_result_file, "w") as f:
            json.dump(metrics, f, indent=4)

    if gbm_test_result_file.exists():
        print(f"The results for GBM already exist at {gbm_test_result_file}")
    else:
        lightgbm_study = optuna.create_study()  # Create a new study.
        train_split, dev_split = train_test_split(feature_train, test_size=0.2)
        train_data = prepare_dataset(train_split, feature_processor)
        dev_data = prepare_dataset(dev_split, feature_processor)
        lightgbm_study.optimize(
            functools.partial(
                lightgbm_objective, train_data=train_data, dev_data=dev_data
            ),
            n_trials=10,
        )
        print("Computing predictions")
        best_num_trees = lightgbm_study.best_trial.user_attrs["num_trees"]
        best_params = lightgbm_study.best_trial.params
        best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
        full_train_set = prepare_dataset(feature_train)
        dtrain_final = lgb.Dataset(
            full_train_set["features"], label=full_train_set["boolean_value"]
        )
        gbm_final = lgb.train(best_params, dtrain_final, num_boost_round=best_num_trees)

        test_data = prepare_dataset(feature_test, feature_processor)
        lightgbm_preds = gbm_final.predict(test_data["features"], raw_score=False)

        lightgbm_predictions = pd.DataFrame(
            {
                "subject_id": test_data["subject_id"].tolist(),
                "prediction_time": test_data["prediction_time"].tolist(),
                "predicted_boolean_probability": lightgbm_preds.tolist(),
                "predicted_boolean_value": None,
                "boolean_value": test_data["boolean_value"].astype(bool).tolist(),
            }
        )
        gbm_test_predictions = gbm_dir / "test_predictions"
        gbm_test_predictions.mkdir(exist_ok=True, parents=True)
        lightgbm_predictions.to_parquet(
            gbm_test_predictions / "test_gbm_predictions.parquet"
        )

        final_lightgbm_auroc2 = -roc_auc_score(
            test_data["boolean_value"], lightgbm_preds
        )
        gbm_precision, gbm_recall, _ = precision_recall_curve(
            test_data["boolean_value"], lightgbm_preds
        )
        gbm_pr_auc = auc(gbm_recall, gbm_precision)
        final_lightgbm_auroc = lightgbm_objective(
            lightgbm_study.best_trial,
            train_data=train_data,
            dev_data=dev_data,
            num_trees=lightgbm_study.best_trial.user_attrs["num_trees"],
        )
        lightgbm_results = {
            "label_name": features_data_dir.name,
            "final_lightgbm_auroc": final_lightgbm_auroc,
            "final_lightgbm_auroc2": final_lightgbm_auroc2,
            "pr_auc": gbm_pr_auc,
        }
        print("gbm:", features_data_dir.name, lightgbm_results)
        with open(gbm_test_result_file, "w") as f:
            json.dump(lightgbm_results, f, indent=4)


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
    main(parser.parse_args())
