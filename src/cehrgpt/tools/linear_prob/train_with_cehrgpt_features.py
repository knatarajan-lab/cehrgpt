import argparse
import functools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split


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


def main(args):
    features_data_dir = Path(args.features_data_dir)
    output_dir = Path(args.output_dir)
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
    # Check if features need flattening and flatten if necessary
    train_features = np.stack(
        feature_train["features"].apply(lambda x: np.array(x).flatten())
    )
    train_labels = feature_train["boolean_value"].to_numpy()

    feature_test = pd.read_parquet(features_data_dir / "test" / "features")
    # Similarly flatten test features
    test_features = np.stack(
        feature_test["features"].apply(lambda x: np.array(x).flatten())
    )
    if not logistic_test_result_file.exists():
        # Train logistic regression
        model = LogisticRegressionCV(scoring="roc_auc")
        model.fit(train_features, train_labels)
        y_pred = model.predict_log_proba(test_features)[:, 1]

        logistic_predictions = pd.DataFrame(
            {
                "subject_id": feature_test["subject_ids"].tolist(),
                "prediction_time": feature_test["prediction_times"].tolist(),
                "predicted_boolean_probability": y_pred.tolist(),
                "predicted_boolean_value": None,
                "boolean_value": feature_test["boolean_value"].astype(bool).tolist(),
            }
        )
        logistic_test_predictions = logistic_dir / "test_predictions"
        logistic_test_predictions.mkdir(exist_ok=True, parents=True)
        logistic_predictions.to_parquet(
            logistic_test_predictions / "test_gbm_predictions.parquet"
        )

        roc_auc = roc_auc_score(feature_test["boolean_value"], y_pred)
        precision, recall, _ = precision_recall_curve(
            feature_test["boolean_value"], y_pred
        )
        pr_auc = auc(recall, precision)

        metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}
        print("Logistic:", features_data_dir.name, metrics)
        with open(logistic_test_result_file, "w") as f:
            json.dump(metrics, f, indent=4)

    if not gbm_test_result_file.exists():
        lightgbm_study = optuna.create_study()  # Create a new study.
        train_df, dev_df = train_test_split(feature_train, test_size=0.2)
        lightgbm_study.optimize(
            functools.partial(lightgbm_objective, train_data=train_df, dev_data=dev_df),
            n_trials=10,
        )
        print("Computing predictions")
        best_num_trees = lightgbm_study.best_trial.user_attrs["num_trees"]
        best_params = lightgbm_study.best_trial.params
        best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
        dtrain_final = lgb.Dataset(
            feature_train["features"], label=feature_train["boolean_value"]
        )
        gbm_final = lgb.train(best_params, dtrain_final, num_boost_round=best_num_trees)
        lightgbm_preds = gbm_final.predict(feature_test["features"], raw_score=False)

        lightgbm_predictions = pd.DataFrame(
            {
                "subject_id": feature_test["subject_ids"].tolist(),
                "prediction_time": feature_test["prediction_times"].tolist(),
                "predicted_boolean_probability": lightgbm_preds.tolist(),
                "predicted_boolean_value": None,
                "boolean_value": feature_test["boolean_value"].astype(bool).tolist(),
            }
        )
        gbm_test_predictions = gbm_dir / "test_predictions"
        gbm_test_predictions.mkdir(exist_ok=True, parents=True)
        lightgbm_predictions.to_parquet(
            gbm_test_predictions / "test_gbm_predictions.parquet"
        )

        final_lightgbm_auroc2 = -roc_auc_score(
            feature_test["boolean_value"], lightgbm_preds
        )
        gbm_precision, gbm_recall, _ = precision_recall_curve(
            feature_test["boolean_value"], lightgbm_preds
        )
        gbm_pr_auc = auc(gbm_recall, gbm_precision)
        final_lightgbm_auroc = lightgbm_objective(
            lightgbm_study.best_trial,
            train_data=feature_train,
            dev_data=feature_test,
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
