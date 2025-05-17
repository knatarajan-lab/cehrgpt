import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_dataset(
    df: pd.DataFrame, feature_processor: Dict[str, Union[StandardScaler, OneHotEncoder]]
) -> Dict[str, Any]:
    age_scaler = feature_processor["age_scaler"]
    gender_encoder = feature_processor["gender_encoder"]
    race_encoder = feature_processor["race_encoder"]
    age_scaler.transform(df[["age_at_index"]].to_numpy())

    one_hot_gender = gender_encoder.transform(
        np.expand_dims(df.gender_concept_id.to_numpy(), axis=1)
    )
    one_hot_race = race_encoder.transform(
        np.expand_dims(df.race_concept_id.to_numpy(), axis=1)
    )

    features = np.stack(df["features"].apply(lambda x: np.array(x).flatten()))
    # features = np.hstack(
    #     [scaled_age, one_hot_gender.toarray(), one_hot_race.toarray(), features]
    # )
    return {
        "subject_id": df["subject_id"].to_numpy(),
        "prediction_time": df["prediction_time"].tolist(),
        "features": features,
        "boolean_value": df["boolean_value"].to_numpy(),
    }


def main(args):
    features_data_dir = Path(args.features_data_dir)
    output_dir = Path(args.output_dir)
    feature_processor_path = output_dir / "feature_processor.pickle"
    logistic_dir = output_dir / "logistic"
    logistic_dir.mkdir(exist_ok=True, parents=True)
    logistic_test_result_file = logistic_dir / "metrics.json"
    if logistic_test_result_file.exists():
        print("The models have been trained, and skip ...")
        exit(0)

    feature_train = pd.read_parquet(
        features_data_dir / "features_with_label" / "train_features"
    )
    feature_test = pd.read_parquet(
        features_data_dir / "features_with_label" / "test_features"
    )

    feature_train = feature_train.sort_values(["subject_id", "prediction_time"]).sample(
        frac=1.0,
        random_state=42,
        replace=False,
    )

    if feature_processor_path.exists():
        with open(feature_processor_path, "rb") as f:
            feature_processor = pickle.load(f)
    else:
        age_scaler, gender_encoder, race_encoder = (
            StandardScaler(),
            OneHotEncoder(handle_unknown="ignore"),
            OneHotEncoder(handle_unknown="ignore"),
        )
        age_scaler = age_scaler.fit(feature_train[["age_at_index"]].to_numpy())
        gender_encoder = gender_encoder.fit(
            feature_train[["gender_concept_id"]].to_numpy()
        )
        race_encoder = race_encoder.fit(feature_train[["race_concept_id"]].to_numpy())
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
        logistic_model_file = logistic_dir / "model.pickle"
        if logistic_model_file.exists():
            print(
                f"The logistic regression model already exist, loading it from {logistic_model_file}"
            )
            with open(logistic_model_file, "rb") as f:
                model = pickle.load(f)
        else:
            train_dataset = prepare_dataset(feature_train, feature_processor)
            # Train logistic regression
            model = LogisticRegressionCV(scoring="roc_auc", random_state=42)
            model.fit(train_dataset["features"], train_dataset["boolean_value"])
            with open(logistic_model_file, "wb") as f:
                pickle.dump(model, f)

        test_dataset = prepare_dataset(feature_test, feature_processor)
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
        logistic_predictions.write_parquet(
            logistic_test_predictions / "predictions.parquet"
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
