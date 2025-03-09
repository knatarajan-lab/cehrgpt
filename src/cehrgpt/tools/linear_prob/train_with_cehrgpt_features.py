import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def main(args):
    features_data_dir = Path(args.features_data_dir)
    feature_train = pd.read_parquet(features_data_dir / "train" / "features")
    # Check if features need flattening and flatten if necessary
    train_features = np.stack(
        feature_train["features"].apply(lambda x: np.array(x).flatten())
    )
    train_labels = feature_train["boolean_value"].to_numpy()

    model = LogisticRegressionCV(scoring="roc_auc")
    model.fit(train_features, train_labels)

    feature_test = pd.read_parquet(features_data_dir / "test" / "features")
    # Similarly flatten test features
    test_features = np.stack(
        feature_test["features"].apply(lambda x: np.array(x).flatten())
    )
    y_pred = model.predict_log_proba(test_features)[:, 1]

    roc_auc = roc_auc_score(feature_test["boolean_value"], y_pred)
    precision, recall, _ = precision_recall_curve(feature_test["boolean_value"], y_pred)
    pr_auc = auc(recall, precision)

    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    print(features_data_dir.name, metrics)
    test_result_file = Path(args.output_dir) / (
        features_data_dir.name + "_test_results.json"
    )
    with open(test_result_file, "w") as f:
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
