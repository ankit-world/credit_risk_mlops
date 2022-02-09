from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from get_data import read_params
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import argparse
import warnings
import joblib
import json
import sys
import os
import mlflow


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return accuracy, precision, recall, roc_auc


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    max_depth = config["estimators"]["RandomForestClassifier"]["params"]["max_depth"]
    max_features = config["estimators"]["RandomForestClassifier"]["params"]["max_features"]
    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        rfc = RandomForestClassifier(max_depth=max_depth,
                                    max_features=max_features,
                                    n_estimators=n_estimators,
                                    random_state=random_state)
        rfc.fit(train_x, train_y)

        predicted_qualities = rfc.predict(test_x)

        (accuracy, precision, recall, roc_auc) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                rfc, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(rfc, "model")

##########################################################


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
