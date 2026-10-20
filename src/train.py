import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROC = Path("data/processed")

MODELS = {
    "random_forest": lambda p: RandomForestClassifier(**p, random_state=42),
    "gradient_boost": lambda p: GradientBoostingClassifier(**p, random_state=42),
    "logistic": lambda p: LogisticRegression(**p, random_state=42, max_iter=1000),
}


def main():
    cfg = yaml.safe_load(open("params.yaml"))["train"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default=cfg["model_type"])
    parser.add_argument("--n_estimators", type=int, default=cfg.get("n_estimators", 100))
    parser.add_argument("--max_depth", type=int, default=cfg.get("max_depth"))
    parser.add_argument("--C", type=float, default=cfg.get("C", 1.0))
    args = parser.parse_args()

    X_train = pd.read_parquet(PROC / "X_train.parquet")
    X_test = pd.read_parquet(PROC / "X_test.parquet")
    y_train = pd.read_parquet(PROC / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(PROC / "y_test.parquet").squeeze()

    if args.model in ("random_forest", "gradient_boost"):
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    else:
        params = {"C": args.C}

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("titanic-classification")

    desc = "_".join(f"{k}{v}" for k, v in params.items())
    with mlflow.start_run(run_name=f"{args.model}_{desc}"):
        mlflow.set_tags({"dataset": "titanic-v1", "stage": "experiment"})
        mlflow.log_param("model_type", args.model)
        mlflow.log_params(params)

        model = MODELS[args.model](params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="titanic-classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{args.model}: {metrics}")


if __name__ == "__main__":
    main()
