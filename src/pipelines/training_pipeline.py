import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.pipelines.toxic_classifier import ToxicClassifier


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def data_md5(raw_path: str) -> str:
    dvc_file = Path(raw_path + ".dvc")
    if not dvc_file.exists():
        return "unknown"
    with open(dvc_file, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return str(meta["outs"][0]["md5"])


def run(cfg: dict[str, Any]) -> None:
    labels = cfg["labels"]
    features_dir = Path(cfg["data"]["features_dir"])

    X_train = sparse.load_npz(features_dir / "X_train.npz")
    X_val = sparse.load_npz(features_dir / "X_val.npz")
    y_train = pd.read_csv(features_dir / "y_train.csv")
    y_val = pd.read_csv(features_dir / "y_val.csv")
    vectorizer = joblib.load(features_dir / "vectorizer.joblib")

    m = cfg["model"]
    params = {
        "model": "tfidf+logreg",
        "max_features": m["max_features"],
        "ngram_range": ",".join(str(n) for n in m["ngram_range"]),
        "C": m["C"],
        "solver": m["solver"],
        "test_size": cfg["split"]["test_size"],
        "random_state": cfg["split"]["random_state"],
        "train_rows": X_train.shape[0],
        "val_rows": X_val.shape[0],
        "data_md5": data_md5(cfg["data"]["raw_path"]),
    }

    ml = cfg["mlflow"]
    mlflow.set_tracking_uri(ml["tracking_uri"])
    mlflow.set_experiment(ml["experiment_name"])

    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        mlflow.log_params(params)
        mlflow.set_tag("git_commit", git_commit_hash())

        models: dict[str, LogisticRegression] = {}
        aucs: list[float] = []
        for label in labels:
            model = LogisticRegression(
                C=m["C"],
                solver=m["solver"],
                random_state=cfg["split"]["random_state"],
                max_iter=m["max_iter"],
            )
            model.fit(X_train, y_train[label])
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val[label], proba)
            models[label] = model
            aucs.append(auc)
            mlflow.log_metric(f"auc_{label}", auc)
            print(f"{label}: ROC-AUC = {auc:.4f}")

        mean_auc = float(np.mean(aucs))
        mlflow.log_metric("mean_auc", mean_auc)
        print(f"mean ROC-AUC = {mean_auc:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            vec_path = os.path.join(tmp, "vectorizer.joblib")
            models_path = os.path.join(tmp, "models.joblib")
            joblib.dump(vectorizer, vec_path)
            joblib.dump(models, models_path)

            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=ToxicClassifier(),
                artifacts={"vectorizer": vec_path, "models": models_path},
                registered_model_name=ml["registered_model_name"],
            )

        version = model_info.registered_model_version
        print(f"registered {ml['registered_model_name']} version {version}")

        client = MlflowClient()
        client.transition_model_version_stage(
            name=ml["registered_model_name"], version=version, stage="Staging"
        )
        print(f"moved version {version} to Staging")


if __name__ == "__main__":
    run(load_config())
