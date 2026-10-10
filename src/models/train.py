import os
import subprocess
import tempfile

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.multilabel import ToxicClassifier

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DATA_PATH = "data/raw/train.csv"
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "toxic-clf"
REGISTERED_MODEL_NAME = "toxic-clf"


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    params = {
        "model": "tfidf+logreg",
        "max_features": 50000,
        "ngram_range": "1,2",
        "C": 4.0,
        "solver": "liblinear",
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": 0,
    }

    df = pd.read_csv(DATA_PATH)
    params["train_rows"] = len(df)

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["comment_text"],
        df[LABEL_COLUMNS],
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["toxic"],
    )

    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        mlflow.log_params(params)
        mlflow.set_tag("git_commit", git_commit_hash())

        models: dict[str, LogisticRegression] = {}
        aucs: list[float] = []
        for label in LABEL_COLUMNS:
            model = LogisticRegression(
                C=params["C"],
                solver=params["solver"],
                random_state=params["random_state"],
                max_iter=1000,
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

        # Save vectorizer and models as artifacts, then log one pyfunc model.
        with tempfile.TemporaryDirectory() as tmp:
            vec_path = os.path.join(tmp, "vectorizer.joblib")
            models_path = os.path.join(tmp, "models.joblib")
            joblib.dump(vectorizer, vec_path)
            joblib.dump(models, models_path)

            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=ToxicClassifier(),
                artifacts={"vectorizer": vec_path, "models": models_path},
                registered_model_name=REGISTERED_MODEL_NAME,
            )

        version = model_info.registered_model_version
        print(f"registered {REGISTERED_MODEL_NAME} version {version}")

        client = MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Staging",
        )
        print(f"moved version {version} to Staging")


if __name__ == "__main__":
    main()
