import subprocess

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DATA_PATH = "data/raw/train.csv"
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "toxic-clf"


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
            aucs.append(auc)
            mlflow.log_metric(f"auc_{label}", auc)
            print(f"{label}: ROC-AUC = {auc:.4f}")

        mean_auc = float(np.mean(aucs))
        mlflow.log_metric("mean_auc", mean_auc)
        print(f"mean ROC-AUC = {mean_auc:.4f}")


if __name__ == "__main__":
    main()
