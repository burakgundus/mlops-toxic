from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import load_config

N_TRIALS = 6


def train_and_score(df: pd.DataFrame, cfg: dict[str, Any], max_features: int, C: float) -> float:
    labels = cfg["labels"]
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["comment_text"],
        df[labels],
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=df["toxic"],
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(cfg["model"]["ngram_range"]),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    aucs: list[float] = []
    for label in labels:
        model = LogisticRegression(
            C=C,
            solver=cfg["model"]["solver"],
            random_state=cfg["split"]["random_state"],
            max_iter=cfg["model"]["max_iter"],
        )
        model.fit(X_train, y_train[label])
        aucs.append(roc_auc_score(y_val[label], model.predict_proba(X_val)[:, 1]))
    return float(np.mean(aucs))


def run(cfg: dict[str, Any]) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = pd.read_csv(cfg["data"]["preprocessed_path"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    def objective(trial: optuna.Trial) -> float:
        max_features = trial.suggest_categorical("max_features", [20000, 50000, 100000])
        C = trial.suggest_float("C", 0.5, 10.0, log=True)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params({"max_features": max_features, "C": C, "search": "optuna"})
            mean_auc = train_and_score(df, cfg, max_features, C)
            mlflow.log_metric("mean_auc", mean_auc)
            print(f"trial {trial.number}: max_features={max_features} C={C:.3f} auc={mean_auc:.4f}")
        return mean_auc

    with mlflow.start_run(run_name="optuna-search"):
        study = optuna.create_study(direction="maximize", study_name="toxic-clf-tune")
        study.optimize(objective, n_trials=N_TRIALS)
        best = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_mean_auc", best.value)
        print(f"best: {best.params} mean ROC-AUC={best.value:.4f}")


if __name__ == "__main__":
    run(load_config())
