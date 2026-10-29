"""Create a tiny synthetic model for container checks, never for real predictions."""

from pathlib import Path

import joblib
import mlflow.pyfunc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.api.main import LABELS
from src.pipelines.toxic_classifier import ToxicClassifier


def create_model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    texts = ["good nice", "bad awful", "great friendly", "terrible hate"]
    vectorizer = TfidfVectorizer().fit(texts)
    features = vectorizer.transform(texts)
    models = {label: LogisticRegression().fit(features, [0, 1, 0, 1]) for label in LABELS}
    vectorizer_path = directory / "vectorizer.joblib"
    models_path = directory / "models.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(models, models_path)
    path = directory / "model"
    mlflow.pyfunc.save_model(
        path=str(path),
        python_model=ToxicClassifier(),
        artifacts={"vectorizer": str(vectorizer_path), "models": str(models_path)},
        pip_requirements=[],
    )
    return path


if __name__ == "__main__":
    create_model(Path("/models"))
