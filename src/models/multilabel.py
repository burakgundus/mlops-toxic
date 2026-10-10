import joblib
import mlflow.pyfunc
import pandas as pd


class ToxicClassifier(mlflow.pyfunc.PythonModel):
    """Six independent logistic regressions sharing one TF-IDF vectorizer."""

    def load_context(self, context) -> None:  # type: ignore[no-untyped-def]
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.models = joblib.load(context.artifacts["models"])

    def predict(self, context, model_input, params=None) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["comment_text"]
        else:
            texts = model_input
        X = self.vectorizer.transform(texts)
        return pd.DataFrame(
            {label: model.predict_proba(X)[:, 1] for label, model in self.models.items()}
        )
