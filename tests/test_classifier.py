import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.pipelines.toxic_classifier import ToxicClassifier

LABELS = load_config("local")["labels"]


def test_classifier_returns_one_probability_per_label():
    texts = ["good nice", "bad awful", "great", "terrible hate"]
    vectorizer = TfidfVectorizer().fit(texts)
    X = vectorizer.transform(texts)
    models = {label: LogisticRegression().fit(X, [0, 1, 0, 1]) for label in LABELS}

    clf = ToxicClassifier()
    clf.vectorizer = vectorizer
    clf.models = models

    out = clf.predict(None, pd.DataFrame({"comment_text": ["good", "awful"]}))
    assert list(out.columns) == LABELS
    assert out.shape == (2, 6)
    assert ((out >= 0) & (out <= 1)).all().all()
