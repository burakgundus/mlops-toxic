"""Batch ordering, validation and model failure behavior."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main
from tests.container_model import create_model


def test_batch_with_serialized_pyfunc_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODEL_URI", str(create_model(tmp_path)))
    with TestClient(main.app) as client:
        response = client.post(
            "/predict/batch",
            json={"comments": [{"text": "good nice"}, {"text": "terrible hate"}]},
        )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["is_toxic"] for result in results] == [False, True]
    assert all(set(result["probabilities"]) == set(main.LABELS) for result in results)


def test_batch_uses_one_model_call_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class Model:
        def predict(self, data: pd.DataFrame) -> pd.DataFrame:
            calls.append(data["comment_text"].tolist())
            return pd.DataFrame({label: [0.1, 0.9] for label in main.LABELS})

    monkeypatch.setattr(main, "model_pipeline", Model())
    client = TestClient(main.app)
    response = client.post(
        "/predict/batch", json={"comments": [{"text": " hello "}, {"text": "bad"}]}
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert calls == [["hello", "bad"]]
    assert [item["text"] for item in results] == ["hello", "bad"]
    assert [item["is_toxic"] for item in results] == [False, True]


@pytest.mark.parametrize("comments", [[], [{"text": "x"}] * 101, [{"text": ""}]])
def test_invalid_batches(comments: list[dict[str, str]]) -> None:
    assert (
        TestClient(main.app).post("/predict/batch", json={"comments": comments}).status_code == 422
    )


def test_whitespace_batch_is_rejected_before_inference() -> None:
    response = TestClient(main.app).post("/predict/batch", json={"comments": [{"text": "  "}]})
    assert response.status_code == 400


def test_batch_missing_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "model_pipeline", None)
    response = TestClient(main.app).post("/predict/batch", json={"comments": [{"text": "hi"}]})
    assert response.status_code == 503


def test_batch_rejects_invalid_later_row(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def predict(self, data: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({label: [0.1, float("nan")] for label in main.LABELS})

    monkeypatch.setattr(main, "model_pipeline", Model())
    response = TestClient(main.app).post(
        "/predict/batch", json={"comments": [{"text": "one"}, {"text": "two"}]}
    )
    assert response.status_code == 500
