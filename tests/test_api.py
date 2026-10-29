"""Serving contract and real pyfunc serialization tests; no registry required."""

from collections.abc import Iterator
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main
from tests.container_model import create_model


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return create_model(tmp_path)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("MODEL_URI", str(model_dir))
    with TestClient(main.app) as http:
        yield http
    assert main.model_pipeline is None


def test_real_pyfunc_model_serves_six_probabilities(client: TestClient) -> None:
    assert client.get("/ready").status_code == 200
    assert client.get("/health").json()["model_loaded"] is True
    response = client.post("/predict", json={"text": "  terrible hate  "})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "terrible hate"
    assert set(body["probabilities"]) == set(main.LABELS)
    assert all(0 <= value <= 1 for value in body["probabilities"].values())
    assert body["flagged_labels"] == list(main.LABELS)
    assert body["is_toxic"] is True


@pytest.mark.parametrize("text,status", [("", 422), ("   ", 400), ("a" * 20001, 422)])
def test_invalid_input(client: TestClient, text: str, status: int) -> None:
    assert client.post("/predict", json={"text": text}).status_code == status


def test_missing_model_does_not_return_fake_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(uri: str, dst_path: str) -> None:
        raise RuntimeError("Registry unavailable")

    monkeypatch.setattr(mlflow.pyfunc, "load_model", unavailable)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["model_loaded"] is False
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"text": "hello"}).status_code == 503


def test_registry_loading_writes_metadata_only_to_private_cache(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path, tmp_path: Path
) -> None:
    from mlflow.tracking import MlflowClient

    tracking_uri = (tmp_path / "registry").as_uri()
    registry = MlflowClient(tracking_uri=tracking_uri)
    registry.create_registered_model("smoke")
    version = registry.create_model_version("smoke", source=model_dir.as_uri())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MODEL_URI", f"models:/smoke/{version.version}")
    original_load = mlflow.pyfunc.load_model
    cache_paths: list[Path] = []

    def load_with_cache(uri: str, dst_path: str) -> main.ProbabilityModel:
        cache_paths.append(Path(dst_path))
        return original_load(uri, dst_path=dst_path)

    monkeypatch.setattr(mlflow.pyfunc, "load_model", load_with_cache)
    with TestClient(main.app) as client:
        assert client.get("/ready").status_code == 200
        assert client.post("/predict", json={"text": "terrible hate"}).status_code == 200
        assert len(cache_paths) == 1
        assert list(cache_paths[0].rglob("registered_model_meta"))
        assert not (model_dir / "registered_model_meta").exists()
    assert not cache_paths[0].exists()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_return_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, invalid: float
) -> None:
    class InvalidModel:
        def predict(self, data: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({label: [invalid] for label in main.LABELS})

    monkeypatch.setattr(main, "model_pipeline", InvalidModel())
    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Model inference failed."


def test_wrong_output_schema_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongModel:
        def predict(self, data: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"other": [0.5]})

    monkeypatch.setattr(main, "model_pipeline", WrongModel())
    assert client.post("/predict", json={"text": "hello"}).status_code == 500
