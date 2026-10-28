from src.config import load_config


def test_local_profile():
    cfg = load_config("local")
    assert cfg["mlflow"]["tracking_uri"] == "http://localhost:5000"
    assert cfg["data"]["raw_path"] == "data/01-raw/train.csv"


def test_prod_profile_uses_service_name():
    cfg = load_config("prod")
    assert cfg["mlflow"]["tracking_uri"] == "http://mlflow:5000"


def test_six_labels():
    cfg = load_config("local")
    assert len(cfg["labels"]) == 6
    assert cfg["labels"][0] == "toxic"


def test_profile_comes_from_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    assert load_config()["mlflow"]["tracking_uri"] == "http://mlflow:5000"
