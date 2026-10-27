from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import load_config


def run(cfg: dict[str, Any], input_path: str | None = None, stage: str = "Staging") -> Path:
    ml = cfg["mlflow"]
    mlflow.set_tracking_uri(ml["tracking_uri"])
    model = mlflow.pyfunc.load_model(f"models:/{ml['registered_model_name']}/{stage}")

    if input_path is None:
        df = pd.read_csv(cfg["data"]["preprocessed_path"], nrows=100)
    else:
        df = pd.read_csv(input_path)

    probs = model.predict(df[["comment_text"]])
    keep = [c for c in ["id", "comment_text"] if c in df.columns]
    out = pd.concat([df[keep].reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    out_dir = Path(cfg["data"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"{len(out)} predictions saved to {out_path}")
    return out_path


if __name__ == "__main__":
    run(load_config())
