"""HTTP serving for the pyfunc model registered by the training pipeline."""

import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config

logger = logging.getLogger(__name__)
LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


class ProbabilityModel(Protocol):
    def predict(self, data: pd.DataFrame) -> pd.DataFrame: ...


model_pipeline: ProbabilityModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global model_pipeline
    cfg = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    model_uri = os.getenv("MODEL_URI", f"models:/{cfg['mlflow']['registered_model_name']}/Staging")
    model_pipeline = None
    # Registry loading writes metadata beside the downloaded model. Keep those
    # writes in a private cache, even when the artifact store is mounted read-only.
    with TemporaryDirectory(prefix="toxic-model-") as model_cache:
        try:
            model_pipeline = cast(
                ProbabilityModel, mlflow.pyfunc.load_model(model_uri, dst_path=model_cache)
            )
            logger.info("Loaded model %s", model_uri)
        except Exception:
            logger.exception(
                "Cannot load model %s; restart after the model is available", model_uri
            )
        try:
            yield
        finally:
            model_pipeline = None


app = FastAPI(title="Toxic Language Detection API", version="1.0.0", lifespan=lifespan)


class CommentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000, description="Comment to classify")


class CommentResponse(BaseModel):
    text: str
    probabilities: dict[str, float]
    is_toxic: bool
    flagged_labels: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    environment: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Liveness: the process can handle HTTP requests."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_pipeline is not None,
        environment=os.getenv("APP_ENV", "local"),
    )


@app.get("/ready", response_model=HealthResponse, tags=["Health"])
def readiness_check() -> HealthResponse:
    """Readiness: a model has been loaded for inference."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    return health_check()


@app.post("/predict", response_model=CommentResponse, tags=["Inference"])
def predict_comment(request: CommentRequest) -> CommentResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    try:
        predictions = model_pipeline.predict(pd.DataFrame({"comment_text": [text]}))
        if len(predictions) != 1 or set(predictions.columns) != set(LABELS):
            raise ValueError("Model output must contain one row and all six labels")
        probabilities = {label: float(predictions.iloc[0][label]) for label in LABELS}
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("Model returned an invalid probability")
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc
    flagged = [label for label, probability in probabilities.items() if probability >= 0.5]
    return CommentResponse(
        text=text,
        probabilities={label: round(p, 4) for label, p in probabilities.items()},
        is_toxic=bool(flagged),
        flagged_labels=flagged,
    )
