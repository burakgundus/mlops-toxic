from contextlib import asynccontextmanager
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from pydantic import BaseModel, Field

# 6 ana toksiklik etiketi (Jigsaw Dataset)
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Global model referansı (Startup'ta bir kez yüklenecek)
model_pipeline: Optional[object] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modeli API ayağa kalkarken (startup) bir kez hafızaya yükler."""
    global model_pipeline
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # MLflow Model Registry'den 'Staging' aşamasındaki modeli yüklemeyi dener
    model_uri = "models:/toxic-clf/Staging"
    try:
        print(f"[*] Loading model from {model_uri} via {mlflow_tracking_uri}...")
        model_pipeline = mlflow.sklearn.load_model(model_uri)
        print("[+] Model loaded successfully from MLflow Registry.")
    except Exception as e:
        print(f"[!] MLflow Registry model could not be loaded: {e}")
        print("[*] Running in mock/fallback mode for development testing.")
        model_pipeline = None

    yield
    print("[*] Shutting down inference service...")


app = FastAPI(
    title="Toxic Language Detection API",
    description="Multi-label toxic comment classification service with 6 probability outputs.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Şemaları ---
class CommentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Classified comment text",
        examples=["You are a terrible person and I hate you."],
    )


class CommentResponse(BaseModel):
    text: str
    probabilities: Dict[str, float]
    is_toxic: bool
    flagged_labels: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    environment: str


# --- Endpoint'ler ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Servisin ve modelin ayakta olup olmadığını kontrol eder."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_pipeline is not None,
        environment=os.getenv("APP_ENV", "local"),
    )


@app.post("/predict", response_model=CommentResponse, tags=["Inference"])
async def predict_comment(request: CommentRequest):
    """Gelen metni 6 farklı toksiklik türü için sınıflandırır ve olasılıkları döner."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")

    probabilities: Dict[str, float] = {}

    if model_pipeline is not None:
        try:
            # Model Pipeline'dan 6 etiket için olasılıkları alma
            # (MultiOutputClassifier veya 6 ayrı model yapısına uygun çıkarım)
            preds = model_pipeline.predict_proba([text])
            for idx, label in enumerate(LABELS):
                # Her etiketin pozitif sınıf (1) olasılığı
                prob = float(preds[idx][0][1]) if isinstance(preds, list) else float(preds[0][idx])
                probabilities[label] = round(prob, 4)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    else:
        # Fallback/Test Modu: MLflow offline olsa bile Swagger UI testlerinin çalışmasını sağlar
        has_bad_words = any(
            word in text.lower() for word in ["terrible", "hate", "idiot", "kill", "stupid"]
        )
        base_prob = 0.88 if has_bad_words else 0.04
        probabilities = {
            "toxic": round(base_prob, 4),
            "severe_toxic": round(base_prob * 0.2, 4),
            "obscene": round(base_prob * 0.7, 4),
            "threat": round(base_prob * 0.1, 4),
            "insult": round(base_prob * 0.8, 4),
            "identity_hate": round(base_prob * 0.15, 4),
        }

    # Eşik değer: Herhangi bir etiketin olasılığı %50'den büyükse etiketlenir
    flagged = [label for label, prob in probabilities.items() if prob >= 0.5]
    is_toxic = len(flagged) > 0

    return CommentResponse(
        text=text,
        probabilities=probabilities,
        is_toxic=is_toxic,
        flagged_labels=flagged,
    )
