# mlops-toxic

End-to-end toxic comment detection with a reproducible pipeline: environment setup, data versioning and validation, experiment tracking, and FastAPI serving with automated CI/CD.

Dataset: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`train.csv`, 159,571 comments, six binary labels).

## Project status

- **Setup:** Done. Poetry, pre-commit (ruff, ruff-format, mypy), Docker Compose (PostgreSQL + MLflow), profile-based config (`config/`).
- **Data:** Done. DVC versioning (local remote), Great Expectations validation, staged data layers.
- **Model:** Done. TF-IDF + Logistic Regression (one model per label), Optuna hyperparameter search, MLflow tracking and model registry (`toxic-clf`).
- **CI/CD & Testing:** Done. GitHub Actions CI pipeline (lint, format, mypy type-checking, pytest unit/integration tests).
- **Serving:** Done. FastAPI multi-label inference service.
- **Containerization & Deployment:** In Progress / Planned (Multi-stage Docker build, CD pipeline).
- **Monitoring:** Planned (Evidently, Prometheus, Grafana).

## Requirements

- Windows with Docker Desktop
- Python 3.11
- Poetry

## Setup

```powershell
git clone [https://github.com/burakgundus/mlops-toxic.git](https://github.com/burakgundus/mlops-toxic.git)
cd mlops-toxic
poetry install
poetry run pre-commit install
docker compose -f docker\docker-compose.yml up -d
MLflow UI: http://localhost:5000

Data
The raw data is not stored in Git. Git keeps only the pointer file data/01-raw/train.csv.dvc.

Download train.csv from the Kaggle page above, place it at data/01-raw/train.csv, and check that its MD5 hash matches the one in data/01-raw/train.csv.dvc.

Data layers:

data/01-raw/           raw input (DVC-tracked)
data/02-preprocessed/  validated and cleaned data
data/03-features/      TF-IDF features and vectorizer
data/04-predictions/   inference output
Run the pipelines
Training pipeline (validate, preprocess, build features, train, register in MLflow):

PowerShell
poetry run python -m entrypoint.train
Batch Inference (loads the registered model and writes predictions):

PowerShell
poetry run python -m entrypoint.inference
Optuna hyperparameter search (nested MLflow runs under optuna-search):

PowerShell
poetry run python -m src.pipelines.tuning_pipeline
Run the API & Tests
Run test suite:

PowerShell
poetry run pytest
Start FastAPI inference service:

PowerShell
poetry run uvicorn src.api.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/docs

Configuration
Settings live in config/local.yaml and config/prod.yaml. The profile is chosen with the APP_ENV environment variable (default: local):

PowerShell
$env:APP_ENV = "prod"
Project structure
.github/workflows/ CI pipeline (GitHub Actions: lint, type-check, tests)
config/            environment profiles (local.yaml, prod.yaml)
data/              staged DVC data layers (01-raw to 04-predictions)
docker/            docker-compose.yml and MLflow Dockerfile
entrypoint/        train.py and inference.py entry scripts
src/api/           FastAPI application and route handlers
src/config.py      config loader
src/pipelines/     feature_eng, training, tuning and inference pipelines
tests/             unit and integration test suite
Notes
The DVC remote is currently a local folder on one machine. A shared remote is planned.

MinIO is not part of the stack yet; MLflow stores artifacts in a Docker volume.

Experiment results are stored in MLflow and not in Git.
