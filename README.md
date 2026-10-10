\# mlops-toxic



End-to-end toxic comment detection with a reproducible pipeline: environment setup, data versioning and validation, experiment tracking, and (planned) API deployment and monitoring.



Dataset: \[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`train.csv`, 159,571 comments, six binary labels).



\## Project status



\- \*\*Setup:\*\* Done. Poetry, pre-commit (ruff, ruff-format, mypy), Docker Compose (PostgreSQL + MLflow).

\- \*\*Data:\*\* Done. DVC versioning with a local remote, Great Expectations validation.

\- \*\*Model:\*\* Baseline done (TF-IDF + Logistic Regression, one model per label). Optuna search done. Model registered as `toxic-clf` version 1 in Staging.

\- \*\*Deploy:\*\* Planned (FastAPI, multi-stage Docker, GitHub Actions).

\- \*\*Monitor:\*\* Planned (Evidently, Prometheus, Grafana).



\## Requirements



\- Windows with Docker Desktop

\- Python 3.11

\- Poetry

\- Git and DVC (installed through Poetry)



\## Setup



```powershell

git clone https://github.com/burakgundus/mlops-toxic.git

cd mlops-toxic

poetry install

poetry run pre-commit install

docker compose -f docker\\docker-compose.yml up -d

```



MLflow UI: http://localhost:5000



\## Data



The raw data is not stored in Git. Git keeps only the pointer file `data/raw/train.csv.dvc`.



To get the data, either:



\- download `train.csv` from the Kaggle competition page above and place it at `data/raw/train.csv`, then check its MD5 hash against `data/raw/train.csv.dvc`, or

\- run `dvc checkout data/raw/train.csv.dvc` if you have access to the DVC remote.



Validate the data:



```powershell

poetry run python -m src.data.validate

```



Expected output: `All 15 expectations passed.`



\## Training



Baseline (one TF-IDF + Logistic Regression per label, logged to MLflow and registered as `toxic-clf`):



```powershell

poetry run python -m src.models.train

```



Optuna hyperparameter search (nested MLflow runs under `optuna-search`):



```powershell

poetry run python -m src.models.tune

```



\## Project structure



```

data/        raw data pointers (DVC); raw CSV is not committed

src/data/    data validation (Great Expectations)

src/models/  training, tuning, and the MLflow pyfunc model wrapper

docker/      docker-compose.yml and MLflow Dockerfile

tests/       planned

.github/     planned (GitHub Actions workflows)

```



\## Notes



\- The DVC remote is currently a local folder (`C:\\dev\\dvc-storage`). Other machines need their own remote to run `dvc pull`.

\- MinIO is not part of the stack yet; MLflow stores artifacts in a Docker volume.

\- Experiment results are stored in MLflow and not in Git.

