# Credit Card Fraud Detection — End-to-End MLOps

This project runs a thirteen-lab workflow on the existing credit-card transaction dataset. It preserves the original CSV and keeps all fraud source code, models, registry records, tracking data, configuration, and outputs inside this project.

- **Dataset:** `data/raw/creditcard.csv`
- **Target:** `Class` (`0` = normal, `1` = fraud)
- **Model:** class-weighted Logistic Regression
- **API:** FastAPI single/batch-record inference endpoint
- **Tracking/registry:** project-local MLflow store and model registry, plus a local lifecycle manifest

Architecture details are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The measured run and changes made during verification are recorded in [`docs/EXECUTION_RESULTS.md`](docs/EXECUTION_RESULTS.md).

## Dataset

The source file contains 284,807 rows and 31 columns: `Time`, `V1`–`V28`, `Amount`, and `Class`. The original dataset has 284,315 normal transactions, 492 fraudulent transactions, no missing values, and 1,081 exact duplicate rows. The preprocessing pipeline removes exact duplicates from the derived training/evaluation data before a stratified split; it does not rewrite `data/raw/creditcard.csv`.

The held-out run used 226,980 training rows and 56,746 test rows. The scaler was fitted only on training data.

### Obtain the source dataset

The raw dataset is intentionally excluded from GitHub because it is large and may have separate redistribution terms. Obtain `creditcard.csv` from the dataset provider authorized for your use, then place it locally at:

```text
data/raw/creditcard.csv
```

The pipeline expects the columns `Time`, `V1` through `V28`, `Amount`, and `Class`, with `Class` containing `0` for normal transactions and `1` for fraud. Keep the downloaded original unchanged; preprocessing writes derived files under `data/processed/`. The raw CSV and generated processed datasets are ignored by Git.

## Labs 1–13

Labs 1–11 retain their existing implementations. Labs 12 and 13 add monitoring/retraining and a full ordered orchestrator.

| Lab | Script | Purpose |
|---|---|---|
| 1 | `01_load_data.py` | Load/profile the existing CSV and record schema, missing values, and duplicates |
| 2 | `02_validate_data.py` | Check required columns, numeric features, and binary target labels |
| 3 | `03_preprocessing.py` | Remove exact duplicates in derived data; stratify split; fit scaler on training rows |
| 4 | `04_validate_outputs.py` | Check shapes, labels, finite values, and processed outputs |
| 5 | `05_train_registry.py` | Train Logistic Regression; log the model and register an MLflow model version |
| 6 | `06_model_evaluation.py` | Compute metrics/plots and log measured metrics to the same MLflow run |
| 7 | `07_quality_gate.py` | Require configured average precision of at least 0.50 |
| 8 | `08_automate_lifecycle.py` | Promote only after the gate; assign the MLflow `production` alias and update the local manifest |
| 9 | `09_inference.py` | Load the Production candidate and write sample predictions |
| 10 | `10_failure_injection.py` | Verify five controlled data/model failure conditions are detected |
| 11 | `11_prepare_deployment.py` | Copy and reload the model/preprocessor bundle and write deployment metadata |
| 12 | `12_monitoring.py` | Monitor held-out performance, feature drift, and prediction drift; trigger the existing pipeline when configured thresholds are exceeded |
| 13 | `13_end_to_end_pipeline.py` | Run Labs 1–12 in order and write a status/duration report for each stage |

Lab 11 in this repository means deployment-bundle preparation. It does not claim that a Docker image or Kubernetes deployment was run.

## Installation and execution (Windows PowerShell)

Run from this project directory. A short environment path avoids Windows' legacy maximum-path issue:

```powershell
python -m venv C:\v\fraud
$python = "C:\v\fraud\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python src\mlops_pipeline\13_end_to_end_pipeline.py
```

Run Labs 1–11 only, or Lab 12 on the latest Labs 1–11 artifacts:

```powershell
& $python src\mlops_pipeline\fraud_detection_pipeline.py
& $python src\mlops_pipeline\12_monitoring.py
```

By default, Lab 12 starts the existing Labs 1–11 pipeline only when performance degradation or configured drift thresholds trigger retraining. Use `--no-retrain` to generate monitoring/decision reports without launching training.

Run the complete Lab 1–13 workflow:

```powershell
& $python src\mlops_pipeline\13_end_to_end_pipeline.py
```

The Lab 13 report is written to `logs/lab13_execution_report.json`. Lab 12 writes `reports/performance_monitoring_report.json`, `reports/monitoring_baseline.json`, `reports/data_drift_report.json`, `reports/prediction_drift_report.json`, and `reports/retraining_decision_report.json`.

Run tests and syntax compilation:

```powershell
& $python -m pytest -q
& $python -m compileall -q src api
```

Lab documentation: [`docs/LAB12_MONITORING.md`](docs/LAB12_MONITORING.md) and [`docs/LAB13_END_TO_END.md`](docs/LAB13_END_TO_END.md).

Run one lab, for example:

```powershell
& $python src\mlops_pipeline\06_model_evaluation.py
```

The checked-in configuration is `config.yaml`. To use a non-local MLflow tracking server, set `MLFLOW_TRACKING_URI` before running the pipeline.

## Verified model evaluation

Metrics below are from the final executed run on the held-out test set; they are not hard-coded expectations.

| Metric | Value |
|---|---:|
| Accuracy | 0.9753 |
| Precision | 0.0564 |
| Recall | 0.8737 |
| F1 | 0.1059 |
| Average precision (PR-AUC) | 0.7194 |
| ROC-AUC | 0.9657 |
| Quality gate | Passed: 0.7194 >= 0.5000 |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[55262, 1389], [12, 83]]`. Accuracy alone is misleading for this imbalanced dataset; precision, recall, F1, PR-AUC, and ROC-AUC are reported.

## MLflow tracking and registry

- Tracking database: project-local `mlflow.db`; artifact and run folders are project-local as well.
- Experiment: `credit_card_fraud_detection`.
- Registered model: `Credit_Card_Fraud_Production_Model`.
- Verified Production alias: `production` -> MLflow version 7.
- Final run ID: `23f208a4df7c42a8b262c4374765d7b7`.
- The run contains accuracy, precision, recall, F1, average precision, and ROC-AUC.
- `models/local_model_registry.json` and `reports/lifecycle_report.json` preserve the local lifecycle view for offline inspection.

The alias version and the local manifest's version are separate identifiers; use the MLflow alias as the MLflow registry selection.

## API usage

Start the service from the project root:

```powershell
& $python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- `GET /health` reports API status.
- `POST /predict` accepts `{"records": [...]}`; each record must contain exactly 30 numeric features in CSV order, excluding `Class`.
- The response contains `predictions` and `fraud_probabilities`.
- Missing deployment artifacts return HTTP 503; malformed feature payloads return HTTP 400.

Example using two existing dataset rows:

```powershell
@'
import json
from urllib.request import Request, urlopen
import pandas as pd

records = pd.read_csv("data/raw/creditcard.csv", nrows=2).drop(columns=["Class"]).values.tolist()
request = Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps({"records": records}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
print(urlopen(request).read().decode("utf-8"))
'@ | & $python -
```

The executed live check returned health `ok`, predictions `[0, 0]`, and fraud probabilities approximately `[0.135905, 0.029413]` for the first two source rows.

## Deployment

Lab 11 prepares `deployment/model.pkl`, `deployment/preprocessor.pkl`, and `deployment/model_metadata.json`. A minimal API image definition is provided in `Dockerfile`; it uses `requirements-api.txt` and does not copy the source dataset.

When Docker is available:

```powershell
docker build -t credit-card-fraud-api:local .
docker run --rm -p 8001:8000 credit-card-fraud-api:local
Invoke-RestMethod http://127.0.0.1:8001/health
```

Docker was unavailable during this execution, so the image build/container run are **not verified**. No Kubernetes manifests are included in this project.

## Tests and operational notes

- `pytest -q`: 9 passed after adding Labs 12 and 13.
- `python -m compileall -q src api`: passed.
- Labs 1–11 and Lab 13 end-to-end orchestrator: passed.
- Lab 12 monitoring and forced retraining path: passed; final integrated monitor detected no drift trigger.
- Failure injection: 5/5 scenarios detected.
- Original dataset SHA-256 remained `76274B691B16A6C49D3F159C883398E03CCD6D1EE12D9D8EE38F4B4B98551A89`.

See [`docs/EXECUTION_RESULTS.md`](docs/EXECUTION_RESULTS.md) for run details and limitations, and [`docs/FACULTY_VIVA.md`](docs/FACULTY_VIVA.md) for review questions and answers.
