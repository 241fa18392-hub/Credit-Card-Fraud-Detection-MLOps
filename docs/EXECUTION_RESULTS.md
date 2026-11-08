# Credit Card Fraud — Execution Results

## Final run

- Date: 2026-11-07
- Python: 3.13.15
- Project-specific environment: `C:\v\fraud`
- Orchestrator: `python src\mlops_pipeline\13_end_to_end_pipeline.py`
- Overall result: Labs 1–13 passed; the orchestrator recorded successful results for Labs 1–12 and its Lab 13 integration step.

## Dataset

- Source: `data/raw/creditcard.csv`
- Rows/columns: 284,807 / 31
- Target: `Class` (`0` normal, `1` fraud)
- Source class counts: 284,315 normal, 492 fraud
- Missing values: 0
- Exact duplicates: 1,081; removed only from derived data before splitting
- Train/test: 226,980 / 56,746; stratified
- Source SHA-256 before and after: `76274B691B16A6C49D3F159C883398E03CCD6D1EE12D9D8EE38F4B4B98551A89` (unchanged)

## Metrics and quality gate

| Metric | Final measured value |
|---|---:|
| Accuracy | 0.975311 |
| Precision | 0.056386 |
| Recall | 0.873684 |
| F1 | 0.105935 |
| Average precision / PR-AUC | 0.719417 |
| ROC-AUC | 0.965686 |
| Confusion matrix (`[[TN, FP], [FN, TP]]`) | `[[55262, 1389], [12, 83]]` |

Quality gate: `average_precision >= 0.50`; observed `0.719417`; **PASSED**.

## MLflow and lifecycle

- Tracking database: project-local `mlflow.db`.
- Experiment: `credit_card_fraud_detection`.
- Registered model: `Credit_Card_Fraud_Production_Model`.
- Final registry version: `7`; alias `production` points to version 7.
- Final run ID: `23f208a4df7c42a8b262c4374765d7b7`.
- Verified run metrics: accuracy, precision, recall, F1, average precision, average_precision_pr_auc, and ROC-AUC.
- Local registry/lifecycle manifest was also written to `models/local_model_registry.json` and `reports/lifecycle_report.json`.

## Inference, failure tests, and deployment

- Lab 9 wrote `outputs/inference_predictions.csv`.
- Lab 10 detected 5/5 injected failure conditions.
- Lab 11 wrote model, preprocessor, and metadata into `deployment/` and reloaded the artifacts successfully.
- Live API `GET /health` returned `ok`; `POST /predict` accepted two source rows and returned predictions `[0, 0]` with probabilities approximately `[0.135905, 0.029413]`.
- Docker image build was not executed: Docker CLI is unavailable. The Dockerfile and API-only requirements were prepared, but container behavior is unverified.
- No Kubernetes manifests are included in this project.

## Tests

- `C:\v\fraud\Scripts\python.exe -m pytest -q`: **9 passed** after Labs 12–13 were added.
- `C:\v\fraud\Scripts\python.exe -m compileall -q src api`: passed.

## Labs 12–13 verification (2026-11-07)

### Lab 12 — monitoring and retraining

- `python src\mlops_pipeline\12_monitoring.py --no-retrain`: passed; average precision `0.719417`, 0/30 drifted features, probability PSI `0.000290`, and retraining not required.
- `python src\mlops_pipeline\12_monitoring.py --force-retrain`: passed; forced the retraining path and completed the existing Labs 1–11 orchestrator, including MLflow registration, quality gate, inference, 5/5 failure checks, and deployment-bundle verification. The dataset itself was not modified.
- Configured triggers cover per-feature PSI, prediction-probability PSI, predicted-positive-rate shift, and average-precision degradation. Baseline persistence is recorded in `reports/monitoring_baseline.json`.

### Lab 13 — end-to-end automation

- `python src\mlops_pipeline\13_end_to_end_pipeline.py`: passed all Labs 1–12 in order; Lab 13 recorded its successful orchestration result.
- Machine-readable evidence: `logs/lab13_execution_report.json`; each stage recorded `PASSED`, exit code 0, and duration.
- Final monitoring reported 0/30 drifted features and no retraining trigger.
- The final run registered MLflow version 7 and assigned the `production` alias. Held-out metrics remained unchanged from the recorded values above.
- Source dataset SHA-256 remained `76274B691B16A6C49D3F159C883398E03CCD6D1EE12D9D8EE38F4B4B98551A89`.

### Tests after adding Labs 12–13

- Focused Lab 12/13 tests: **7 passed**.
- Full fraud test suite: **9 passed**.
- `python -m compileall -q src api`: passed.

## Issues found and changes made

1. Initial execution used MLflow tracking but did not register a model version. `05_train_registry.py` now registers the model under the configured registry name; `08_automate_lifecycle.py` assigns the `production` alias after the quality gate and records the alias in the local manifest.
2. The evaluation stage initially did not attach measured metrics to the MLflow run. `06_model_evaluation.py` now logs its held-out metrics to the run created during training. An initial attempt to use a nonexistent `MlflowClient.log_metrics` method failed; it was corrected to call the installed client's supported `log_metric` method per metric. The final complete pipeline rerun passed, and the alias target's run metrics were queried and verified.
3. Added a project-local `Dockerfile`, `.dockerignore`, and pinned `requirements-api.txt` so the prepared bundle has an API image definition. Docker was not present to build or run the image.
4. Updated the README, status, architecture, and viva documentation to reflect current measurements rather than earlier stale metrics.
5. Added Lab 12 performance/data/prediction monitoring, persisted baselines, threshold-based retraining, and monitoring reports; the forced retraining path executed successfully.
6. Added the Lab 13 sequential orchestrator and JSON stage report. Full end-to-end execution passed all labs; test coverage now validates PSI and retraining decisions as well as lab discovery.

Generated evidence includes `logs/pipeline_run_log.json`, `logs/lab13_execution_report.json`, `artifacts/data_load_report.json`, `artifacts/data_validation_report.json`, `reports/evaluation_report.json`, `reports/quality_gate_report.json`, `reports/failure_report.json`, `reports/lab11_deployment_report.json`, `reports/performance_monitoring_report.json`, `reports/monitoring_baseline.json`, `reports/data_drift_report.json`, `reports/prediction_drift_report.json`, `reports/retraining_decision_report.json`, `models/model_metadata.json`, and `models/local_model_registry.json`.
