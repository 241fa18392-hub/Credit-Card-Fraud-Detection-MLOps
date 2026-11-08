# Lab 12 — Model Performance and Drift Monitoring

## Purpose

Lab 12 monitors the fraud model produced by Labs 1–11 without changing the source dataset. It evaluates model performance on the existing held-out test split and compares the training population with that held-out monitoring population.

## Monitored signals

- **Performance:** accuracy, precision, recall, F1, average precision (PR-AUC), and ROC-AUC, calculated from `X_test_final.npy` and `y_test.npy`.
- **Data drift:** population stability index (PSI) for each numeric feature, comparing `X_train_raw.csv` with `X_test_raw.csv`.
- **Prediction drift:** PSI of model fraud probabilities and the difference in predicted-positive rates between the training and held-out populations.
- **Retraining trigger:** any feature PSI or probability PSI at/above the configured limit, a positive-rate shift at/above its configured limit, or an average-precision drop at/above the configured limit.
- **Persistent performance baseline:** `reports/monitoring_baseline.json` preserves the last baseline across Lab 13 retraining runs. It is initialized from the existing evaluation report and refreshed after successful Lab 12 retraining.

The source `data/raw/creditcard.csv` is read by Labs 1–3 but never modified by Lab 12. Drift baselines and monitoring rows are derived artifacts from the existing deterministic stratified split.

## Configuration

`config.yaml` has a `lab12` section:

| Setting | Default | Meaning |
|---|---:|---|
| `psi_bins` | 10 | Number of reference-quantile bins |
| `psi_threshold` | 0.20 | PSI threshold for feature/probability drift |
| `performance_metric` | `average_precision_pr_auc` | Metric monitored for degradation |
| `performance_drop_threshold` | 0.05 | Minimum drop from the previous evaluation that triggers retraining |
| `prediction_rate_delta_threshold` | 0.05 | Absolute positive-prediction-rate difference that triggers retraining |
| `retraining_enabled` | `true` | Allow the Labs 1–11 orchestrator to run when a trigger fires |

## Run

Run Labs 1–11 first to create the model and processed split, then from the fraud project root:

```powershell
$python = "C:\v\fraud\Scripts\python.exe"
& $python src\mlops_pipeline\12_monitoring.py
```

For report-only execution, which does not launch retraining:

```powershell
& $python src\mlops_pipeline\12_monitoring.py --no-retrain
```

The monitoring report in `reports/retraining_decision_report.json` records whether retraining was needed, enabled, and completed. A retraining failure returns a non-zero exit code and is recorded as `FAILED`.

## Outputs

- `reports/performance_monitoring_report.json`
- `reports/monitoring_baseline.json`
- `reports/data_drift_report.json`
- `reports/prediction_drift_report.json`
- `reports/retraining_decision_report.json`

The stage fails explicitly if required model, evaluation, or processed-data artifacts are absent or inconsistent.
