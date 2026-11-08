# Lab 13 — End-to-End Integration and Automation

## Purpose

Lab 13 executes the Credit Card Fraud labs in their numbered order and creates one machine-readable execution record. It invokes the existing Labs 1–11 scripts unchanged, then runs the new Lab 12 monitor.

## Run

From the project root:

```powershell
$python = "C:\v\fraud\Scripts\python.exe"
& $python src\mlops_pipeline\13_end_to_end_pipeline.py
```

The orchestrator uses the current Python executable, runs each lab as a separate process, stops on the first non-zero exit code, and records the result and duration for each executed stage.

## Stage order

| Lab | Script |
|---|---|
| 1 | `01_load_data.py` |
| 2 | `02_validate_data.py` |
| 3 | `03_preprocessing.py` |
| 4 | `04_validate_outputs.py` |
| 5 | `05_train_registry.py` |
| 6 | `06_model_evaluation.py` |
| 7 | `07_quality_gate.py` |
| 8 | `08_automate_lifecycle.py` |
| 9 | `09_inference.py` |
| 10 | `10_failure_injection.py` |
| 11 | `11_prepare_deployment.py` |
| 12 | `12_monitoring.py` |
| 13 | `13_end_to_end_pipeline.py` (the orchestration and final status record) |

When Lab 12 triggers retraining, it calls the existing Labs 1–11 orchestrator. Lab 13 does not recursively call itself.

## Result

`logs/lab13_execution_report.json` contains the overall status, timestamps, each lab's script/status/exit code/duration, and the Lab 13 orchestration status. The report is written on success and on the first stage failure.

## Scope and data safety

Labs 1–11 retain their existing files and behavior. Lab 12 uses the existing raw/processed dataset, model, MLflow settings, reports, and dependency set. No source dataset is overwritten, and all Lab 12/13 outputs remain in this project.
