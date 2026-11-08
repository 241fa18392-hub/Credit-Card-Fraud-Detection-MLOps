# Lab 9 — Failure Handling

Failure tests are implemented in `src/mlops_pipeline/10_failure_injection.py`.
The current local run detected 5/5 simulated failure conditions:
missing target column, invalid target label, missing feature value, infinite feature,
and missing model path. Tests use temporary/in-memory corruption and do not delete
the real dataset or trained model.

Run: `python src/mlops_pipeline/10_failure_injection.py`.
Result: `reports/failure_report.json`.
