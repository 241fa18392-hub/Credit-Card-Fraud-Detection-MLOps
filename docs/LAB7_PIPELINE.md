# Lab 7 — Quality Gate

The pipeline reads `reports/evaluation_report.json`, checks average precision (PR-AUC)
against the configured threshold in `config.yaml`, and writes
`reports/quality_gate_report.json`. A failed gate stops promotion.

Run: `python src/mlops_pipeline/07_quality_gate.py`

Final executed run: average precision 0.7194; configured minimum 0.5000; gate PASSED.
