# Lab 8 — CI and Model Lifecycle

GitHub Actions workflow: `.github/workflows/lab8_ci.yml`.
It installs requirements, compiles source files, and runs pytest. The lifecycle
script promotes the candidate only after the configured quality gate passes,
assigns the MLflow `production` alias, and updates the local Production manifest.
The verified local run registered `Credit_Card_Fraud_Production_Model` version 3.

Local lifecycle script: `python src/mlops_pipeline/08_automate_lifecycle.py`.

Note: The workflow file is prepared locally; a GitHub Actions run requires pushing
the repository to GitHub. Local pytest and the complete 11-lab orchestrator passed;
this is not evidence of a remote Actions run.
