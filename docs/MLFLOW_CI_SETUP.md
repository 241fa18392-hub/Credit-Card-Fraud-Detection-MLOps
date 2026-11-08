# MLflow and CI Setup

Install project dependencies with `pip install -r requirements.txt`.
For local MLflow tracking, run `mlflow ui` from the project root after installation.
Set `MLFLOW_TRACKING_URI` if using a remote tracking server. The training stage logs
the model when MLflow is installed. Without MLflow, local model artifacts and the
local registry manifest still support the project's offline workflow.

GitHub Actions workflows are in `.github/workflows/`.
