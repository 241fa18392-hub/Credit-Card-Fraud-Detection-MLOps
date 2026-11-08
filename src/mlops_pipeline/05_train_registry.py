import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from common import ROOT, load_config, write_json

def main():
    cfg = load_config()
    p = ROOT / cfg["data"]["processed_dir"]
    X = np.load(p/"X_train_final.npy")
    y = np.load(p/"y_train.npy")
    model = LogisticRegression(
        max_iter=cfg["model"]["max_iter"],
        class_weight=cfg["model"]["class_weight"],
        random_state=cfg["model"]["random_state"],
        solver="lbfgs"
    )
    model.fit(X, y)
    joblib.dump(model, ROOT/cfg["paths"]["model"])
    joblib.dump(model, ROOT/cfg["paths"]["full_pipeline"])
    meta = {
        "model_name": cfg["registry"]["model_name"],
        "model_type": "LogisticRegression",
        "status": "candidate",
        "features": int(X.shape[1]),
        "training_rows": int(len(y))
    }
    write_json("models/model_metadata.json", meta)
    # Optional MLflow tracking. Local model artifacts remain usable without MLflow.
    try:
        import mlflow
        import mlflow.sklearn
        tracking_uri = cfg["mlflow"].get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run() as run:
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("class_weight", "balanced")
            model_info = mlflow.sklearn.log_model(
                model,
                name="fraud_model",
                registered_model_name=cfg["registry"]["model_name"],
            )
            registered_version = getattr(
                model_info, "registered_model_version", None
            )
            if registered_version is None:
                raise RuntimeError("MLflow did not return a registered model version.")
            meta["mlflow_run_id"] = run.info.run_id
            meta["mlflow_model_version"] = str(registered_version)
            write_json("models/model_metadata.json", meta)
        print("[INFO] Model logged to MLflow.")
    except ImportError:
        print("[INFO] MLflow not installed; local model and registry metadata created.")
    print("[LAB 5 PASS] Logistic Regression trained and saved.")

if __name__ == "__main__":
    main()
