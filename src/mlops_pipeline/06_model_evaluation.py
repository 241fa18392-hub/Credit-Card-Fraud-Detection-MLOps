import joblib
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from common import ROOT, load_config, read_json, write_json

def main():
    cfg = load_config()
    model = joblib.load(ROOT/cfg["paths"]["model"])
    p = ROOT/cfg["data"]["processed_dir"]
    X = np.load(p/"X_test_final.npy")
    y = np.load(p/"y_test.npy")
    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    metrics = {
        "accuracy": float(accuracy_score(y,pred)),
        "precision": float(precision_score(y,pred,zero_division=0)),
        "recall": float(recall_score(y,pred,zero_division=0)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y,prob)),
        "roc_auc": float(roc_auc_score(y,prob)),
        "confusion_matrix": confusion_matrix(y,pred).tolist(),
        "test_rows": int(len(y))
    }
    write_json(cfg["paths"]["evaluation_report"], metrics)
    run_id = read_json("models/model_metadata.json").get("mlflow_run_id")
    if run_id:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg["mlflow"].get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow_client = MlflowClient()
        logged_metrics = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "average_precision": metrics["average_precision_pr_auc"],
            "average_precision_pr_auc": metrics["average_precision_pr_auc"],
            "roc_auc": metrics["roc_auc"],
        }
        for metric_name, metric_value in logged_metrics.items():
            mlflow_client.log_metric(run_id, metric_name, metric_value)
    (ROOT/"artifacts").mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y,pred)
    plt.tight_layout(); plt.savefig(ROOT/"artifacts/confusion_matrix.png"); plt.close()
    RocCurveDisplay.from_predictions(y,prob)
    plt.tight_layout(); plt.savefig(ROOT/"artifacts/roc_curve.png"); plt.close()
    PrecisionRecallDisplay.from_predictions(y,prob)
    plt.tight_layout(); plt.savefig(ROOT/"artifacts/precision_recall_curve.png"); plt.close()
    print("[LAB 6 PASS] Evaluation complete.")
    for k in ["precision","recall","f1","average_precision_pr_auc","roc_auc"]:
        print(f"{k}: {metrics[k]:.4f}")

if __name__ == "__main__":
    main()
