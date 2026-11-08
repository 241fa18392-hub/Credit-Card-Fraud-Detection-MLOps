import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import ROOT, load_config, read_json, write_json


def population_stability_index(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float).reshape(-1)
    current = np.asarray(current, dtype=float).reshape(-1)
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI requires non-empty reference and current populations.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("PSI populations must contain only finite numeric values.")

    inner_edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])
    )
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)

    reference_proportions = (reference_counts + 0.5) / (
        reference_counts.sum() + 0.5 * len(reference_counts)
    )
    current_proportions = (current_counts + 0.5) / (
        current_counts.sum() + 0.5 * len(current_counts)
    )
    value = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )
    return float(value)


def evaluate_retraining_trigger(
    data_report, prediction_report, performance_report, settings
):
    psi_threshold = float(settings["psi_threshold"])
    performance_drop_threshold = float(
        settings["performance_drop_threshold"]
    )
    prediction_rate_delta_threshold = float(
        settings["prediction_rate_delta_threshold"]
    )
    reasons = []

    if data_report["drifted_features"]:
        reasons.append("data_drift")
    if prediction_report["probability_psi"] >= psi_threshold:
        reasons.append("prediction_drift")
    if (
        prediction_report["positive_rate_delta"]
        >= prediction_rate_delta_threshold
    ):
        reasons.append("prediction_rate_shift")

    baseline = performance_report.get("baseline")
    current = performance_report["current"]
    if (
        baseline is not None
        and baseline - current >= performance_drop_threshold
    ):
        reasons.append("performance_degradation")

    return {
        "retraining_required": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "psi_threshold": psi_threshold,
            "performance_drop_threshold": performance_drop_threshold,
            "prediction_rate_delta_threshold": prediction_rate_delta_threshold,
        },
    }


def _performance_metrics(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision_pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(predictions)),
    }


def _read_or_initialize_baseline(cfg, model_version):
    baseline_path = "reports/monitoring_baseline.json"
    target = ROOT / baseline_path
    if target.exists():
        return read_json(baseline_path)

    evaluation = read_json(cfg["paths"]["evaluation_report"])
    baseline = {
        "source": cfg["paths"]["evaluation_report"],
        "model_version": model_version,
        "metrics": evaluation,
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(baseline_path, baseline)
    return baseline


def run_monitoring(force_retrain=False, allow_retrain=True):
    cfg = load_config()
    processed = ROOT / cfg["data"]["processed_dir"]
    train_raw_path = processed / "X_train_raw.csv"
    test_raw_path = processed / "X_test_raw.csv"
    required_paths = [
        ROOT / cfg["paths"]["model"],
        ROOT / cfg["paths"]["preprocessor"],
        processed / "X_train_final.npy",
        processed / "X_test_final.npy",
        processed / "y_test.npy",
        train_raw_path,
        test_raw_path,
        ROOT / cfg["paths"]["evaluation_report"],
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Lab 12 prerequisites are missing; run Labs 1–11 first: "
            + ", ".join(missing)
        )

    settings = cfg["lab12"]
    psi_threshold = float(settings["psi_threshold"])
    model = joblib.load(ROOT / cfg["paths"]["model"])
    X_train = np.load(processed / "X_train_final.npy")
    X_test = np.load(processed / "X_test_final.npy")
    y_test = np.load(processed / "y_test.npy")
    y_train = np.load(processed / "y_train.npy")
    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)

    if list(train_raw.columns) != list(test_raw.columns):
        raise ValueError("Training and current monitoring feature columns differ.")
    if X_train.shape[0] != len(train_raw) or X_test.shape[0] != len(test_raw):
        raise ValueError("Raw and scaled monitoring populations have different row counts.")
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError("Processed feature and target row counts do not match.")

    model_metadata = read_json("models/model_metadata.json")
    model_version = model_metadata.get("mlflow_model_version")
    baseline_record = _read_or_initialize_baseline(cfg, model_version)
    baseline_metrics = baseline_record.get("metrics", {})

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()

    feature_rows = []
    for column in train_raw.columns:
        psi = population_stability_index(
            train_raw[column].to_numpy(),
            test_raw[column].to_numpy(),
            bins=int(settings.get("psi_bins", 10)),
        )
        feature_rows.append(
            {"feature": column, "psi": psi, "drift_detected": psi >= psi_threshold}
        )
    drifted_features = [
        row for row in feature_rows if row["drift_detected"]
    ]
    data_report = {
        "status": "PASSED",
        "reference_source": "data/processed/X_train_raw.csv",
        "current_source": "data/processed/X_test_raw.csv",
        "reference_rows": int(len(train_raw)),
        "current_rows": int(len(test_raw)),
        "psi_threshold": psi_threshold,
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "features": feature_rows,
    }

    current_metrics = _performance_metrics(model, X_test, y_test)
    performance_metric = settings["performance_metric"]
    baseline_metric = baseline_metrics.get(performance_metric)
    performance_report = {
        "status": "PASSED",
        "source": "data/processed/X_test_final.npy",
        "baseline_source": baseline_record["source"],
        "baseline_model_version": baseline_record.get("model_version"),
        "metric": performance_metric,
        "baseline": (
            float(baseline_metric) if baseline_metric is not None else None
        ),
        "current": float(current_metrics[performance_metric]),
        "drop_from_baseline": (
            float(baseline_metric - current_metrics[performance_metric])
            if baseline_metric is not None
            else None
        ),
        "metrics": current_metrics,
    }

    reference_probabilities = model.predict_proba(X_train)[:, 1]
    current_probabilities = model.predict_proba(X_test)[:, 1]
    reference_predictions = model.predict(X_train)
    current_predictions = model.predict(X_test)
    prediction_report = {
        "status": "PASSED",
        "reference_source": "data/processed/X_train_final.npy",
        "current_source": "data/processed/X_test_final.npy",
        "reference_rows": int(len(X_train)),
        "current_rows": int(len(X_test)),
        "reference_positive_rate": float(np.mean(reference_predictions)),
        "current_positive_rate": float(np.mean(current_predictions)),
        "positive_rate_delta": float(
            abs(np.mean(current_predictions) - np.mean(reference_predictions))
        ),
        "probability_psi": population_stability_index(
            reference_probabilities,
            current_probabilities,
            bins=int(settings.get("psi_bins", 10)),
        ),
        "reference_probability_mean": float(np.mean(reference_probabilities)),
        "current_probability_mean": float(np.mean(current_probabilities)),
    }
    decision = evaluate_retraining_trigger(
        data_report, prediction_report, performance_report, settings
    )
    if force_retrain:
        decision["retraining_required"] = True
        decision["reasons"] = sorted(set(decision["reasons"] + ["forced"]))

    write_json("reports/data_drift_report.json", data_report)
    write_json("reports/prediction_drift_report.json", prediction_report)
    write_json("reports/performance_monitoring_report.json", performance_report)

    decision_report = {
        **decision,
        "status": "PASSED",
        "retraining_enabled": bool(settings.get("retraining_enabled", True)),
        "retraining_status": "NOT_REQUIRED",
        "started_at": started_at,
    }
    retraining_enabled = (
        allow_retrain and bool(settings.get("retraining_enabled", True))
    )
    if decision["retraining_required"] and retraining_enabled:
        retraining_started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src/mlops_pipeline/fraud_detection_pipeline.py"),
            ],
            cwd=ROOT,
            check=False,
        )
        decision_report["retraining_status"] = (
            "PASSED" if result.returncode == 0 else "FAILED"
        )
        decision_report["retraining_exit_code"] = int(result.returncode)
        decision_report["retraining_duration_seconds"] = round(
            time.monotonic() - retraining_started, 3
        )
        if result.returncode != 0:
            decision_report["status"] = "FAILED"
            write_json("reports/retraining_decision_report.json", decision_report)
            raise RuntimeError(
                "Retraining pipeline failed with exit code "
                f"{result.returncode}."
            )
        updated_baseline = {
            "source": cfg["paths"]["evaluation_report"],
            "model_version": read_json("models/model_metadata.json").get(
                "mlflow_model_version"
            ),
            "metrics": read_json(cfg["paths"]["evaluation_report"]),
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "updated_after_retraining": True,
        }
        write_json("reports/monitoring_baseline.json", updated_baseline)
    elif decision["retraining_required"]:
        decision_report["retraining_status"] = "DISABLED"

    decision_report["duration_seconds"] = round(time.monotonic() - started, 3)
    decision_report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json("reports/retraining_decision_report.json", decision_report)

    print(
        "[LAB 12 PASS] Performance, data drift, and prediction drift reports "
        "written."
    )
    print(
        f"[INFO] Average precision={performance_report['current']:.4f}; "
        f"drifted features={len(drifted_features)}; "
        f"probability PSI={prediction_report['probability_psi']:.4f}."
    )
    print(
        "[INFO] Retraining "
        + (
            f"completed with status {decision_report['retraining_status']}."
            if decision["retraining_required"] and retraining_enabled
            else (
                "is required but disabled by configuration/option."
                if decision["retraining_required"]
                else "not required."
            )
        )
    )
    return decision_report


def main():
    parser = argparse.ArgumentParser(description="Monitor fraud model and drift.")
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Report retraining triggers without starting Labs 1–11.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Run the Labs 1–11 pipeline regardless of measured triggers.",
    )
    args = parser.parse_args()
    run_monitoring(
        force_retrain=args.force_retrain,
        allow_retrain=not args.no_retrain,
    )


if __name__ == "__main__":
    main()
