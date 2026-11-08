import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "src" / "mlops_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, PIPELINE_DIR / file_name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab12 = load_module("lab12_monitoring", "12_monitoring.py")
lab13 = load_module("lab13_pipeline", "13_end_to_end_pipeline.py")


def test_psi_is_zero_for_identical_populations():
    values = [0, 1, 2, 3, 4, 5]
    assert lab12.population_stability_index(values, values) == 0.0


def test_psi_detects_a_shifted_population():
    reference = [0] * 100 + [1] * 100
    current = [8] * 100 + [9] * 100
    assert lab12.population_stability_index(reference, current) > 0.2


def test_monitoring_decision_detects_feature_and_prediction_drift():
    settings = {
        "psi_threshold": 0.2,
        "performance_metric": "average_precision_pr_auc",
        "performance_drop_threshold": 0.05,
        "prediction_rate_delta_threshold": 0.05,
    }
    data_report = {"drifted_features": [{"feature": "V1", "psi": 0.25}]}
    prediction_report = {"probability_psi": 0.01, "positive_rate_delta": 0.0}
    performance_report = {"metric": "average_precision_pr_auc", "baseline": 0.7, "current": 0.69}

    decision = lab12.evaluate_retraining_trigger(
        data_report, prediction_report, performance_report, settings
    )

    assert decision["retraining_required"] is True
    assert "data_drift" in decision["reasons"]


def test_monitoring_decision_detects_performance_degradation():
    settings = {
        "psi_threshold": 0.2,
        "performance_metric": "average_precision_pr_auc",
        "performance_drop_threshold": 0.05,
        "prediction_rate_delta_threshold": 0.05,
    }
    data_report = {"drifted_features": []}
    prediction_report = {"probability_psi": 0.01, "positive_rate_delta": 0.0}
    performance_report = {"metric": "average_precision_pr_auc", "baseline": 0.7, "current": 0.6}

    decision = lab12.evaluate_retraining_trigger(
        data_report, prediction_report, performance_report, settings
    )

    assert decision["retraining_required"] is True
    assert "performance_degradation" in decision["reasons"]


def test_monitoring_decision_detects_prediction_rate_shift():
    settings = {
        "psi_threshold": 0.2,
        "performance_metric": "average_precision_pr_auc",
        "performance_drop_threshold": 0.05,
        "prediction_rate_delta_threshold": 0.05,
    }
    data_report = {"drifted_features": []}
    prediction_report = {"probability_psi": 0.01, "positive_rate_delta": 0.08}
    performance_report = {
        "metric": "average_precision_pr_auc",
        "baseline": 0.7,
        "current": 0.69,
    }

    decision = lab12.evaluate_retraining_trigger(
        data_report, prediction_report, performance_report, settings
    )

    assert "prediction_rate_shift" in decision["reasons"]


def test_monitoring_decision_does_not_retrain_without_signals():
    settings = {
        "psi_threshold": 0.2,
        "performance_metric": "average_precision_pr_auc",
        "performance_drop_threshold": 0.05,
        "prediction_rate_delta_threshold": 0.05,
    }
    data_report = {"drifted_features": []}
    prediction_report = {"probability_psi": 0.01, "positive_rate_delta": 0.01}
    performance_report = {"metric": "average_precision_pr_auc", "baseline": 0.7, "current": 0.69}

    decision = lab12.evaluate_retraining_trigger(
        data_report, prediction_report, performance_report, settings
    )

    assert decision["retraining_required"] is False
    assert decision["reasons"] == []


def test_lab13_includes_labs_1_through_12_and_has_entrypoint():
    lab_numbers = [stage["lab"] for stage in lab13.LAB_STAGES]

    assert lab_numbers == list(range(1, 13))
    assert lab13.LAB_STAGES[-1]["script"] == "12_monitoring.py"
    assert (PIPELINE_DIR / "13_end_to_end_pipeline.py").exists()
