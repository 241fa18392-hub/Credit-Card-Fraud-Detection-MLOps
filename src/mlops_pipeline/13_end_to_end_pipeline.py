import subprocess
import sys
import time
from datetime import datetime, timezone

from common import ROOT, write_json


LAB_STAGES = [
    {"lab": 1, "name": "Data loading", "script": "01_load_data.py"},
    {"lab": 2, "name": "Data validation", "script": "02_validate_data.py"},
    {"lab": 3, "name": "Preprocessing", "script": "03_preprocessing.py"},
    {"lab": 4, "name": "Processed output validation", "script": "04_validate_outputs.py"},
    {"lab": 5, "name": "Model training and MLflow registry", "script": "05_train_registry.py"},
    {"lab": 6, "name": "Model evaluation", "script": "06_model_evaluation.py"},
    {"lab": 7, "name": "Quality gate", "script": "07_quality_gate.py"},
    {"lab": 8, "name": "Model lifecycle", "script": "08_automate_lifecycle.py"},
    {"lab": 9, "name": "Production inference", "script": "09_inference.py"},
    {"lab": 10, "name": "Failure injection", "script": "10_failure_injection.py"},
    {"lab": 11, "name": "Deployment preparation", "script": "11_prepare_deployment.py"},
    {"lab": 12, "name": "Performance and drift monitoring", "script": "12_monitoring.py"},
]


def run_end_to_end():
    started_at = datetime.now(timezone.utc).isoformat()
    stages = []

    for definition in LAB_STAGES:
        script_path = ROOT / "src/mlops_pipeline" / definition["script"]
        if not script_path.is_file():
            raise FileNotFoundError(f"Lab {definition['lab']} script missing: {script_path}")

        print(
            f"\n{'=' * 68}\n"
            f"RUNNING LAB {definition['lab']}: {definition['name']}\n"
            f"{'=' * 68}",
            flush=True,
        )
        stage_started = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            check=False,
        )
        stages.append(
            {
                **definition,
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "exit_code": int(result.returncode),
                "duration_seconds": round(time.monotonic() - stage_started, 3),
            }
        )
        if result.returncode != 0:
            report = {
                "project": "Credit Card Fraud Detection",
                "status": "FAILED",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "stages": stages,
                "lab13": {"status": "FAILED"},
            }
            write_json("logs/lab13_execution_report.json", report)
            raise SystemExit(result.returncode)

    report = {
        "project": "Credit Card Fraud Detection",
        "status": "PASSED",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "lab13": {
            "status": "PASSED",
            "name": "End-to-end integration and automation",
            "stage_count": len(stages),
        },
    }
    write_json("logs/lab13_execution_report.json", report)
    print("\n[LAB 13 PASS] Labs 1-12 completed successfully.")
    print("[INFO] End-to-end report: logs/lab13_execution_report.json")
    return report


def main():
    run_end_to_end()


if __name__ == "__main__":
    main()
