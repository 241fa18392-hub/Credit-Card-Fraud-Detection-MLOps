import subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    "01_load_data.py","02_validate_data.py","03_preprocessing.py",
    "04_validate_outputs.py","05_train_registry.py","06_model_evaluation.py",
    "07_quality_gate.py","08_automate_lifecycle.py","09_inference.py",
    "10_failure_injection.py","11_prepare_deployment.py"
]

def main():
    log = {"project":"Credit Card Fraud Detection","labs":[]}
    for script in SCRIPTS:
        print("\n"+"="*68+"\nRUNNING "+script+"\n"+"="*68,flush=True)
        result = subprocess.run([sys.executable,str(ROOT/"src/mlops_pipeline"/script)],cwd=ROOT)
        log["labs"].append({"script":script,"status":"PASSED" if result.returncode==0 else "FAILED","exit_code":result.returncode})
        if result.returncode:
            log["overall_status"]="FAILED"
            break
    else:
        log["overall_status"]="PASSED"
    (ROOT/"logs").mkdir(exist_ok=True)
    (ROOT/"logs/pipeline_run_log.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    print("\nPIPELINE STATUS:",log["overall_status"])
    return 0 if log["overall_status"]=="PASSED" else 1

if __name__=="__main__":
    raise SystemExit(main())
