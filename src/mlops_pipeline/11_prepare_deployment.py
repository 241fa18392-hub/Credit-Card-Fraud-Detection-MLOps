import json, shutil
from pathlib import Path
import joblib
from common import ROOT, load_config

def main():
    cfg = load_config()
    registry_path = ROOT/"models/local_model_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("Run Lab 8 and promote a model before deployment.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    production = [v for v in registry["versions"] if v["stage"]=="Production"]
    if len(production) != 1:
        raise RuntimeError("Expected exactly one Production model version.")
    deployment = ROOT/"deployment"
    deployment.mkdir(exist_ok=True)
    shutil.copy2(ROOT/cfg["paths"]["model"], deployment/"model.pkl")
    shutil.copy2(ROOT/cfg["paths"]["preprocessor"], deployment/"preprocessor.pkl")
    metadata = {
        "model_name": registry["model_name"],
        "version": production[0]["version"],
        "stage": production[0]["stage"],
        "metric": production[0]["metric"],
        "metric_value": production[0]["metric_value"]
    }
    (deployment/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    # Verify serialized artifacts can be loaded.
    joblib.load(deployment/"model.pkl")
    joblib.load(deployment/"preprocessor.pkl")
    print("[LAB 11 PASS] Deployment bundle prepared and verified at deployment/.")

if __name__ == "__main__":
    main()
