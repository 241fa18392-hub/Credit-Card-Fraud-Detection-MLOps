import json
from common import ROOT, load_config, read_json, write_json

def main():
    cfg = load_config()
    gate = read_json(cfg["paths"]["quality_gate_report"])
    if gate["status"] != "PASSED":
        raise RuntimeError("Quality gate did not pass; cannot promote candidate.")
    metadata_path = ROOT/"models/model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"status":"Production","production_metric":gate["metric"],"production_metric_value":gate["value"]})
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    registry = {
        "model_name": cfg["registry"]["model_name"],
        "versions": [{"version":"1","stage":"Production","model_path":cfg["paths"]["model"],"metric":gate["metric"],"metric_value":gate["value"]}]
    }
    mlflow_model_version = metadata.get("mlflow_model_version")
    if mlflow_model_version is not None:
        from mlflow.tracking import MlflowClient

        alias = "production"
        MlflowClient().set_registered_model_alias(
            cfg["registry"]["model_name"],
            alias,
            str(mlflow_model_version),
        )
        registry["mlflow_registry"] = {
            "model_name": cfg["registry"]["model_name"],
            "model_version": str(mlflow_model_version),
            "alias": alias,
        }
    write_json("models/local_model_registry.json", registry)
    write_json("reports/lifecycle_report.json", registry)
    print("[LAB 8 PASS] Candidate model promoted to local Production registry.")

if __name__ == "__main__":
    main()
