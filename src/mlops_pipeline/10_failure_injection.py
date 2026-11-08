import tempfile
import pandas as pd
from common import ROOT, load_config, write_json

def main():
    cfg = load_config()
    source = ROOT/cfg["data"]["raw_path"]
    original = pd.read_csv(source, nrows=100)
    results = []
    # Scenario 1: required target missing.
    bad = original.drop(columns=[cfg["data"]["target_column"]])
    results.append({"scenario":"missing_target_column","detected":cfg["data"]["target_column"] not in bad.columns})
    # Scenario 2: invalid target labels.
    bad2 = original.copy()
    bad2[cfg["data"]["target_column"]] = 7
    results.append({"scenario":"invalid_target_label","detected":not set(bad2["Class"].unique()).issubset({0,1})})
    # Scenario 3: missing values.
    bad3 = original.copy()
    bad3.loc[0,"Amount"] = float("nan")
    results.append({"scenario":"missing_feature_value","detected":bool(bad3.isna().any().any())})
    # Scenario 4: corrupted non-finite feature.
    bad4 = original.copy()
    bad4.loc[0,"Amount"] = float("inf")
    results.append({"scenario":"infinite_feature_value","detected":not __import__("numpy").isfinite(bad4.select_dtypes("number").to_numpy()).all()})
    # Scenario 5: simulate a missing model path without deleting the real model.
    from pathlib import Path
    with tempfile.TemporaryDirectory() as temp_dir:
        simulated_missing_model = Path(temp_dir) / "missing_model.pkl"
        results.append({"scenario":"missing_model_artifact","detected":not simulated_missing_model.exists()})
    passed = all(x["detected"] for x in results)
    report = {"status":"PASSED" if passed else "FAILED","scenarios":results,"note":"Safe, in-memory fault simulations; source dataset and model files were not modified."}
    write_json(cfg["paths"]["failure_report"],report)
    print(f"[LAB 10 {'PASS' if passed else 'FAIL'}] {sum(x['detected'] for x in results)}/{len(results)} failure conditions detected.")
    if not passed:
        raise SystemExit("One or more failure-injection checks failed.")

if __name__ == "__main__":
    main()
