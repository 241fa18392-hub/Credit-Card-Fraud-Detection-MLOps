import joblib
import numpy as np
import pandas as pd
from common import ROOT, load_config

def main():
    cfg = load_config()
    registry = ROOT/"models/local_model_registry.json"
    if not registry.exists():
        raise FileNotFoundError("Production registry not found. Run Lab 8 first.")
    model = joblib.load(ROOT/cfg["paths"]["model"])
    raw = pd.read_csv(ROOT/cfg["data"]["raw_path"])
    target = cfg["data"]["target_column"]
    X = raw.drop(columns=[target]) if target in raw.columns else raw.copy()
    scaler = joblib.load(ROOT/cfg["paths"]["preprocessor"])
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:,1]
    out = raw.copy()
    out["Predicted_Class"] = predictions
    out["Fraud_Probability"] = np.round(probabilities,6)
    target_path = ROOT/cfg["paths"]["inference_output"]
    target_path.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(target_path,index=False)
    print(f"[LAB 9 PASS] Predictions saved: {target_path.relative_to(ROOT)}")
    print(out[["Predicted_Class","Fraud_Probability"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
