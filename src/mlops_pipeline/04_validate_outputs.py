import numpy as np
import json
from common import ROOT, write_json

def main():
    p = ROOT / "data/processed"
    files = ["X_train_final.npy", "X_test_final.npy", "y_train.npy", "y_test.npy"]
    missing = [f for f in files if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"Processed files missing: {missing}")
    Xtr, Xte = np.load(p/files[0]), np.load(p/files[1])
    ytr, yte = np.load(p/files[2]), np.load(p/files[3])
    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional.")
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError("Train/test feature counts do not match.")
    if len(Xtr) != len(ytr) or len(Xte) != len(yte):
        raise ValueError("Feature and target row counts do not match.")
    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise ValueError("Processed features contain NaN or infinite values.")
    report = {"status":"PASSED","X_train_shape":list(Xtr.shape),"X_test_shape":list(Xte.shape)}
    write_json("artifacts/processed_output_validation.json", report)
    print("[LAB 4 PASS] Processed arrays have consistent shapes and finite values.")

if __name__ == "__main__":
    main()
