import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from common import ROOT, load_config, write_json

def main():
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["raw_path"])
    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    target = cfg["data"]["target_column"]
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()
    # Split before fitting the scaler to prevent test-data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"], stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    out = ROOT / cfg["data"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train_final.npy", X_train_scaled.astype("float32"))
    np.save(out / "X_test_final.npy", X_test_scaled.astype("float32"))
    np.save(out / "y_train.npy", y_train.to_numpy(dtype="int64"))
    np.save(out / "y_test.npy", y_test.to_numpy(dtype="int64"))
    X_train.to_csv(out / "X_train_raw.csv", index=False)
    X_test.to_csv(out / "X_test_raw.csv", index=False)
    joblib.dump(scaler, ROOT / "models/preprocessor.pkl")
    metadata = {
        "feature_columns": X.columns.tolist(),
        "original_rows": int(original_rows),
        "duplicate_rows_removed_before_split": duplicate_rows,
        "rows_after_deduplication": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "scaler": "StandardScaler fitted on training data only", "duplicate_policy": "Exact duplicate rows removed before stratified split"
    }
    write_json("data/processed/dataset_metadata.json", metadata)
    write_json("artifacts/preprocessing_summary_report.json", metadata)
    print(f"[LAB 3 PASS] Train={len(X_train):,}, test={len(X_test):,}; scaler fitted on train only.")

if __name__ == "__main__":
    main()
