import pandas as pd
from common import ROOT, load_config, write_json

def main():
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["raw_path"])
    target = cfg["data"]["target_column"]
    required = {"Time", "Amount", target}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    if df[target].isna().any():
        raise ValueError("Target column contains missing values.")
    labels = set(pd.Series(df[target]).dropna().unique().tolist())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Expected binary Class labels 0 and 1; found {labels}")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError("Target column must be numeric.")
    feature_df = df.drop(columns=[target])
    non_numeric = feature_df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric}")
    report = {
        "status": "PASSED",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": target,
        "class_distribution": {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }
    write_json("artifacts/data_validation_report.json", report)
    print("[LAB 2 PASS] Schema, numeric features, and binary target validated.")

if __name__ == "__main__":
    main()
