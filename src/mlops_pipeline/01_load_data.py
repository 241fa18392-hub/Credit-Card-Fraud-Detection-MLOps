import pandas as pd
from common import ROOT, load_config, ensure_dirs, write_json

def main():
    ensure_dirs()
    cfg = load_config()
    source = ROOT / cfg["data"]["raw_path"]
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    df = pd.read_csv(source)
    report = {
        "status": "loaded",
        "source": str(source.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": cfg["data"]["target_column"]
    }
    write_json("artifacts/data_load_report.json", report)
    print(f"[LAB 1 PASS] Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print("[INFO] Profile saved to artifacts/data_load_report.json")

if __name__ == "__main__":
    main()
