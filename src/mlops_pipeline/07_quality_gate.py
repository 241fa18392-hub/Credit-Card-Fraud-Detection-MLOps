from common import ROOT, load_config, read_json, write_json

def main():
    cfg = load_config()
    metrics = read_json(cfg["paths"]["evaluation_report"])
    metric = cfg["quality_gate"]["metric"]
    value = float(metrics[metric if metric != "average_precision" else "average_precision_pr_auc"])
    threshold = float(cfg["quality_gate"]["minimum_value"])
    passed = value >= threshold
    report = {"metric":metric,"value":value,"minimum_value":threshold,"status":"PASSED" if passed else "FAILED"}
    write_json(cfg["paths"]["quality_gate_report"], report)
    print(f"[LAB 7 {'PASS' if passed else 'FAIL'}] {metric}={value:.4f}; threshold={threshold:.4f}")
    if not passed:
        raise SystemExit("Quality gate failed; model will not be promoted.")

if __name__ == "__main__":
    main()
