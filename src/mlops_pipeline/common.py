from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def path(key):
    return ROOT / key

def ensure_dirs():
    for name in ["artifacts", "data/processed", "logs", "models", "outputs", "reports"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)

def write_json(relative_path, data):
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target

def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
