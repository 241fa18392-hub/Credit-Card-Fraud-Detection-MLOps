"""Convenience entry point; canonical implementation is in mlops_pipeline/02_validate_data.py."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call(
    [sys.executable, str(ROOT / "src/mlops_pipeline/02_validate_data.py")], cwd=ROOT
))
