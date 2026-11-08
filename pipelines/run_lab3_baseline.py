import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call(
    [sys.executable, str(ROOT/"src/mlops_pipeline/03_preprocessing.py")], cwd=ROOT
))
