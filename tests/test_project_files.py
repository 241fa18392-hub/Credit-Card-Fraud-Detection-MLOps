from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_exists():
    assert (ROOT / "data/raw/creditcard.csv").exists()

def test_all_lab_scripts_exist():
    for i in range(1, 14):
        assert (ROOT / f"src/mlops_pipeline/{i:02d}_" ).exists() or any(
            (ROOT / "src/mlops_pipeline").glob(f"{i:02d}_*.py")
        )
