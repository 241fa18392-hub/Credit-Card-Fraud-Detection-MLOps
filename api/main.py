from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from .schemas import PredictionRequest, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

def load_bundle():
    model_path = ROOT / "deployment/model.pkl"
    scaler_path = ROOT / "deployment/preprocessor.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Deployment bundle missing. Run Lab 11 first.")
    return joblib.load(model_path), joblib.load(scaler_path)

@app.get("/health")
def health():
    return {"status":"ok","project":"credit-card-fraud-detection"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model, scaler = load_bundle()
        X = np.asarray(request.records, dtype=float)
        if X.ndim != 2 or X.shape[1] != 30:
            raise ValueError("Each record must contain exactly 30 numeric features.")
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled).astype(int).tolist()
        prob = model.predict_proba(X_scaled)[:,1].astype(float).tolist()
        return PredictionResponse(predictions=pred, fraud_probabilities=prob)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
