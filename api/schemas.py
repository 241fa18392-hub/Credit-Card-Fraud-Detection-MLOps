from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    records: List[List[float]] = Field(..., description="Rows of 30 numeric transaction features")

class PredictionResponse(BaseModel):
    predictions: List[int]
    fraud_probabilities: List[float]
