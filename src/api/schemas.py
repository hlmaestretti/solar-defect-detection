from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    inputs: List[float]


class PredictionResponse(BaseModel):
    prediction: float
