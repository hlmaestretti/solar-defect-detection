from pydantic import BaseModel
from typing import Dict


class PredictResponse(BaseModel):
    probabilities: Dict[str, float]
    predictions: Dict[str, bool]
