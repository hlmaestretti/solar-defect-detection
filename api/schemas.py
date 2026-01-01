"""
Pydantic schemas for the EL defect detection API.

Responsibilities:
- Define response models for inference endpoints.
- Provide a clear, explicit contract for API consumers.
"""


from pydantic import BaseModel
from typing import Dict


class PredictResponse(BaseModel):
    probabilities: Dict[str, float]
    predictions: Dict[str, bool]
