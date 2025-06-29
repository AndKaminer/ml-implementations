from pydantic import BaseModel
from typing import List

class InferenceInput(BaseModel):
    model_type: str
    model_id: str = 0
    features: List[float]
