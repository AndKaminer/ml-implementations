from pydantic import BaseModel
from typing import List

class InferenceInput(BaseModel):
    model_type: str
    features: List[float]
