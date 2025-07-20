from pydantic import BaseModel
from typing import List, Dict, Any

class InferenceInput(BaseModel):
    model_type: str
    model_id: int = 0
    features: List[float]

class RegisterInput(BaseModel):
    model_type: str
    model_id: int
    kwargs: Dict[str, Any] # kwargs to pass to model

class ListInput(BaseModel):
    model_type: str
