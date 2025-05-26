from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
from datetime import datetime

class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    predicted_digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0, le=1)
    probabilities: List[float] = Field(..., description="Probabilities for each digit (0-9)")

class FeedbackRequest(BaseModel):
    prediction_id: uuid.UUID
    actual_digit: int = Field(..., ge=0, le=9)

class DigitStats(BaseModel):
    digit: int
    total: int
    correct: int
    accuracy: float

class OverallStats(BaseModel):
    accuracy: float
    total_predictions: int
    correct_predictions: int

class ModelStats(BaseModel):
    overall: OverallStats
    per_digit: List[DigitStats]

class Prediction(BaseModel):
    id: str
    timestamp: datetime
    image_data: bytes
    prediction: int
    confidence: float
    true_label: Optional[int] = None
    is_correct: Optional[bool] = None 