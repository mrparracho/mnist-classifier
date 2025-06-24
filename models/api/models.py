from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    predicted_digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0, le=1)
    probabilities: List[float] = Field(..., description="Probabilities for each digit (0-9)")
    model_name: str = Field(..., description="Name of the model used for prediction")

class FeedbackRequest(BaseModel):
    prediction_id: uuid.UUID
    actual_digit: int = Field(..., ge=0, le=9)
    model_name: Optional[str] = Field(default="cnn_mnist", description="Name of the model used for prediction")

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
    model_name: str
    true_label: Optional[int] = None
    is_correct: Optional[bool] = None

class ModelInfo(BaseModel):
    name: str
    display_name: str
    description: str
    model_type: str
    version: str
    is_active: bool
    checkpoint_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total: int

class ModelStatisticsResponse(BaseModel):
    model_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    per_digit_stats: Dict[str, Dict[str, Any]] 