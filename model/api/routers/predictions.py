import os
from datetime import datetime
from typing import List, Optional
import time

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
import logging
import torchvision.transforms as transforms
from pydantic import BaseModel

from training.model import MNISTModel
from api.utils import preprocess_image
from api.models import PredictionResponse, FeedbackRequest, ModelStats
from api.db import db

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get model instance
def get_model():
    model = MNISTModel()
    model_path = os.getenv('MODEL_PATH', '/app/checkpoints/mnist_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@router.post("/predict", response_model=PredictionResponse)
async def predict_digit(
    file: UploadFile = File(...),
    model: MNISTModel = Depends(get_model)
):
    """
    Predict the digit in the uploaded image.
    
    Args:
        file: Image file containing a handwritten digit
        model: The loaded MNIST model
        
    Returns:
        PredictionResponse: Model's prediction and confidence
    """
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_digit = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_digit].item()
            all_probs = probabilities[0].tolist()  # Get probabilities for all digits
        
        # Log prediction to database
        prediction_id = db.store_prediction(contents, predicted_digit, confidence)
        
        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_digit=predicted_digit,
            confidence=confidence,
            probabilities=all_probs
        )
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def provide_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a prediction.
    
    Args:
        feedback: FeedbackRequest containing prediction_id and actual label
        
    Returns:
        dict: Success message
    """
    try:
        logger.info(f"Received feedback request: {feedback}")
        
        prediction, is_correct = db.update_prediction_feedback(
            feedback.prediction_id,
            feedback.actual_digit
        )
        
        return {
            "message": "Feedback recorded successfully",
            "prediction_id": str(feedback.prediction_id),
            "true_label": feedback.actual_digit,
            "original_prediction": prediction,
            "is_correct": is_correct
        }
        
    except ValueError as e:
        logger.error(f"Invalid prediction ID: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_model_stats():
    """
    Get model performance statistics.
    
    Returns:
        dict: Model performance metrics
    """
    try:
        return db.get_model_stats()
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class PredictionHistoryItem(BaseModel):
    timestamp: str
    prediction: int
    true_label: Optional[int] = None
    confidence: float

@router.get("/history", response_model=List[PredictionHistoryItem])
def get_prediction_history(limit: int = 10):
    try:
        rows = db.get_prediction_history(limit=limit)
        return [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": row["prediction"],
                "true_label": row["true_label"],
                "confidence": row["confidence"]
            }
            for row in rows
        ]
    except Exception as e:
        logging.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch prediction history") 