import os
from datetime import datetime
from typing import List, Optional
import time

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
import logging
import torchvision.transforms as transforms
from pydantic import BaseModel

from api.utils import preprocess_image
from api.models import PredictionResponse, FeedbackRequest, ModelStats
from api.db import Database

# Create database instance with mnist_classifier schema for original app
db = Database(schema="mnist_classifier")

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the model loading function
import sys
from pathlib import Path
sys.path.insert(0, '/app')
from model_factory import get_model
from config import get_active_model_names

# Model cache
_model_instances = {}

def get_model_instance(model_name: str = "cnn_mnist"):
    """Get or load a model instance."""
    if model_name not in _model_instances:
        try:
            # Load model using the centralized loader
            model = get_model(model_name)
            _model_instances[model_name] = model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {model_name}: {str(e)}"
            )
    
    return _model_instances[model_name]

@router.post("/predict", response_model=PredictionResponse)
async def predict_digit(
    file: UploadFile = File(...),
    model_name: str = Query(default="cnn_mnist", description="Name of the model to use")
):
    """
    Predict the digit in the uploaded image.
    
    Args:
        file: Image file containing a handwritten digit
        model_name: The model to use for prediction
        
    Returns:
        PredictionResponse: Model's prediction and confidence
    """
    try:
        # Get model instance
        model = get_model_instance(model_name)
        
        # Read and preprocess the image
        contents = await file.read()
        
        # Make prediction using the model's predict method
        predicted_digit, confidence, all_probs = model.predict(contents)
        
        # Log prediction to database
        prediction_id = db.store_prediction(contents, predicted_digit, confidence, model_name)
        
        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_digit=predicted_digit,
            confidence=confidence,
            probabilities=all_probs,
            model_name=model_name
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
        
        # Update the specific prediction and get session info
        prediction, is_correct, session_id = db.update_prediction_feedback(
            feedback.prediction_id,
            feedback.actual_digit
        )
        
        # If this prediction has a session_id, update all other predictions in the same session
        updated_count = 1
        if session_id:
            updated_count = db.update_session_feedback(session_id, feedback.actual_digit, feedback.prediction_id)
        
        return {
            "message": "Feedback recorded successfully",
            "prediction_id": str(feedback.prediction_id),
            "true_label": feedback.actual_digit,
            "original_prediction": prediction,
            "is_correct": is_correct,
            "updated_predictions": updated_count
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
    model_name: str

@router.get("/history", response_model=List[PredictionHistoryItem])
def get_prediction_history(
    limit: int = 10,
    model_name: Optional[str] = Query(default=None, description="Filter by model name")
):
    try:
        rows = db.get_prediction_history(limit=limit)
        return [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": row["prediction"],
                "true_label": row["true_label"],
                "confidence": row["confidence"],
                "model_name": row.get("model_name", "cnn_mnist")  # Default to cnn_mnist for old records
            }
            for row in rows
        ]
    except Exception as e:
        logging.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch prediction history")

class MultiModelPredictionResponse(BaseModel):
    """Response model for predictions from multiple models."""
    predictions: List[PredictionResponse]
    image_processed: bool = True

@router.post("/predict-all", response_model=MultiModelPredictionResponse)
async def predict_digit_all_models(
    file: UploadFile = File(...)
):
    """
    Predict the digit using all available models for comparison.
    
    Args:
        file: Image file containing a handwritten digit
        
    Returns:
        MultiModelPredictionResponse: Predictions from all available models
    """
    try:
        # Read image data once
        contents = await file.read()
        
        # Generate a session ID to link all predictions from this request
        import uuid
        session_id = str(uuid.uuid4())
        
        # Get all active models
        active_models = get_active_model_names()
        predictions = []
        
        for model_name in active_models:
            try:
                # Get model instance
                model = get_model_instance(model_name)
                
                # Make prediction using the model's predict method
                predicted_digit, confidence, all_probs = model.predict(contents)
                
                # Log prediction to database with session_id
                prediction_id = db.store_prediction(contents, predicted_digit, confidence, model_name, session_id)
                
                predictions.append(PredictionResponse(
                    prediction_id=prediction_id,
                    predicted_digit=predicted_digit,
                    confidence=confidence,
                    probabilities=all_probs,
                    model_name=model_name
                ))
                
            except Exception as e:
                logger.error(f"Error with model {model_name}: {str(e)}")
                # Continue with other models even if one fails
                continue
        
        if not predictions:
            raise HTTPException(status_code=500, detail="No models could process the image")
        
        return MultiModelPredictionResponse(
            predictions=predictions,
            image_processed=True
        )
        
    except Exception as e:
        logger.error(f"Error processing multi-model prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 