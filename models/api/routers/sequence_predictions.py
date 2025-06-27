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

# Create database instance with mnist_sequence schema for sequence prediction app
db = Database(schema="mnist_sequence")

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

# Model cache for grid-specific models
_model_instances = {}

def get_encoder_decoder_model(grid_size: int):
    """Get or load the encoder-decoder model instance for a specific grid size."""
    cache_key = f"encoder_decoder_grid_{grid_size}"
    
    if cache_key not in _model_instances:
        try:
            # Load model using the grid-specific loader
            from model_factory import get_encoder_decoder_model as get_grid_model
            model = get_grid_model(grid_size)
            _model_instances[cache_key] = model
        except Exception as e:
            logger.error(f"Error loading encoder-decoder model for grid {grid_size}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load encoder-decoder model for grid {grid_size}: {str(e)}"
            )
    
    return _model_instances[cache_key]

class SequencePredictionResponse(BaseModel):
    """Response model for sequence predictions."""
    prediction_id: str
    predicted_sequence: List[int]
    confidence: float
    grid_size: int
    model_name: str = "encoder_decoder"

class SequenceFeedbackRequest(BaseModel):
    """Request model for sequence feedback."""
    prediction_id: str
    true_sequence: List[int]
    grid_size: int

@router.post("/predict-sequence", response_model=SequencePredictionResponse)
async def predict_sequence(
    file: UploadFile = File(...),
    grid_size: int = Query(..., ge=1, le=4, description="Grid size (1-4)")
):
    """
    Predict a sequence of digits using the encoder-decoder model.
    
    Args:
        file: Image file containing a handwritten digit
        grid_size: Size of the grid (1-4)
        
    Returns:
        SequencePredictionResponse: Model's prediction and confidence
    """
    try:
        # Get model instance
        model = get_encoder_decoder_model(grid_size)
        
        # Read and preprocess the image
        contents = await file.read()
        
        # Make prediction using the model's predict method
        # The encoder-decoder model should return a sequence of digits
        predicted_sequence, confidence = model.predict_sequence(contents, grid_size)
        
        # Log prediction to database
        prediction_id = db.store_sequence_prediction(
            contents, predicted_sequence, confidence, grid_size
        )
        
        return SequencePredictionResponse(
            prediction_id=prediction_id,
            predicted_sequence=predicted_sequence,
            confidence=confidence,
            grid_size=grid_size,
            model_name=f"encoder_decoder_grid_{grid_size}"
        )
        
    except Exception as e:
        logger.error(f"Error processing sequence prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback-sequence")
async def provide_sequence_feedback(feedback: SequenceFeedbackRequest):
    """
    Submit feedback for a sequence prediction.
    
    Args:
        feedback: SequenceFeedbackRequest containing prediction_id and true sequence
        
    Returns:
        dict: Success message
    """
    try:
        logger.info(f"Received sequence feedback request: {feedback}")
        
        # Update the specific prediction and get session info
        prediction, is_correct, session_id = db.update_sequence_prediction_feedback(
            feedback.prediction_id,
            feedback.true_sequence,
            feedback.grid_size
        )
        
        return {
            "message": "Sequence feedback recorded successfully",
            "prediction_id": str(feedback.prediction_id),
            "true_sequence": feedback.true_sequence,
            "original_prediction": prediction,
            "is_correct": is_correct
        }
        
    except ValueError as e:
        logger.error(f"Invalid prediction ID: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error recording sequence feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats-sequence")
async def get_sequence_model_stats():
    """
    Get encoder-decoder model performance statistics.
    
    Returns:
        dict: Model performance metrics
    """
    try:
        return db.get_sequence_model_stats()
    except Exception as e:
        logger.error(f"Error getting sequence model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SequencePredictionHistoryItem(BaseModel):
    timestamp: str
    predicted_sequence: List[int]
    true_sequence: Optional[List[int]] = None
    confidence: float
    grid_size: int
    model_name: str = "encoder_decoder"

@router.get("/history-sequence", response_model=List[SequencePredictionHistoryItem])
def get_sequence_prediction_history(
    limit: int = 10
):
    try:
        rows = db.get_sequence_prediction_history(limit=limit)
        return [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_sequence": row["predicted_sequence"],
                "true_sequence": row["true_sequence"],
                "confidence": row["confidence"],
                "grid_size": row["grid_size"],
                "model_name": f"encoder_decoder_grid_{row['grid_size']}"
            }
            for row in rows
        ]
    except Exception as e:
        logging.error(f"Error fetching sequence prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sequence prediction history") 