import io
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import DictCursor
import torchvision.transforms as transforms
import sys
from pathlib import Path

# Since we're in the models directory (/app), import from current directory
sys.path.insert(0, '/app')

# Import from the current package (models)
from model_factory import get_model, get_available_models, get_model_metadata
from api.routers import predictions
from api.routers import models
from api.utils import preprocess_image
from api.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MNIST Classifier API",
    description="API for MNIST digit classification using multiple PyTorch models",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    await init_db()
    logger.info("Database connection initialized")

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])

# Define Pydantic models for request/response
class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]
    prediction_id: int
    model_name: str


class FeedbackRequest(BaseModel):
    prediction_id: int
    true_label: int
    model_name: str = "cnn_mnist"


# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_NAME = os.getenv("DB_NAME", "mnist")
DB_PORT = os.getenv("DB_PORT", "5432")


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Database connection error"
        )


# Model management
_model_instances: Dict[str, Any] = {}


def get_model_instance(model_name: str = "cnn_mnist"):
    """Get or create a model instance."""
    if model_name not in _model_instances:
        try:
            # Get checkpoint path from environment or use default
            checkpoint_path = os.getenv(f"{model_name.upper()}_CHECKPOINT_PATH")
            model = get_model(model_name, checkpoint_path)
            _model_instances[model_name] = model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {model_name}: {str(e)}"
            )
    
    return _model_instances[model_name]


# Image preprocessing
def preprocess_image(image_bytes, model_name: str = "cnn_mnist"):
    """
    Preprocess the image for model inference.
    
    Args:
        image_bytes: Raw bytes of the uploaded image
        model_name: Name of the model to use for preprocessing
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Get model instance and use its preprocessing
        model = get_model_instance(model_name)
        return model.preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image format: {e}"
        )


# Database logging function
def log_prediction(
    image_data, 
    prediction, 
    confidence, 
    model_name: str = "cnn_mnist",
    true_label=None
):
    """
    Log prediction details to PostgreSQL database.
    
    Args:
        image_data: Binary image data
        prediction: Predicted digit
        confidence: Prediction confidence
        model_name: Name of the model used
        true_label: User-provided true label (if available)
    
    Returns:
        int: ID of the inserted record
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Insert prediction record
            query = """
            INSERT INTO predictions 
            (timestamp, image_data, prediction, confidence, model_name, true_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            
            cursor.execute(
                query, 
                (
                    datetime.now(),
                    psycopg2.Binary(image_data),
                    prediction,
                    confidence,
                    model_name,
                    true_label
                )
            )
            
            # Get the ID of the inserted record
            prediction_id = cursor.fetchone()[0]
            
            conn.commit()
            return prediction_id
    except Exception as e:
        logger.error(f"Database logging error: {e}")
        return None
    finally:
        conn.close()


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {"message": "MNIST Classifier API", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Query(default="cnn_mnist", description="Name of the model to use")
):
    """
    Make a prediction on an uploaded image using the specified model.
    
    Args:
        file: The uploaded image file
        model_name: The model to use for prediction
        background_tasks: Background tasks for async operations
    
    Returns:
        PredictionResponse: Prediction result with confidence and probabilities
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get model instance
        model = get_model_instance(model_name)
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes, model_name)
        
        # Make prediction
        with torch.no_grad():
            output = model(processed_image.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            prediction = predicted.item()
            confidence_score = confidence.item()
            all_probabilities = probabilities.squeeze().tolist()
        
        # Log prediction in background
        prediction_id = log_prediction(
            image_bytes, 
            prediction, 
            confidence_score, 
            model_name
        )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence_score,
            probabilities=all_probabilities,
            prediction_id=prediction_id or 0,
            model_name=model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/feedback")
async def feedback(request: FeedbackRequest):
    """
    Submit feedback for a prediction.
    
    Args:
        request: Feedback request containing prediction ID and true label
    
    Returns:
        dict: Success message
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Update the prediction with the true label
            query = """
            UPDATE predictions 
            SET true_label = %s, 
                is_correct = (prediction = %s),
                feedback_submitted = true
            WHERE id = %s AND model_name = %s;
            """
            
            cursor.execute(
                query, 
                (
                    request.true_label,
                    request.true_label,
                    request.prediction_id,
                    request.model_name
                )
            )
            
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404, 
                    detail="Prediction not found"
                )
            
            conn.commit()
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit feedback: {str(e)}"
        )
    finally:
        conn.close()


@app.get("/api/v1/stats")
async def get_stats(model_name: str = Query(default=None, description="Filter by model name")):
    """
    Get prediction statistics.
    
    Args:
        model_name: Optional model name to filter statistics
    
    Returns:
        dict: Statistics including accuracy and per-digit performance
    """
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            if model_name:
                # Get statistics for specific model
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN is_correct THEN 1 END) as correct_predictions,
                        ROUND(
                            COUNT(CASE WHEN is_correct THEN 1 END)::float / 
                            NULLIF(COUNT(*), 0), 4
                        ) as accuracy
                    FROM predictions
                    WHERE true_label IS NOT NULL AND model_name = %s;
                """, (model_name,))
                
                overall = cursor.fetchone()
                
                # Get per-digit statistics
                cursor.execute("""
                    SELECT 
                        prediction as digit,
                        COUNT(*) as total,
                        COUNT(CASE WHEN is_correct THEN 1 END) as correct,
                        ROUND(
                            COUNT(CASE WHEN is_correct THEN 1 END)::float / 
                            NULLIF(COUNT(*), 0), 4
                        ) as accuracy
                    FROM predictions
                    WHERE true_label IS NOT NULL AND model_name = %s
                    GROUP BY prediction
                    ORDER BY prediction;
                """, (model_name,))
                
            else:
                # Get overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN is_correct THEN 1 END) as correct_predictions,
                        ROUND(
                            COUNT(CASE WHEN is_correct THEN 1 END)::float / 
                            NULLIF(COUNT(*), 0), 4
                        ) as accuracy
                    FROM predictions
                    WHERE true_label IS NOT NULL;
                """)
                
                overall = cursor.fetchone()
                
                # Get per-digit statistics
                cursor.execute("""
                    SELECT 
                        prediction as digit,
                        COUNT(*) as total,
                        COUNT(CASE WHEN is_correct THEN 1 END) as correct,
                        ROUND(
                            COUNT(CASE WHEN is_correct THEN 1 END)::float / 
                            NULLIF(COUNT(*), 0), 4
                        ) as accuracy
                    FROM predictions
                    WHERE true_label IS NOT NULL
                    GROUP BY prediction
                    ORDER BY prediction;
                """)
            
            per_digit = cursor.fetchall()
        
        return {
            "accuracy": float(overall["accuracy"]) if overall["accuracy"] else 0.0,
            "total_predictions": overall["total_predictions"],
            "correct_predictions": overall["correct_predictions"],
            "per_digit_stats": [dict(row) for row in per_digit],
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get statistics: {str(e)}"
        )
    finally:
        conn.close() 