import io
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import DictCursor
import torchvision.transforms as transforms
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.model import get_model
from api.routers import predictions
from api.utils import preprocess_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MNIST Classifier API",
    description="API for MNIST digit classification using PyTorch",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])

# Define Pydantic models for request/response
class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]
    prediction_id: int


class FeedbackRequest(BaseModel):
    prediction_id: int
    true_label: int


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


# Model loading
MODEL_PATH = os.getenv("MODEL_PATH", "/app/checkpoints/mnist_model.pt")


def load_model():
    """Load the PyTorch model for inference."""
    try:
        model = get_model()
        
        # Load the trained weights
        model.load_state_dict(torch.load(
            MODEL_PATH, 
            map_location=torch.device('cpu')
        ))
        
        # Set model to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise RuntimeError(f"Failed to load model: {e}")


# Image preprocessing
def preprocess_image(image_bytes):
    """
    Preprocess the image for model inference.
    
    Args:
        image_bytes: Raw bytes of the uploaded image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28 if necessary
        if image.size != (28, 28):
            image = image.resize((28, 28))
        
        # Define transformations (same as training)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
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
    true_label=None
):
    """
    Log prediction details to PostgreSQL database.
    
    Args:
        image_data: Binary image data
        prediction: Predicted digit
        confidence: Prediction confidence
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
            (timestamp, image_data, prediction, confidence, true_label)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """
            
            cursor.execute(
                query, 
                (
                    datetime.now(),
                    psycopg2.Binary(image_data),
                    prediction,
                    confidence,
                    true_label
                )
            )
            
            # Get the ID of the inserted record
            prediction_id = cursor.fetchone()[0]
            
            conn.commit()
            return prediction_id
    except Exception as e:
        logger.error(f"Database logging error: {e}")
        # Don't raise exception to prevent API errors
        return None
    finally:
        if conn:
            conn.close()


# Initialize model
model = load_model()
logger.info("Model loaded successfully")


@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST Classifier API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Predict the digit from an uploaded image.
    
    Args:
        file: Uploaded image file
        background_tasks: FastAPI background tasks
        
    Returns:
        JSON response with prediction details
    """
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
            all_probs = probabilities.tolist()
        
        # Log prediction in background
        prediction_id = log_prediction(
            image_bytes,
            prediction,
            confidence
        )
        
        # Return prediction results
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=all_probs,
            prediction_id=prediction_id
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
    Update a prediction record with user feedback.
    
    Args:
        request: Feedback request with prediction ID and true label
        
    Returns:
        JSON response confirming feedback submission
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Update prediction record with true label
            query = """
            UPDATE predictions
            SET true_label = %s,
                feedback = TRUE
            WHERE id = %s;
            """
            
            cursor.execute(
                query,
                (request.true_label, request.prediction_id)
            )
            
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Prediction ID {request.prediction_id} not found"
                )
            
            conn.commit()
            
        return {
            "message": "Feedback submitted successfully",
            "prediction_id": request.prediction_id,
            "true_label": request.true_label
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/stats")
async def get_stats():
    """
    Get model performance statistics.
    
    Returns:
        JSON response with model performance statistics
    """
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            # Query for overall accuracy
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct_predictions
                FROM predictions
                WHERE true_label IS NOT NULL;
            """)
            
            overall = cursor.fetchone()
            
            # Query for per-digit accuracy
            cursor.execute("""
                SELECT 
                    true_label,
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE true_label IS NOT NULL
                GROUP BY true_label
                ORDER BY true_label;
            """)
            
            per_digit = cursor.fetchall()
            
            # Calculate overall accuracy
            total = overall['total_predictions']
            correct = overall['correct_predictions']
            accuracy = (correct / total) if total > 0 else 0
            
            # Format per-digit statistics
            digit_stats = []
            for row in per_digit:
                digit_accuracy = (row['correct'] / row['total']) if row['total'] > 0 else 0
                digit_stats.append({
                    "digit": row['true_label'],
                    "total": row['total'],
                    "correct": row['correct'],
                    "accuracy": digit_accuracy
                })
            
            return {
                "overall": {
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "accuracy": accuracy
                },
                "per_digit": digit_stats
            }
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
