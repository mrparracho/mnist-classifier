"""
Router for model-related endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from api.models import ModelInfo, ModelListResponse, ModelStatisticsResponse
from api.database import get_available_models, get_model_statistics, get_all_model_statistics

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """
    Get list of available models.
    
    Returns:
        ModelListResponse: List of available models with metadata
    """
    try:
        models_data = await get_available_models()
        models = [
            ModelInfo(
                name=model["name"],
                display_name=model["display_name"],
                description=model["description"],
                model_type=model["model_type"],
                version=model["version"],
                is_active=model["is_active"]
            )
            for model in models_data
        ]
        
        return ModelListResponse(models=models, total=len(models))
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelInfo: Model information
        
    Raises:
        HTTPException: If model not found
    """
    try:
        models_data = await get_available_models()
        model_data = next((m for m in models_data if m["name"] == model_name), None)
        
        if not model_data:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return ModelInfo(
            name=model_data["name"],
            display_name=model_data["display_name"],
            description=model_data["description"],
            model_type=model_data["model_type"],
            version=model_data["version"],
            is_active=model_data["is_active"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch model information")

@router.get("/models/{model_name}/stats", response_model=ModelStatisticsResponse)
async def get_model_stats(model_name: str):
    """
    Get statistics for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelStatisticsResponse: Model statistics
        
    Raises:
        HTTPException: If model not found or error occurs
    """
    try:
        stats = await get_model_statistics(model_name)
        
        return ModelStatisticsResponse(
            model_name=stats["model_name"],
            total_predictions=stats["total_predictions"],
            correct_predictions=stats["correct_predictions"],
            accuracy=stats["accuracy"],
            per_digit_stats=stats["per_digit_stats"]
        )
    except Exception as e:
        logger.error(f"Error fetching stats for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch model statistics")

@router.get("/models/stats/all")
async def get_all_models_stats():
    """
    Get statistics for all models.
    
    Returns:
        List[ModelStatisticsResponse]: Statistics for all models
    """
    try:
        all_stats = await get_all_model_statistics()
        
        return [
            ModelStatisticsResponse(
                model_name=stats["model_name"],
                total_predictions=stats["total_predictions"],
                correct_predictions=stats["correct_predictions"],
                accuracy=float(stats["accuracy"]) if stats["accuracy"] else 0.0,
                per_digit_stats=stats["per_digit_stats"]
            )
            for stats in all_stats
        ]
    except Exception as e:
        logger.error(f"Error fetching all models stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch all models statistics") 