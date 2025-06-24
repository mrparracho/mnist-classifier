import os
import asyncpg
from datetime import datetime
from typing import List, Optional, Dict, Any

# Database connection pool
pool = None

async def init_db():
    """Initialize database connection pool."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            database=os.getenv("DB_NAME", "mnist"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )

async def store_prediction(prediction_id: str, image_data: bytes, prediction: int, confidence: float, model_name: str = "cnn_mnist"):
    """Store a prediction in the database."""
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO predictions (id, timestamp, image_data, prediction, confidence, model_name)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, prediction_id, datetime.utcnow(), image_data, prediction, confidence, model_name)

async def update_prediction(prediction_id: str, true_label: int, model_name: str = "cnn_mnist"):
    """Update a prediction with feedback."""
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE predictions
            SET true_label = $1, 
                is_correct = (prediction = $1),
                feedback_submitted = true,
                model_name = $3
            WHERE id = $2
        """, true_label, prediction_id, model_name)

async def get_statistics(model_name: Optional[str] = None) -> dict:
    """Get model performance statistics."""
    async with pool.acquire() as conn:
        if model_name:
            # Get statistics for specific model
            overall = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN is_correct THEN 1 END) as correct_predictions,
                    ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::float / COUNT(*), 4) as accuracy
                FROM predictions
                WHERE true_label IS NOT NULL AND model_name = $1
            """, model_name)
            
            per_digit = await conn.fetch("""
                SELECT 
                    prediction as digit,
                    COUNT(*) as total,
                    COUNT(CASE WHEN is_correct THEN 1 END) as correct,
                    ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::float / COUNT(*), 4) as accuracy
                FROM predictions
                WHERE true_label IS NOT NULL AND model_name = $1
                GROUP BY prediction
                ORDER BY prediction
            """, model_name)
        else:
            # Get overall statistics for all models
            overall = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN is_correct THEN 1 END) as correct_predictions,
                    ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::float / COUNT(*), 4) as accuracy
                FROM predictions
                WHERE true_label IS NOT NULL
            """)
            
            per_digit = await conn.fetch("""
                SELECT 
                    prediction as digit,
                    COUNT(*) as total,
                    COUNT(CASE WHEN is_correct THEN 1 END) as correct,
                    ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::float / COUNT(*), 4) as accuracy
                FROM predictions
                WHERE true_label IS NOT NULL
                GROUP BY prediction
                ORDER BY prediction
            """)
        
        return {
            "accuracy": overall["accuracy"],
            "total_predictions": overall["total_predictions"],
            "correct_predictions": overall["correct_predictions"],
            "per_digit_stats": [dict(row) for row in per_digit]
        }

async def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available models from database."""
    async with pool.acquire() as conn:
        models = await conn.fetch("""
            SELECT name, display_name, description, model_type, version, is_active
            FROM models
            WHERE is_active = TRUE
            ORDER BY name
        """)
        return [dict(model) for model in models]

async def get_model_statistics(model_name: str) -> Dict[str, Any]:
    """Get statistics for a specific model."""
    async with pool.acquire() as conn:
        # Get model performance using the database function
        stats = await conn.fetchrow("""
            SELECT * FROM get_model_statistics($1)
        """, model_name)
        
        if stats:
            import json
            per_digit_stats = stats["per_digit_stats"]
            if isinstance(per_digit_stats, str):
                per_digit_stats = json.loads(per_digit_stats)
            
            return {
                "model_name": stats["model_name"],
                "total_predictions": stats["total_predictions"],
                "correct_predictions": stats["correct_predictions"],
                "accuracy": float(stats["accuracy"]) if stats["accuracy"] else 0.0,
                "per_digit_stats": per_digit_stats
            }
        else:
            return {
                "model_name": model_name,
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "per_digit_stats": {}
            }

async def get_all_model_statistics() -> List[Dict[str, Any]]:
    """Get statistics for all models."""
    async with pool.acquire() as conn:
        stats = await conn.fetch("""
            SELECT * FROM get_model_statistics()
        """)
        import json
        result = []
        for stat in stats:
            per_digit_stats = stat["per_digit_stats"]
            if isinstance(per_digit_stats, str):
                per_digit_stats = json.loads(per_digit_stats)
            
            result.append({
                "model_name": stat["model_name"],
                "total_predictions": stat["total_predictions"],
                "correct_predictions": stat["correct_predictions"],
                "accuracy": float(stat["accuracy"]) if stat["accuracy"] else 0.0,
                "per_digit_stats": per_digit_stats
            })
        return result 