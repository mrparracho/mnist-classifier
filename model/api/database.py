import os
import asyncpg
from datetime import datetime
from typing import List, Optional

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

async def store_prediction(prediction_id: str, image_data: bytes, prediction: int, confidence: float):
    """Store a prediction in the database."""
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO predictions (id, timestamp, image_data, prediction, confidence)
            VALUES ($1, $2, $3, $4, $5)
        """, prediction_id, datetime.utcnow(), image_data, prediction, confidence)

async def update_prediction(prediction_id: str, true_label: int):
    """Update a prediction with feedback."""
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE predictions
            SET true_label = $1, 
                is_correct = (prediction = $1),
                feedback_submitted = true
            WHERE id = $2
        """, true_label, prediction_id)

async def get_statistics() -> dict:
    """Get model performance statistics."""
    async with pool.acquire() as conn:
        # Get overall statistics
        overall = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN is_correct THEN 1 END) as correct_predictions,
                ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::float / COUNT(*), 4) as accuracy
            FROM predictions
            WHERE true_label IS NOT NULL
        """)
        
        # Get per-digit statistics
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