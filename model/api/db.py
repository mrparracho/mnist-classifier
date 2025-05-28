import os
import logging
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, max_retries: int = 5, retry_delay: int = 2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conn = None
        
    def get_connection(self):
        """Get a database connection with retry mechanism."""
        if self.conn and not self.conn.closed:
            return self.conn
            
        hosts = [
            os.getenv("DB_HOST", "mnist-db"),
            "db",
            "localhost"
        ]
        
        for host in hosts:
            try:
                self.conn = psycopg2.connect(
                    dbname=os.getenv("DB_NAME", "mnist"),
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD", "postgres"),
                    host=host,
                    port=os.getenv("DB_PORT", "5432")
                )
                logger.info(f"Connected to database at {host}")
                return self.conn
            except psycopg2.Error as e:
                logger.warning(f"Failed to connect to {host}: {e}")
                continue
        
        raise psycopg2.OperationalError("Could not connect to database")

    def store_prediction(self, image_data: bytes, prediction: int, confidence: float) -> uuid.UUID:
        """Store a new prediction."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions 
                    (image_data, prediction, confidence, timestamp)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (image_data, prediction, confidence, datetime.utcnow())
                )
                prediction_id = cur.fetchone()[0]
                conn.commit()
                return prediction_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing prediction: {e}")
            raise

    def update_prediction_feedback(self, prediction_id: uuid.UUID, actual_digit: int) -> Tuple[int, bool]:
        """Update prediction with feedback."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE predictions
                    SET true_label = %s,
                        is_correct = (prediction = %s),
                        feedback_submitted = true,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING prediction, is_correct
                    """,
                    (actual_digit, actual_digit, datetime.utcnow(), str(prediction_id))
                )
                result = cur.fetchone()
                if not result:
                    raise ValueError(f"Prediction {prediction_id} not found")
                
                prediction, is_correct = result
                
                # Record feedback history
                cur.execute(
                    """
                    INSERT INTO feedback_history (prediction_id, true_label)
                    VALUES (%s, %s)
                    """,
                    (str(prediction_id), actual_digit)
                )
                conn.commit()
                return prediction, is_correct
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating prediction feedback: {e}")
            raise

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                # Get overall stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN feedback_submitted THEN 1 END) as total_feedback,
                        COALESCE(
                            (
                                SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END)::float / 
                                NULLIF(COUNT(CASE WHEN true_label IS NOT NULL THEN 1 END), 0)
                            )::numeric(10,4),
                            0
                        ) as accuracy
                    FROM predictions
                """)
                overall = cur.fetchone()
                
                # Get per-digit stats
                cur.execute("""
                    SELECT 
                        prediction as digit,
                        COUNT(*) as total,
                        COUNT(CASE WHEN true_label IS NOT NULL THEN 1 END) as with_feedback,
                        SUM(CASE WHEN prediction = true_label THEN 1 ELSE 0 END) as correct
                    FROM predictions
                    WHERE true_label IS NOT NULL
                    GROUP BY prediction
                    ORDER BY prediction
                """)
                per_digit = cur.fetchall()
                
                # Format response
                digit_stats = []
                for row in per_digit:
                    accuracy = row['correct'] / row['with_feedback'] if row['with_feedback'] > 0 else 0
                    digit_stats.append({
                        'digit': row['digit'],
                        'total': row['total'],
                        'correct': row['correct'],
                        'accuracy': round(accuracy, 4)
                    })
                
                return {
                    'overall': {
                        'total_predictions': overall['total_predictions'],
                        'total_feedback': overall['total_feedback'],
                        'accuracy': overall['accuracy'],
                        'correct_predictions': sum(d['correct'] for d in digit_stats)
                    },
                    'per_digit': digit_stats
                }
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            raise

    def get_prediction_history(self, limit: int = 10):
        """Return recent predictions with timestamp, prediction, true_label, and confidence."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT timestamp, prediction, true_label, confidence
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching prediction history: {e}")
            raise

# Create a global database instance
db = Database() 