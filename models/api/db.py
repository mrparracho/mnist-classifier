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
    def __init__(self, max_retries: int = 5, retry_delay: int = 2, schema: str = "mnist_sequence"):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.schema = schema  # Use mnist_sequence schema for sequence prediction app
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
                    port=os.getenv("DB_PORT", "5432"),
                    options=f"-c search_path={self.schema},public"  # Set schema search path
                )
                logger.info(f"Connected to database at {host} using schema {self.schema}")
                return self.conn
            except psycopg2.Error as e:
                logger.warning(f"Failed to connect to {host}: {e}")
                continue
        
        raise psycopg2.OperationalError("Could not connect to database")

    def store_prediction(self, image_data: bytes, prediction: int, confidence: float, model_name: str = "cnn_mnist", session_id: Optional[str] = None) -> uuid.UUID:
        """Store a new prediction."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions 
                    (image_data, prediction, confidence, timestamp, model_name, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (image_data, prediction, confidence, datetime.utcnow(), model_name, session_id)
                )
                prediction_id = cur.fetchone()[0]
                conn.commit()
                return prediction_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing prediction: {e}")
            raise

    def update_prediction_feedback(self, prediction_id: uuid.UUID, actual_digit: int) -> Tuple[int, bool, Optional[str]]:
        """Update prediction with feedback and return session_id if available."""
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
                    RETURNING prediction, is_correct, session_id
                    """,
                    (actual_digit, actual_digit, datetime.utcnow(), str(prediction_id))
                )
                result = cur.fetchone()
                if not result:
                    raise ValueError(f"Prediction {prediction_id} not found")
                
                prediction, is_correct, session_id = result
                
                # Record feedback history
                cur.execute(
                    """
                    INSERT INTO feedback_history (prediction_id, true_label)
                    VALUES (%s, %s)
                    """,
                    (str(prediction_id), actual_digit)
                )
                conn.commit()
                return prediction, is_correct, session_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating prediction feedback: {e}")
            raise

    def update_session_feedback(self, session_id: str, actual_digit: int, original_prediction_id: uuid.UUID) -> int:
        """Update all predictions in the same session with the true label."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Update all predictions in the session except the one already updated
                cur.execute(
                    """
                    UPDATE predictions
                    SET true_label = %s,
                        is_correct = (prediction = %s),
                        feedback_submitted = true,
                        updated_at = %s
                    WHERE session_id = %s AND id != %s
                    """,
                    (actual_digit, actual_digit, datetime.utcnow(), session_id, str(original_prediction_id))
                )
                updated_count = cur.rowcount + 1  # +1 for the original prediction
                
                # Record feedback history for all updated predictions
                cur.execute(
                    """
                    INSERT INTO feedback_history (prediction_id, true_label)
                    SELECT id, %s 
                    FROM predictions 
                    WHERE session_id = %s AND id != %s AND true_label = %s
                    """,
                    (actual_digit, session_id, str(original_prediction_id), actual_digit)
                )
                
                conn.commit()
                return updated_count
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating session feedback: {e}")
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
        """Get recent prediction history."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT timestamp, prediction, true_label, confidence, model_name
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            raise

    def store_sequence_prediction(self, image_data: bytes, predicted_sequence: List[int], 
                                confidence: float, grid_size: int, session_id: str = None) -> str:
        """
        Store a sequence prediction in the database.
        
        Args:
            image_data: Image data as bytes
            predicted_sequence: List of predicted digits
            confidence: Prediction confidence
            grid_size: Size of the grid
            session_id: Optional session ID
            
        Returns:
            str: Prediction ID
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO sequence_predictions 
                        (image_data, predicted_sequence, confidence, grid_size, model_name, session_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (image_data, predicted_sequence, confidence, grid_size, 'encoder_decoder', session_id))
                    
                    prediction_id = cursor.fetchone()[0]
                    conn.commit()
                    return str(prediction_id)
                    
        except Exception as e:
            logger.error(f"Error storing sequence prediction: {e}")
            raise

    def update_sequence_prediction_feedback(self, prediction_id: str, true_sequence: List[int], 
                                          grid_size: int) -> Tuple[Dict, bool, Optional[str]]:
        """
        Update a sequence prediction with feedback.
        
        Args:
            prediction_id: ID of the prediction
            true_sequence: List of true digits
            grid_size: Size of the grid
            
        Returns:
            Tuple of (prediction_data, is_correct, session_id)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get the original prediction
                    cursor.execute("""
                        SELECT predicted_sequence, session_id
                        FROM sequence_predictions
                        WHERE id = %s
                    """, (prediction_id,))
                    
                    result = cursor.fetchone()
                    if not result:
                        raise ValueError(f"Prediction with ID {prediction_id} not found")
                    
                    predicted_sequence, session_id = result
                    
                    # Check if prediction is correct
                    is_correct = predicted_sequence == true_sequence
                    
                    # Update the prediction with feedback
                    cursor.execute("""
                        UPDATE sequence_predictions
                        SET true_sequence = %s, is_correct = %s
                        WHERE id = %s
                    """, (true_sequence, is_correct, prediction_id))
                    
                    conn.commit()
                    
                    return {
                        'predicted_sequence': predicted_sequence,
                        'true_sequence': true_sequence,
                        'is_correct': is_correct
                    }, is_correct, session_id
                    
        except Exception as e:
            logger.error(f"Error updating sequence prediction feedback: {e}")
            raise

    def get_sequence_prediction_history(self, limit: int = 10) -> List[Dict]:
        """
        Get sequence prediction history.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction history items
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    cursor.execute("""
                        SELECT timestamp, predicted_sequence, true_sequence, confidence, grid_size
                        FROM sequence_predictions
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                    
                    rows = cursor.fetchall()
                    return [
                        {
                            'timestamp': row[0],
                            'predicted_sequence': row[1],
                            'true_sequence': row[2],
                            'confidence': row[3],
                            'grid_size': row[4]
                        }
                        for row in rows
                    ]
                    
        except Exception as e:
            logger.error(f"Error getting sequence prediction history: {e}")
            return []

    def get_sequence_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the encoder-decoder model.
        
        Returns:
            Dict containing model statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_predictions,
                            COUNT(CASE WHEN is_correct = true THEN 1 END) as correct_predictions,
                            AVG(confidence) as avg_confidence
                        FROM sequence_predictions
                        WHERE model_name = 'encoder_decoder'
                    """)
                    
                    result = cursor.fetchone()
                    total_predictions, correct_predictions, avg_confidence = result
                    
                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                    
                    return {
                        'total_predictions': total_predictions,
                        'correct_predictions': correct_predictions,
                        'accuracy': accuracy,
                        'avg_confidence': float(avg_confidence) if avg_confidence else 0.0
                    }
                    
        except Exception as e:
            logger.error(f"Error getting sequence model stats: {e}")
            return {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0
            }

# Global database instance
db = Database() 