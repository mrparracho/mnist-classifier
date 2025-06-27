import requests
import logging
from typing import List, Dict, Any, Optional
import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class SequenceApiClient:
    """API client for sequence prediction using encoder-decoder model."""
    
    def __init__(self, base_url: str, retries: int = 3, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()
    
    def predict_sequence(self, image_bytes: bytes, grid_size: int) -> Dict[str, Any]:
        """
        Predict a sequence of digits using the encoder-decoder model.
        
        Args:
            image_bytes: Image data as bytes
            grid_size: Size of the grid (1, 2, 3, or 4)
            
        Returns:
            Dict containing prediction results
        """
        try:
            url = f"{self.base_url}/api/v1/predict-sequence"
            
            files = {'file': ('image.png', image_bytes, 'image/png')}
            params = {'grid_size': grid_size}
            
            response = self.session.post(
                url,
                files=files,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'predicted_sequence': result.get('predicted_sequence', []),
                'confidence': result.get('confidence', 0.0),
                'prediction_id': result.get('prediction_id', ''),
                'grid_size': grid_size
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise Exception(f"Failed to get prediction: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing prediction response: {str(e)}")
            raise Exception(f"Error processing prediction: {str(e)}")
    
    def submit_sequence_feedback(self, prediction_id: str, true_sequence: List[int], grid_size: int) -> Dict[str, Any]:
        """
        Submit feedback for a sequence prediction.
        
        Args:
            prediction_id: ID of the prediction
            true_sequence: List of true digits
            grid_size: Size of the grid
            
        Returns:
            Dict containing feedback response
        """
        try:
            url = f"{self.base_url}/api/v1/feedback-sequence"
            
            data = {
                'prediction_id': prediction_id,
                'true_sequence': true_sequence,
                'grid_size': grid_size
            }
            
            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise Exception(f"Failed to submit feedback: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing feedback response: {str(e)}")
            raise Exception(f"Error processing feedback: {str(e)}")
    
    def get_prediction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get prediction history for sequence predictions.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction history items
        """
        try:
            url = f"{self.base_url}/api/v1/history-sequence"
            params = {'limit': limit}
            
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing history response: {str(e)}")
            return []
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the encoder-decoder model.
        
        Returns:
            Dict containing model statistics
        """
        try:
            url = f"{self.base_url}/api/v1/stats-sequence"
            
            response = self.session.get(
                url,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error processing statistics response: {str(e)}")
            return {} 