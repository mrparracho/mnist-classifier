"""
API client for communicating with the MNIST classifier API.
"""

import requests
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MNISTApiClient:
    """
    Client for interacting with the MNIST classifier API.
    """
    
    def __init__(self, base_url: str, retries: int = 3, timeout: int = 30):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
            retries: Number of retry attempts
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retries}): {e}")
                continue
        
        # This should never be reached due to the raise in the loop, but satisfies type checker
        raise requests.RequestException("All retry attempts failed")
    
    def predict(self, image_bytes: bytes, model_name: str = "cnn_mnist") -> Dict[str, Any]:
        """
        Make a prediction using the specified model.
        
        Args:
            image_bytes: Image data as bytes
            model_name: Name of the model to use
            
        Returns:
            Dict[str, Any]: Prediction results
            
        Raises:
            requests.RequestException: If request fails
        """
        files = {'file': ('image.png', image_bytes, 'image/png')}
        params = {'model_name': model_name}
        
        response = self._make_request(
            method='POST',
            endpoint='/api/v1/predict',
            files=files,
            params=params
        )
        
        return response.json()
    
    def predict_all_models(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Make predictions using all available models for comparison.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dict[str, Any]: Predictions from all models
            
        Raises:
            requests.RequestException: If request fails
        """
        files = {'file': ('image.png', image_bytes, 'image/png')}
        
        response = self._make_request(
            method='POST',
            endpoint='/api/v1/predict-all',
            files=files
        )
        
        return response.json()
    
    def submit_feedback(self, prediction_id: int, actual_digit: int, model_name: str = "cnn_mnist") -> Dict[str, Any]:
        """
        Submit feedback for a prediction.
        
        Args:
            prediction_id: ID of the prediction
            actual_digit: Actual digit value
            model_name: Name of the model used
            
        Returns:
            Dict[str, Any]: Response from feedback submission
            
        Raises:
            requests.RequestException: If request fails
        """
        data = {
            'prediction_id': str(prediction_id),  # Convert to string for UUID
            'actual_digit': actual_digit,         # Use correct field name
            'model_name': model_name
        }
        
        response = self._make_request(
            method='POST',
            endpoint='/api/v1/feedback',
            json=data
        )
        
        return response.json()
    
    def get_statistics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model performance statistics.
        
        Args:
            model_name: Optional model name to filter statistics
            
        Returns:
            Dict[str, Any]: Model statistics
            
        Raises:
            requests.RequestException: If request fails
        """
        params = {}
        if model_name:
            params['model_name'] = model_name
        
        response = self._make_request(
            method='GET',
            endpoint='/api/v1/stats',
            params=params
        )
        
        return response.json()
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List[Dict[str, Any]]: List of available models
            
        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request(
            method='GET',
            endpoint='/api/v1/models'
        )
        
        data = response.json()
        return data.get('models', [])
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model information
            
        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request(
            method='GET',
            endpoint=f'/api/v1/models/{model_name}'
        )
        
        return response.json()
    
    def get_model_statistics(self, model_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model statistics
            
        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request(
            method='GET',
            endpoint=f'/api/v1/models/{model_name}/stats'
        )
        
        return response.json()
    
    def get_all_models_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all models.
        
        Returns:
            List[Dict[str, Any]]: Statistics for all models
            
        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request(
            method='GET',
            endpoint='/api/v1/models/stats/all'
        )
        
        return response.json()
    
    def get_prediction_history(self, limit: int = 10, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get prediction history.
        
        Args:
            limit: Maximum number of predictions to return
            model_name: Optional model name to filter
            
        Returns:
            List[Dict[str, Any]]: Prediction history
            
        Raises:
            requests.RequestException: If request fails
        """
        params: Dict[str, Any] = {'limit': limit}
        if model_name:
            params['model_name'] = model_name
        
        response = self._make_request(
            method='GET',
            endpoint='/api/v1/history',
            params=params
        )
        
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.
        
        Returns:
            Dict[str, Any]: Health status
            
        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request(
            method='GET',
            endpoint='/health'
        )
        
        return response.json() 