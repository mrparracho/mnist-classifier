import os
import time
from typing import Dict, Any, Optional
import requests
from requests.exceptions import RequestException

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class MNISTApiClient:
    def __init__(self, base_url: Optional[str] = None, retries: int = 3, timeout: int = 10):
        """Initialize the API client.
        
        Args:
            base_url: Base URL for the API. Defaults to MODEL_API_URL environment variable.
            retries: Number of retries for failed requests.
            timeout: Timeout in seconds for requests.
        """
        self.base_url = base_url or os.getenv("MODEL_API_URL", "http://model-service:8000")
        self.retries = retries
        self.timeout = timeout

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request with retry mechanism."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        kwargs.setdefault('timeout', self.timeout)

        for attempt in range(self.retries):
            try:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                if attempt == self.retries - 1:
                    raise APIError(f"Failed after {self.retries} attempts: {str(e)}")
                time.sleep(1 * (attempt + 1))  # Exponential backoff

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Make a prediction request.
        
        Args:
            image_data: Image bytes to predict.
            
        Returns:
            Dict containing prediction results.
            
        Raises:
            APIError: If the request fails after retries.
        """
        files = {'file': ('image.png', image_data, 'image/png')}
        return self._make_request('POST', 'predict', files=files)

    def submit_feedback(self, prediction_id: str, actual_digit: int) -> Dict[str, Any]:
        """Submit feedback for a prediction.
        
        Args:
            prediction_id: ID of the prediction to provide feedback for.
            actual_digit: The correct digit (0-9).
            
        Returns:
            Dict containing feedback submission results.
            
        Raises:
            APIError: If the request fails after retries.
        """
        data = {
            'prediction_id': prediction_id,
            'actual_digit': actual_digit
        }
        return self._make_request('POST', 'feedback', json=data)

    def get_stats(self) -> Dict[str, Any]:
        """Get model performance statistics.
        
        Returns:
            Dict containing model statistics.
            
        Raises:
            APIError: If the request fails after retries.
        """
        return self._make_request('GET', 'stats')

    def get_prediction_history(self, limit: int = 10):
        """Fetch prediction history from the backend API."""
        return self._make_request('GET', f'history?limit={limit}') 