import pytest
import streamlit as st
from app.components.drawing_canvas import DrawingCanvas
from app.components.prediction_display import PredictionDisplay
import numpy as np
from unittest.mock import patch, MagicMock
import torch
from PIL import Image
import io

def test_drawing_canvas_initialization():
    """Test the initialization of the DrawingCanvas component."""
    canvas = DrawingCanvas()
    assert canvas.canvas_width == 280
    assert canvas.canvas_height == 280
    assert canvas.stroke_width == 20
    assert canvas.stroke_color == "#FFFFFF"
    assert canvas.bg_color == "#000000"

def test_prediction_display_initialization():
    """Test the initialization of the PredictionDisplay component."""
    test_image = np.zeros((28, 28), dtype=np.float32)
    display = PredictionDisplay(test_image)
    assert display.image is test_image
    assert display.prediction is None
    assert display.confidence is None
    assert display.probabilities is None

def test_prediction_display_get_prediction():
    """Test the _get_prediction method of the PredictionDisplay."""
    test_image = np.zeros((28, 28), dtype=np.float32)
    display = PredictionDisplay(test_image)
    
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prediction": 5,
        "confidence": 0.95,
        "probabilities": [0.01] * 10
    }
    
    with patch('requests.post', return_value=mock_response):
        display._get_prediction()
        assert display.prediction == 5
        assert display.confidence == 0.95
        assert len(display.probabilities) == 10

def test_prediction_display_show_prediction():
    """Test the show_prediction method of the PredictionDisplay."""
    test_image = np.zeros((28, 28), dtype=np.float32)
    display = PredictionDisplay(test_image)
    
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prediction": 5,
        "confidence": 0.95,
        "probabilities": [0.01] * 10
    }
    
    with patch('requests.post', return_value=mock_response):
        with patch('streamlit.markdown') as mock_markdown:
            with patch('streamlit.pyplot') as mock_plt:
                display.show_prediction()
                mock_markdown.assert_called()
                mock_plt.assert_called()

def test_prediction_display_submit_feedback():
    """Test the _submit_feedback method of the PredictionDisplay."""
    test_image = np.zeros((28, 28), dtype=np.float32)
    display = PredictionDisplay(test_image)
    display.prediction = 5
    display.confidence = 0.95
    display.probabilities = np.array([0.01] * 10)
    
    # Mock the API response
    mock_response = MagicMock()
    
    with patch('requests.post', return_value=mock_response):
        with patch('streamlit.success') as mock_success:
            display._submit_feedback(7)
            mock_success.assert_called_once()

# Mock Streamlit's session state
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit's session state and other functions."""
    with patch('streamlit.session_state', {}):
        with patch('streamlit.write') as mock_write:
            with patch('streamlit.image') as mock_image:
                with patch('streamlit.progress') as mock_progress:
                    yield {
                        'write': mock_write,
                        'image': mock_image,
                        'progress': mock_progress
                    }

@pytest.fixture
def mock_api_client():
    """Mock the API client for testing."""
    with patch('app.utils.api_client') as mock:
        mock.predict.return_value = {
            'prediction': 5,
            'confidence': 0.95,
            'probabilities': [0.01] * 10
        }
        yield mock