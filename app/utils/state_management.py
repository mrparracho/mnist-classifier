from typing import Optional, Dict, Any, List
import streamlit as st
from PIL import Image

class AppState:
    """Manages application state using Streamlit's session state."""
    
    @staticmethod
    def initialize() -> None:
        """Initialize all required session state variables."""
        if "initialized" not in st.session_state:
            st.session_state.update({
                "prediction": None,
                "confidence": None,
                "probabilities": None,
                "prediction_id": None,
                "canvas_image": None,
                "feedback_submitted": False,
                "error_message": None,
                "initialized": True
            })

    @staticmethod
    def reset_prediction() -> None:
        """Reset all prediction-related state."""
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.probabilities = None
        st.session_state.prediction_id = None
        st.session_state.canvas_image = None
        st.session_state.feedback_submitted = False
        st.session_state.error_message = None

    @staticmethod
    def set_prediction(prediction_data: Dict[str, Any]) -> None:
        """Set prediction data in the session state.
        
        Args:
            prediction_data: Dictionary containing prediction results from the API.
        """
        st.session_state.prediction = prediction_data.get("predicted_digit")
        st.session_state.confidence = prediction_data.get("confidence")
        st.session_state.probabilities = prediction_data.get("probabilities")
        st.session_state.prediction_id = prediction_data.get("prediction_id")
        st.session_state.feedback_submitted = False
        st.session_state.error_message = None

    @staticmethod
    def set_canvas_image(image: Image.Image) -> None:
        """Store the canvas image in session state.
        
        Args:
            image: PIL Image object from the canvas.
        """
        st.session_state.canvas_image = image

    @staticmethod
    def set_error(message: str) -> None:
        """Set an error message in the session state.
        
        Args:
            message: Error message to display.
        """
        st.session_state.error_message = message

    @staticmethod
    def set_feedback_submitted(submitted: bool = True) -> None:
        """Mark feedback as submitted.
        
        Args:
            submitted: Whether feedback was submitted successfully.
        """
        st.session_state.feedback_submitted = submitted

    @staticmethod
    def get_prediction_data() -> Dict[str, Any]:
        """Get all prediction-related data from session state.
        
        Returns:
            Dictionary containing current prediction data.
        """
        return {
            "prediction": st.session_state.prediction,
            "confidence": st.session_state.confidence,
            "probabilities": st.session_state.probabilities,
            "prediction_id": st.session_state.prediction_id,
            "feedback_submitted": st.session_state.feedback_submitted
        }

    @staticmethod
    def has_prediction() -> bool:
        """Check if there is a current prediction.
        
        Returns:
            True if there is a prediction, False otherwise.
        """
        return st.session_state.prediction is not None

    @staticmethod
    def has_error() -> bool:
        """Check if there is an error message.
        
        Returns:
            True if there is an error message, False otherwise.
        """
        return st.session_state.error_message is not None

    @staticmethod
    def get_error() -> Optional[str]:
        """Get the current error message.
        
        Returns:
            Current error message or None if no error.
        """
        return st.session_state.error_message 