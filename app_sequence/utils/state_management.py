import streamlit as st
from typing import Dict, Any, Optional

class AppState:
    """Manages application state for sequence predictions."""
    
    @staticmethod
    def initialize():
        """Initialize the application state."""
        if "sequence_prediction" not in st.session_state:
            st.session_state.sequence_prediction = None
    
    @staticmethod
    def set_prediction(prediction_data: Dict[str, Any]):
        """Store prediction results in session state."""
        st.session_state.sequence_prediction = prediction_data
    
    @staticmethod
    def get_prediction_data() -> Optional[Dict[str, Any]]:
        """Get the current prediction data."""
        return st.session_state.get("sequence_prediction")
    
    @staticmethod
    def has_prediction() -> bool:
        """Check if there's a prediction stored."""
        return st.session_state.get("sequence_prediction") is not None
    
    @staticmethod
    def reset_prediction():
        """Reset the prediction state."""
        st.session_state.sequence_prediction = None 