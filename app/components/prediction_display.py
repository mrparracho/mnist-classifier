import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
from PIL import Image
import io

class PredictionDisplay:
    def __init__(self, image):
        self.image = image
        self.api_url = os.getenv("MODEL_API_URL", "http://model-service:8000")
        self.prediction = None
        self.confidence = None
        self.probabilities = None
        
    def _get_prediction(self):
        """Get prediction from the model API."""
        try:
            # Convert numpy array to PIL Image
            img = Image.fromarray((self.image * 255).astype('uint8'))
            
            # Create a byte stream for the image
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create a file-like object for the API request
            files = {'file': ('image.png', img_byte_arr, 'image/png')}
            
            # Make prediction request
            response = requests.post(f"{self.api_url}/api/v1/predict", files=files)
            response.raise_for_status()
            
            result = response.json()
            self.prediction = result["prediction"]
            self.confidence = result["confidence"]
            self.probabilities = result["probabilities"]
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the model API: {str(e)}")
            return None
    
    def show_prediction(self):
        """Display the prediction results with confidence visualization."""
        if self.prediction is None:
            self._get_prediction()
        
        if self.prediction is not None:
            # Display the prediction
            st.markdown(f"""
            <div style='text-align: center;'>
                <h2>Predicted Digit: {self.prediction}</h2>
                <p>Confidence: {self.confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create confidence bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(
                range(10),
                self.probabilities,
                color=['#1f77b4' if i != self.prediction else '#2ca02c' for i in range(10)]
            )
            
            # Add labels and title
            ax.set_xlabel('Digit')
            ax.set_ylabel('Confidence')
            ax.set_xticks(range(10))
            ax.set_xticklabels([str(i) for i in range(10)])
            ax.set_ylim(0, 1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only show labels for significant values
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f'{height:.2f}',
                        ha='center',
                        fontsize=8
                    )
            
            st.pyplot(fig)
            
            # Add feedback section
            st.markdown("### Provide Feedback")
            st.markdown("If the prediction is incorrect, please select the correct digit:")
            
            # Create a row of buttons for feedback
            cols = st.columns(10)
            for i, col in enumerate(cols):
                with col:
                    if st.button(str(i), key=f"feedback_{i}"):
                        self._submit_feedback(i)
    
    def _submit_feedback(self, true_label):
        """Submit feedback to the API."""
        try:
            feedback_data = {
                "prediction": self.prediction,
                "true_label": true_label,
                "confidence": self.confidence,
                "probabilities": self.probabilities.tolist()
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/feedback",
                json=feedback_data
            )
            response.raise_for_status()
            
            st.success(f"Thank you for your feedback! The correct digit is {true_label}.")
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error submitting feedback: {str(e)}") 