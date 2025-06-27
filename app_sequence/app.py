import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from streamlit_drawable_canvas import st_canvas
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

def create_grid_background(width, height, grid_size):
    """Create a background image with grid lines."""
    # Create a black background
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    # Calculate grid cell size
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    # Draw grid lines in white
    for i in range(1, grid_size):
        # Vertical lines
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill='white', width=2)
        
        # Horizontal lines
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill='white', width=2)
    
    return img

def display_sequence_in_grid(sequence, grid_size):
    """Display the predicted sequence in a grid layout."""
    if len(sequence) != grid_size ** 2:
        st.warning(f"Expected {grid_size**2} digits, got {len(sequence)}")
        return
    
    # Create a grid layout
    cols = st.columns(grid_size)
    
    for i, col in enumerate(cols):
        with col:
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(sequence):
                    digit = sequence[idx]
                    # Create a styled box for each digit
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        padding: 10px;
                        margin: 5px;
                        text-align: center;
                        font-size: 24px;
                        font-weight: bold;
                        background-color: #f0f0f0;
                    ">
                        {digit}
                    </div>
                    """, unsafe_allow_html=True)

def display_sequence_in_line(sequence):
    """Display the predicted sequence in a horizontal line with black background and white digits."""
    if not sequence:
        return
    digits_html = ''.join([
        f'<span style="display:inline-block; min-width:48px; min-height:48px; font-size:2.5rem; color:#fff; text-align:center; margin:0 8px;">{digit}</span>'
        for digit in sequence
    ])
    st.markdown(f'''
    <div style="background:#111; border-radius:10px; padding:24px 12px; display:flex; justify-content:center; align-items:center; margin-bottom:1rem;">
        {digits_html}
    </div>
    ''', unsafe_allow_html=True)

# Set Streamlit page config as the VERY FIRST Streamlit command
st.set_page_config(
    page_title="MNIST Sequence Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else after set_page_config
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import requests

# Create config inline to avoid import issues
class Config:
    title = "MNIST Sequence Classifier"
    page_icon = "🔢"
    layout = "wide"
    initial_sidebar_state = "expanded"
    style_path = "styles/main.css"
    
    class api:
        base_url = os.getenv("MODEL_API_URL", "http://model-service:8000")
        retries = 3
        timeout = 10
    
    class canvas:
        width = 500
        height = 500
        stroke_width = 25
        stroke_color = "#FFFFFF"
        bg_color = "#000000"
        target_size = (28, 28)

config = Config()

# Now import modules that use streamlit
from utils.state_management import AppState
from utils.image_processing import preprocess_canvas_image, validate_canvas_image

# Now it's safe to use st.session_state and AppState
AppState.initialize()
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "selected_grid_size" not in st.session_state:
    st.session_state.selected_grid_size = 1

# Load custom CSS
with open(config.style_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.markdown("""
<div class="header-container">
    <h1>MNIST Sequence Classifier</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Draw a single digit (0-9) in the canvas below and select a grid size.
The encoder-decoder model will predict a sequence of digits based on your input.
""")

# Grid size selection in sidebar
with st.sidebar:
    st.title("Grid Configuration")
    grid_size = st.selectbox(
        "Select Grid Size",
        options=[1, 2, 3, 4],
        index=0,
        format_func=lambda x: f"{x}x{x} Grid ({x**2} digits)"
    )
    st.session_state.selected_grid_size = grid_size
    
    st.markdown(f"**Current Grid:** {grid_size}x{grid_size}")
    st.markdown(f"**Expected Output:** {grid_size**2} digits")
    
    # Model information
    st.title("Model Information")
    st.markdown("""
    **Encoder-Decoder Model**
    
    This model uses a vision transformer encoder-decoder architecture to predict sequences of digits.
    
    - **Encoder**: Processes the input image using patch embeddings and self-attention
    - **Decoder**: Generates a sequence of digits autoregressively
    - **Output**: Predicts {} digits in sequence
    """.format(grid_size * grid_size))

# Create main layout with columns
col1, col2 = st.columns([3, 2])

# Drawing canvas in the first column
with col1:
    st.markdown(f"### Draw a digit here")
    
    # Create grid background
    grid_bg = create_grid_background(config.canvas.width, config.canvas.height, grid_size)
    
    # Instructions based on grid size
    if grid_size == 1:
        st.info("Draw a single digit in the canvas below.")
    else:
        st.info(f"Draw a single digit in any of the {grid_size}x{grid_size} grid cells below.")
    
    # Create canvas for drawing with grid background
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=config.canvas.stroke_width,
        stroke_color=config.canvas.stroke_color,
        background_image=grid_bg,
        width=config.canvas.width,
        height=config.canvas.height,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Buttons for submit and clear
    col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
    
    with col1_1:
        submit_button = st.button("Predict Sequence", use_container_width=True)
    
    with col1_2:
        clear_button = st.button("Clear", use_container_width=True)
        if clear_button:
            # Reset prediction state
            AppState.reset_prediction()
            # Increment canvas key to force a reset
            st.session_state.canvas_key += 1
            # Clear the refresh history flag
            if "refresh_history" in st.session_state:
                del st.session_state["refresh_history"]
            st.rerun()

    # Recent Predictions table under the buttons
    st.markdown("### Recent Predictions")
    
    # Check if we need to refresh history due to feedback submission
    refresh_needed = st.session_state.get("refresh_history", False)
    if refresh_needed:
        # Clear the flag
        st.session_state["refresh_history"] = False
    
    try:
        # Get prediction history for encoder-decoder model (last 10 predictions)
        from utils.api import SequenceApiClient
        api_client = SequenceApiClient(
            base_url=config.api.base_url,
            retries=config.api.retries,
            timeout=config.api.timeout
        )
        history = api_client.get_prediction_history(limit=10)
        
        if history:
            # Format history data for display
            history_data = []
            for pred in history:
                # Handle predicted sequence - convert list to string
                predicted_seq = pred.get('predicted_sequence', [])
                if isinstance(predicted_seq, list):
                    predicted_seq_str = ' '.join(map(str, predicted_seq))
                else:
                    predicted_seq_str = str(predicted_seq)
                
                # Handle true sequence - convert list to string or show dash for None
                true_seq = pred.get('true_sequence', None)
                if true_seq is not None and isinstance(true_seq, list):
                    true_seq_str = ' '.join(map(str, true_seq))
                elif true_seq is not None:
                    true_seq_str = str(true_seq)
                else:
                    true_seq_str = '-'
                
                history_data.append({
                    'Timestamp': pred.get('timestamp', 'N/A'),
                    'Grid Size': pred.get('grid_size', 'N/A'),
                    'Predicted Sequence': predicted_seq_str,
                    'True Sequence': true_seq_str,
                    'Confidence': f"{pred.get('confidence', 0):.3f}"
                })
            
            # Create and display dataframe
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent predictions available. Make a prediction to see history!")
            
    except Exception as e:
        st.warning(f"Could not load prediction history: {str(e)}")

# Prediction display in the second column
with col2:
    st.markdown("### Sequence Prediction Results")
    
    # Process and predict when submit button is clicked
    if submit_button and canvas_result.image_data is not None:
        # Validate the image
        error = validate_canvas_image(canvas_result.image_data)
        if error:
            st.error(error)
        else:
            # Process the image
            img_bytes = preprocess_canvas_image(
                canvas_result.image_data,
                target_size=config.canvas.target_size
            )
            
            with st.spinner(f"Predicting sequence for {grid_size}x{grid_size} grid..."):
                try:
                    # Make prediction request to the API
                    from utils.api import SequenceApiClient
                    api_client = SequenceApiClient(
                        base_url=config.api.base_url,
                        retries=config.api.retries,
                        timeout=config.api.timeout
                    )
                    result = api_client.predict_sequence(img_bytes, grid_size)
                    # Store prediction results
                    AppState.set_prediction(result)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    # Display sequence prediction results
    if AppState.has_prediction():
        prediction_data = AppState.get_prediction_data()
        
        # Display the predicted sequence in a horizontal line
        predicted_sequence = prediction_data.get('predicted_sequence', [])
        confidence = prediction_data.get('confidence', 0.0)
        
        if predicted_sequence:
            st.markdown(f"**Predicted Sequence:**")
            st.markdown(f"**Confidence:** {confidence:.3f}")
            
            # Show which model was used
            model_name = prediction_data.get('model_name', f'encoder_decoder_grid_{grid_size}')
            st.markdown(f"**Model Used:** {model_name}")
            
            # Display as a horizontal line
            display_sequence_in_line(predicted_sequence)
            
            # Feedback section
            prediction_id = prediction_data.get('prediction_id', '')
            if prediction_id and not prediction_data.get("feedback_submitted", False):
                st.markdown("**Provide feedback for the sequence:**")
                
                # Create a text input for the true sequence
                true_sequence_input = st.text_input(
                    "Enter the true sequence (space-separated digits):",
                    placeholder="e.g., 1 2 3 4",
                    key="true_sequence_input"
                )
                
                if st.button("Submit Feedback", key="submit_feedback"):
                    if true_sequence_input.strip():
                        try:
                            # Parse the input sequence
                            true_sequence = [int(x.strip()) for x in true_sequence_input.split()]
                            
                            with st.spinner("Submitting feedback..."):
                                api_client.submit_sequence_feedback(
                                    prediction_id,
                                    true_sequence,
                                    grid_size
                                )
                                
                                # Mark as having feedback submitted
                                prediction_data["feedback_submitted"] = True
                                prediction_data["true_sequence"] = true_sequence
                                
                                # Set flag to refresh history on next render
                                st.session_state["refresh_history"] = True
                                
                                st.success("Thank you for the feedback!")
                                st.rerun()
                        except ValueError:
                            st.error("Please enter valid digits separated by spaces (e.g., 1 2 3 4)")
                        except Exception as e:
                            st.error(f"Error submitting feedback: {str(e)}")
                    else:
                        st.error("Please enter the true sequence")
            elif prediction_data.get("feedback_submitted", False):
                st.markdown("✅ Feedback submitted")
                if prediction_data.get("true_sequence"):
                    st.markdown(f"True sequence: {prediction_data.get('true_sequence')}")
        else:
            st.error("Failed to get prediction")
    else:
        st.info("Draw a digit and click 'Predict Sequence' to see the results.")

# Add info in the sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses encoder-decoder models trained on the MNIST dataset to predict sequences of digits.
    
    ### How to use
    1. Select a grid size (1x1 to 4x4)
    2. Draw a single digit (0-9) in the canvas
    3. Click the "Predict Sequence" button
    4. View the predicted sequence in a grid layout
    5. Provide feedback if desired
    
    ### Model Architecture
    - **Encoder**: Vision Transformer that processes the input image
    - **Decoder**: Autoregressive decoder that generates digit sequences
    - **Grid-Specific Models**: Different trained models for each grid size (1x1, 2x2, 3x3, 4x4)
    - **Output**: Sequence of digits based on the selected grid size
    
    ### Model Files
    Each grid size uses a specific trained model:
    - 1x1: `mnist-encoder-decoder-1-varlen.pt`
    - 2x2: `mnist-encoder-decoder-2-varlen.pt`
    - 3x3: `mnist-encoder-decoder-3-varlen.pt`
    - 4x4: `mnist-encoder-decoder-4-varlen.pt`
    """)
    
    # Add model statistics from API
    st.markdown("### Model Statistics")
    
    if st.button("Load Model Statistics"):
        try:
            from utils.api import SequenceApiClient
            api_client = SequenceApiClient(
                base_url=config.api.base_url,
                retries=config.api.retries,
                timeout=config.api.timeout
            )
            
            # Get statistics for the encoder-decoder model
            stats = api_client.get_model_statistics()
            
            if stats:
                accuracy = stats.get('accuracy', 0)
                total_predictions = stats.get('total_predictions', 0)
                
                st.metric(
                    label="Encoder-Decoder Accuracy",
                    value=f"{accuracy:.2%}",
                    help=f"Based on {total_predictions} predictions"
                )
            else:
                st.info("No statistics available yet")
            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")

# Add footer
st.markdown("""
---
Made with ❤️ using PyTorch, Streamlit, and FastAPI
""") 