import streamlit as st

# Set Streamlit page config as the VERY FIRST Streamlit command
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else after set_page_config
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os

# Create config inline to avoid import issues
class Config:
    title = "MNIST Digit Classifier"
    page_icon = "✍️"
    layout = "wide"
    initial_sidebar_state = "expanded"
    style_path = "styles/main.css"
    
    class api:
        base_url = os.getenv("MODEL_API_URL", "http://model-service:8000")
        retries = 3
        timeout = 10
    
    class canvas:
        width = 280
        height = 280
        # MNIST stroke width calculation:
        # MNIST digits have ~2-3 pixel stroke width in 28x28 images
        # Canvas scale factor: 280/28 = 10
        # Optimal stroke width: 2.5 * 10 = 25 pixels
        base_stroke_width = 25  # Matches MNIST training data characteristics
        stroke_color = "#FFFFFF"
        bg_color = "#000000"
        target_size = (28, 28)
        
        @staticmethod
        def get_stroke_width(grid_size=1):
            """Calculate stroke width to match MNIST training data characteristics."""
            # For the main app, we use a single digit, so return base width
            # This matches the MNIST stroke width characteristics
            return Config.canvas.base_stroke_width

config = Config()

# Now import modules that use streamlit
from utils.state_management import AppState
from utils.image_processing import preprocess_canvas_image, validate_canvas_image

# Now it's safe to use st.session_state and AppState
AppState.initialize()
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "cnn_mnist"

# Load custom CSS
with open(config.style_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.markdown("""
<div class="header-container">
    <h1>MNIST Digit Classifier</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Draw a single digit (0-9) in the canvas below and submit for classification.
The model will predict which digit you've drawn and show its confidence.
""")

# Model selection in sidebar
with st.sidebar:
    st.title("Model Comparison")
    from utils.api import MNISTApiClient
    api_client = MNISTApiClient(
        base_url=config.api.base_url,
        retries=config.api.retries,
        timeout=config.api.timeout
    )
    # Get available models
    try:
        available_models = api_client.get_available_models()
        model_info = {model['name']: model for model in available_models}
        
        if available_models:
            st.markdown("**Available Models:**")
            for model in available_models:
                st.markdown(f"• **{model['display_name']}** ({model['model_type']})")
            
            st.markdown("*All models will be used for prediction comparison*")
        else:
            st.warning("No models available")
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")

# Create main layout with columns
col1, col2 = st.columns([3, 2])

# Drawing canvas in the first column
with col1:
    st.markdown(f"### Draw a digit here")
    
    # Create canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=config.canvas.get_stroke_width(),
        stroke_color=config.canvas.stroke_color,
        background_color=config.canvas.bg_color,
        width=config.canvas.width,
        height=config.canvas.height,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Buttons for submit and clear
    col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
    
    with col1_1:
        submit_button = st.button("Predict", use_container_width=True)
    
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
        # Get prediction history for all models (last 10 predictions)
        from utils.api import MNISTApiClient
        api_client = MNISTApiClient(
            base_url=config.api.base_url,
            retries=config.api.retries,
            timeout=config.api.timeout
        )
        history = api_client.get_prediction_history(limit=10)
        
        if history:
            # Format history data for display
            history_data = []
            for pred in history:
                history_data.append({
                    'Timestamp': pred.get('timestamp', 'N/A'),
                    'Model': pred.get('model_name', 'N/A'),
                    'Predicted': pred.get('prediction', 'N/A'),
                    'True Label': pred.get('true_label', 'N/A') if pred.get('true_label') is not None else '-',
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
    st.markdown("### Prediction Results Comparison")
    
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
            
            with st.spinner(f"Classifying with all models..."):
                try:
                    # Make prediction request to the API with all models
                    from utils.api import MNISTApiClient
                    api_client = MNISTApiClient(
                        base_url=config.api.base_url,
                        retries=config.api.retries,
                        timeout=config.api.timeout
                    )
                    result = api_client.predict_all_models(img_bytes)
                    # Store prediction results
                    AppState.set_prediction(result)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    # Display comparison of all models
    if AppState.has_prediction():
        prediction_data = AppState.get_prediction_data()
        
        # Check if we have multiple model predictions
        if 'predictions' in prediction_data and isinstance(prediction_data['predictions'], list):
            predictions = prediction_data['predictions']
            
            # Create tabs for each model
            model_names = [pred.get('model_name', 'Unknown') for pred in predictions]
            tabs = st.tabs(model_names)
            
            for i, (tab, prediction) in enumerate(zip(tabs, predictions)):
                with tab:
                    model_name = prediction.get('model_name', 'Unknown')
                    predicted_digit = prediction.get('predicted_digit', -1)
                    confidence = prediction.get('confidence', 0.0)
                    probabilities = prediction.get('probabilities', [0.0] * 10)
                    
                    if predicted_digit >= 0:  # Valid prediction
                        # Display prediction
                        st.markdown(f"**Predicted Digit:** {predicted_digit}")
                        st.markdown(f"**Confidence:** {confidence:.3f}")
                        
                        # Create bar plot for this model
                        fig, ax = plt.subplots(figsize=(8, 3))
                        bars = ax.bar(
                            range(10),
                            probabilities,
                            color=['#1f77b4' if j != predicted_digit else '#2ca02c' for j in range(10)]
                        )
                        
                        # Customize plot
                        ax.set_xticks(range(10))
                        ax.set_xlabel("Digit")
                        ax.set_ylabel("Probability")
                        ax.set_title(f"Probabilities - {model_name}")
                        ax.set_ylim(0, 1)
                        
                        # Display plot
                        st.pyplot(fig)
                        
                        # Feedback section for this model
                        prediction_id = prediction.get('prediction_id', '')
                        if prediction_id and not prediction.get("feedback_submitted", False):
                            st.markdown(f"**Provide feedback for {model_name}:**")
                            
                            # Create buttons for each digit
                            cols = st.columns(10)
                            for j, col in enumerate(cols):
                                with col:
                                    if st.button(str(j), key=f"digit_{j}_{model_name}", use_container_width=True):
                                        with st.spinner("Submitting feedback..."):
                                            try:
                                                api_client.submit_feedback(
                                                    prediction_id,
                                                    actual_digit=j,
                                                    model_name=model_name
                                                )
                                                # Mark this specific prediction as having feedback submitted
                                                prediction["feedback_submitted"] = True
                                                prediction["true_label"] = j
                                                
                                                # Set flag to refresh history on next render
                                                st.session_state["refresh_history"] = True
                                                
                                                # Update the multi-model predictions in session state
                                                if st.session_state.multi_predictions is not None:
                                                    for pred in st.session_state.multi_predictions.get("predictions", []):
                                                        if pred.get("prediction_id") == prediction_id:
                                                            pred["feedback_submitted"] = True
                                                            pred["true_label"] = j
                                                
                                                if j == predicted_digit:
                                                    st.success(f"Thank you for confirming {model_name}'s correct prediction!")
                                                else:
                                                    st.warning(f"Thank you for the correction! {model_name} predicted {predicted_digit} but it was actually {j}.")
                                                
                                                # Force a rerun to refresh the UI and history
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error submitting feedback: {str(e)}")
                        elif prediction.get("feedback_submitted", False):
                            st.markdown("✅ Feedback submitted for this model")
                            if prediction.get("true_label") is not None:
                                st.markdown(f"True label: {prediction.get('true_label')}")
                    else:
                        st.error(f"Failed to get prediction from {model_name}")
            
            # Summary comparison
            st.markdown("### Summary Comparison")
            summary_data = []
            for prediction in predictions:
                model_name = prediction.get('model_name', 'Unknown')
                predicted_digit = prediction.get('predicted_digit', -1)
                confidence = prediction.get('confidence', 0.0)
                
                summary_data.append({
                    'Model': model_name,
                    'Prediction': predicted_digit if predicted_digit >= 0 else 'Failed',
                    'Confidence': f"{confidence:.3f}" if predicted_digit >= 0 else 'N/A'
                })
            
            if summary_data:
                import pandas as pd
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        else:
            # Fallback for single model prediction (backward compatibility)
            st.markdown("### Single Model Prediction")
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(
                range(10),
                prediction_data["probabilities"],
                color=['#1f77b4' if i != prediction_data["prediction"] else '#2ca02c' for i in range(10)]
            )
            
            # Customize plot
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities for Each Digit")
            
            # Display plot
            st.pyplot(fig)
    else:
        st.info("Draw a digit and click 'Predict' to see the results.")

# Add info in the sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses multiple deep learning models trained on the MNIST dataset to recognize handwritten digits.
    
    ### How to use
    1. Draw a single digit (0-9) in the canvas
    2. Click the "Predict" button
    3. View all models' predictions and confidences in separate tabs
    4. Compare the results in the summary table
    5. Provide feedback for each model if desired
    
    ### Available Models
    - **CNN MNIST**: Convolutional Neural Network
    - **Transformer1 MNIST**: Vision Transformer with encoder layers
    - **Transformer2 MNIST**: Vision Transformer with encoder layers and MLP
    """)
    
    # Add model statistics from API
    st.markdown("### Model Statistics")
    
    if st.button("Load All Model Statistics"):
        try:
            # Get statistics for all models
            all_stats = api_client.get_all_models_statistics()
            
            if all_stats:
                st.markdown("**Performance Comparison:**")
                for stats in all_stats:
                    model_name = stats.get('model_name', 'Unknown')
                    accuracy = stats.get('accuracy', 0)
                    total_predictions = stats.get('total_predictions', 0)
                    
                    st.metric(
                        label=f"{model_name} Accuracy",
                        value=f"{accuracy:.2%}",
                        help=f"Based on {total_predictions} predictions"
                    )
            else:
                st.info("No statistics available yet")
            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
    
    # Add comparison of all models
    st.markdown("### Compare All Models")
    
    if st.button("Compare Models"):
        try:
            # Get all available models first
            available_models = api_client.get_available_models()
            all_stats = api_client.get_all_models_statistics()
            
            comparison_data = []
            
            # Create a mapping of model stats for easy lookup
            stats_map = {stats['model_name']: stats for stats in all_stats}
            
            for model in available_models:
                model_name = model['name']
                stats = stats_map.get(model_name, {})
                
                comparison_data.append({
                    'Model': model['display_name'],
                    'Type': model['model_type'].upper(),
                    'Accuracy': f"{stats.get('accuracy', 0):.2%}",
                    'Total': stats.get('total_predictions', 0),
                    'Correct': stats.get('correct_predictions', 0)
                })
            
            if comparison_data:
                import pandas as pd
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No models available")
                
        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")

# Add footer
st.markdown("""
---
Made with ❤️ using PyTorch, Streamlit, and FastAPI
""")