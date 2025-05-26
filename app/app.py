import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import pandas as pd
import requests

from config import config
from utils.state_management import AppState
from utils.api import MNISTApiClient
from utils.image_processing import preprocess_canvas_image, validate_canvas_image

# Initialize the application state
AppState.initialize()

# Initialize API client
api_client = MNISTApiClient(
    base_url=config.api.base_url,
    retries=config.api.retries,
    timeout=config.api.timeout
)

# Initialize canvas state if not exists
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Page configuration
st.set_page_config(
    page_title=config.title,
    page_icon=config.page_icon,
    layout=config.layout,
    initial_sidebar_state=config.initial_sidebar_state
)

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

# Create main layout with columns
col1, col2 = st.columns([3, 2])

# Drawing canvas in the first column
with col1:
    st.markdown("### Draw a digit here")
    
    # Create canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=config.canvas.stroke_width,
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
            st.rerun()

# Prediction display in the second column
with col2:
    st.markdown("### Prediction Results")
    
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
            
            with st.spinner("Classifying..."):
                try:
                    # Make prediction request to the API
                    result = api_client.predict(img_bytes)
                    # Store prediction results
                    AppState.set_prediction(result)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    # Display probability distribution
    if AppState.has_prediction():
        prediction_data = AppState.get_prediction_data()
        
        st.markdown("### Probability Distribution")
        
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
        
        # Feedback section
        if not prediction_data["feedback_submitted"]:
            st.markdown("""
            <div class="feedback-container">
                <div class="feedback-text">
                    Select True Label
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create buttons for each digit
            cols = st.columns(10)
            for i, col in enumerate(cols):
                with col:
                    if st.button(str(i), key=f"digit_{i}", use_container_width=True):
                        with st.spinner("Submitting feedback..."):
                            try:
                                api_client.submit_feedback(
                                    prediction_data["prediction_id"],
                                    actual_digit=i
                                )
                                AppState.set_feedback_submitted()
                                st.session_state["refresh_history"] = True  # Set flag
                                if i == prediction_data["prediction"]:
                                    st.success("Thank you for confirming the correct prediction!")
                                else:
                                    st.warning(f"Thank you for the correction! The model predicted {prediction_data['prediction']} but it was actually {i}.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error submitting feedback: {str(e)}")
        else:
            st.markdown("""
            <div class="feedback-container">
                <div class="feedback-thanks">
                    Thank you for your feedback!
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Draw a digit and click 'Predict' to see the results.")

# Add info in the sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses a deep learning model trained on the MNIST dataset to recognize handwritten digits.
    
    ### How to use
    1. Draw a single digit (0-9) in the canvas
    2. Click the "Predict" button
    3. View the model's prediction and confidence
    4. Provide feedback if the prediction is incorrect
    
    ### Model Details
    - Neural Network: Convolutional Neural Network (CNN)
    - Training Dataset: MNIST (70,000 handwritten digits)
    - Accuracy: ~99% on test set
    """)
    
    # Add model statistics from API
    st.markdown("### Model Statistics")
    
    if st.button("Load Statistics"):
        try:
            with st.spinner("Loading statistics..."):
                stats = api_client.get_stats()
                
                # Display overall accuracy
                overall = stats["overall"]
                st.markdown(f"""
                **Overall Accuracy**: {overall["accuracy"]*100:.2f}%  
                **Total Predictions**: {overall["total_predictions"]}  
                **Correct Predictions**: {overall["correct_predictions"]}
                """)
                
                # Create and display a per-digit accuracy chart
                if "per_digit" in stats and stats["per_digit"]:
                    # Create DataFrame with all required columns
                    digit_data = pd.DataFrame(stats["per_digit"]).assign(
                        accuracy_pct=lambda x: x["accuracy"] * 100
                    )
                    
                    # Create accuracy bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(
                        digit_data["digit"].astype(str),
                        digit_data["accuracy_pct"],
                        color='#1f77b4'
                    )
                    
                    # Add labels and title
                    ax.set_xlabel('Digit')
                    ax.set_ylabel('Accuracy (%)')
                    ax.set_ylim(0, 100)
                    ax.set_title('Accuracy by Digit')
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Display the data as a table
                    st.markdown("#### Detailed Accuracy by Digit")
                    
                    # Format the dataframe for display
                    display_df = digit_data[["digit", "total", "correct", "accuracy_pct"]].copy()
                    display_df.columns = ["Digit", "Total", "Correct", "Accuracy (%)"]
                    display_df.loc[:, "Accuracy (%)"] = display_df["Accuracy (%)"].round(2)
                    
                    # Show the table
                    st.dataframe(display_df)
        
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")


# --- Prediction History Table ---
# Fetch history only if needed
if "history_data" not in st.session_state:
    st.session_state["history_data"] = api_client.get_prediction_history(limit=10)

if st.session_state.get("refresh_history", False):
    st.session_state["history_data"] = api_client.get_prediction_history(limit=10)
    st.session_state["refresh_history"] = False

# Always display the table using the cached data
prediction_history = st.session_state["history_data"]

st.markdown("### Prediction History")
if prediction_history:
    df = pd.DataFrame(prediction_history)
    df = df.rename(columns={
        "timestamp": "Timestamp",
        "prediction": "Pred",
        "true_label": "True",
        "confidence": "Conf"
    })
    df["Conf"] = (df["Conf"] * 100).map("{:.1f}%".format)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No prediction history found.")

# Add footer
st.markdown("""
---
Made with ❤️ using PyTorch, Streamlit, and FastAPI
""")