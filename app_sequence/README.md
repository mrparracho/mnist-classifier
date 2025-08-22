# MNIST Sequence Classifier App

This is a Streamlit application that uses an encoder-decoder model to predict sequences of digits based on a single handwritten digit input.

## Features

- **Grid Size Selection**: Choose from 1x1, 2x2, 3x3, or 4x4 grids
- **Sequence Prediction**: The encoder-decoder model predicts a sequence of digits based on the input
- **Visual Grid Display**: Shows predicted digits in a visual grid layout
- **Feedback System**: Users can provide true sequences for model improvement
- **History Tracking**: Maintains prediction history with grid sizes

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: FastAPI with encoder-decoder model
- **Database**: PostgreSQL for storing predictions and feedback
- **Model**: Vision Transformer encoder-decoder architecture

## Usage

1. Select a grid size (1x1 to 4x4)
2. Draw a single digit (0-9) in the canvas
3. Click "Predict Sequence" to get the model's prediction
4. View the predicted sequence in a grid layout
5. Optionally provide feedback with the true sequence

## API Endpoints

- `POST /api/v1/predict-sequence`: Predict a sequence of digits
- `POST /api/v1/feedback-sequence`: Submit feedback for a prediction
- `GET /api/v1/history-sequence`: Get prediction history
- `GET /api/v1/stats-sequence`: Get model statistics

## Running the App

```bash
# Using Docker Compose
docker-compose -f infrastructure/docker-compose-sequence.yml up

# Or run directly with Streamlit
cd app_sequence
streamlit run app.py
```

The app will be available at `http://localhost:8502`

## Model Details

The encoder-decoder model uses:
- **Encoder**: Vision Transformer that processes the input image
- **Decoder**: Autoregressive decoder that generates digit sequences
- **Output**: Sequence of digits based on the selected grid size

## Database Schema

The app uses a `sequence_predictions` table with the following structure:
- `id`: Primary key
- `image_data`: Binary image data
- `predicted_sequence`: Array of predicted digits
- `true_sequence`: Array of true digits (optional)
- `confidence`: Prediction confidence
- `grid_size`: Size of the grid
- `model_name`: Name of the model used
- `session_id`: Session identifier
- `timestamp`: Prediction timestamp
- `is_correct`: Whether the prediction was correct 