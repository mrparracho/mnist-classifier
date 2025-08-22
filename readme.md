# MNIST Multi-Model Classifier

![MNIST Example](https://img.shields.io/badge/MNIST-Multi%20Model%20Classifier-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)

A comprehensive web application for recognizing handwritten digits using multiple PyTorch neural network architectures trained on the MNIST dataset.

**[Live Demo](https://195.201.26.250)**

## Features

### Main Application (Port 8501)
- Interactive digit drawing interface
- Multi-model selection (CNN, Transformer1, Transformer2)
- Real-time digit recognition
- Model performance comparison
- User feedback collection

### Sequence Application (Port 8502)
- Sequence prediction using encoder-decoder model
- Grid size selection (1x1 to 4x4)
- Visual grid display of predicted sequences
- Sequence feedback system

## Quick Start

1. Clone and setup:
   ```bash
   git clone https://github.com/mrparracho/mnist-classifier.git
   cd mnist-classifier
   make setup
   ```

2. Train the model (dataset is downloaded as part of the training):
   ```bash
   make train # trains only the CNN model as default.
   make train-all # trains all the 4 models
   ```

3. Start the application:
   ```bash
   make dev
   ```

3. Access the applications:
   - Main App: http://localhost:8501
   - Sequence App: http://localhost:8502
   - FastAPI Docs: http://localhost:8000/docs
   - PgAdmin: http://localhost:5050

## Development

### Prerequisites

- Python 3.11
- Docker and Docker Compose
- Git

### Commands

```bash
make setup     # Create virtual environments and install dependencies
make dev       # Start development environment
make train     # Train the models
make test      # Run tests
make deploy    # Deploy to production
```

### Project Structure

```
mnist-classifier/
│
├── app/                  # Main Streamlit application (port 8501)
├── app_sequence/         # Sequence prediction app (port 8502)
├── models/               # Multi-model architecture
│   ├── cnn_mnist/       # CNN model implementation
│   ├── transformer1_mnist/ # Transformer1 model
│   ├── transformer2_mnist/ # Transformer2 model
│   ├── encoder_decoder/ # Encoder-decoder for sequences
│   └── api/             # FastAPI model serving
├── model/                # Legacy single model (deprecated)
├── db/                   # Database setup and migrations
├── infrastructure/       # Docker and infrastructure configs
├── scripts/              # Utility scripts
└── tests/                # Test suite
```

## Model Architecture

### CNN MNIST
- Convolutional Neural Network for digit classification
- 2 convolutional layers with max pooling
- Fully connected layers with dropout
- Optimized for MNIST digit recognition

### Transformer1 MNIST
- Vision Transformer with encoder layers
- Patch-based image processing
- Self-attention mechanism
- Suitable for digit classification tasks

### Transformer2 MNIST
- Enhanced Vision Transformer with MLP layers
- Improved feature extraction
- Better performance on complex digit patterns
- Advanced attention mechanisms

### Encoder-Decoder
- Sequence prediction model
- Vision Transformer encoder
- Autoregressive decoder
- Predicts digit sequences based on input

## API Endpoints

### Main Application
- `POST /api/v1/predict` - Get digit prediction
- `GET /api/v1/stats` - View model statistics
- `POST /api/v1/feedback` - Submit prediction feedback
- `GET /api/v1/history` - View prediction history

### Sequence Application
- `POST /api/v1/predict-sequence` - Predict digit sequence
- `POST /api/v1/feedback-sequence` - Submit sequence feedback
- `GET /api/v1/history-sequence` - View sequence history
- `GET /api/v1/stats-sequence` - Get sequence model statistics

## Database Schema

### Main Predictions
- `predictions` table for digit classification
- Stores image data, predictions, confidence, and feedback

### Sequence Predictions
- `sequence_predictions` table for sequence classification
- Stores image data, predicted sequences, grid sizes, and feedback

## Model Training

Each model can be trained independently:

```bash
# Train CNN model
make train-cnn

# Train Transformer1 model
make train-transformer1

# Train Transformer2 model
make train-transformer2

# Train Encoder-Decoder model
make train-enconder-decoder
```

## Configuration

Model configurations are centralized in `models/config.py`:
- Model registry with all available models
- Checkpoint paths and model parameters
- Environment variable overrides
- Model activation status

## Docker Services

- **app**: Main Streamlit application (port 8501)
- **app-sequence**: Sequence prediction app (port 8502)
- **model-service**: FastAPI model serving (port 8000)
- **db**: PostgreSQL database (port 5432)
- **pgadmin**: Database administration (port 5050)
