# MNIST Digit Classifier

![MNIST Example](https://img.shields.io/badge/MNIST-Digit%20Classifier-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)

A web application for recognizing handwritten digits using a PyTorch neural network trained on the MNIST dataset.

**[Live Demo](https://mnist.example.com)**

- Interactive digit drawing interface
- Real-time digit recognition
- Model performance tracking
- User feedback collection

## Quick Start

1. Clone and setup:
   ```bash
   git clone https://github.com/mrparracho/mnist-classifier.git
   cd mnist-classifier
   make setup
   ```

2. Start the application:
   ```bash
   make dev
   ```

3. Access the application:
   - Streamlit App: http://localhost:8501
   - FastAPI Docs: http://localhost:8000/docs

## Development

### Prerequisites

- Python 3.11
- Docker and Docker Compose
- Git

### Commands

```bash
make setup     # Create virtual environments and install dependencies
make dev       # Start development environment
make train     # Train the model
make test      # Run tests
make deploy    # Deploy to production
```

### Project Structure

```
mnist-classifier/
│
├── app/                  # Streamlit web application
├── model/                # Model training and serving
├── db/                   # Database setup and migrations
├── infrastructure/       # Docker and infrastructure configs
├── scripts/              # Utility scripts
└── tests/                # Test suite
```

## API Endpoints

- `POST /api/v1/predict` - Get digit prediction
- `GET /api/v1/stats` - View model statistics
- `POST /api/v1/feedback` - Submit prediction feedback
- `GET /api/v1/history` - View prediction history
