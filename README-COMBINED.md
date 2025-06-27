# Running Both MNIST Apps Simultaneously

This guide explains how to run both the original MNIST classifier app and the new sequence prediction app at the same time using a single docker-compose file.

## Port Configuration

### **Original App (Single Digit Classification)**
- **App**: http://localhost:8501
- **Nginx**: http://localhost:80 (or https://localhost:443)
- **API**: http://localhost:8000

### **Sequence App (Sequence Prediction)**
- **App**: http://localhost:8502
- **Nginx**: http://localhost:8080 (or https://localhost:8443)
- **API**: http://localhost:8000 (shared)

### **Shared Services**
- **Database**: localhost:5432 (automatically configured)
- **Model Service**: localhost:8000

## Running Both Apps

### Single Command Setup
```bash
# Run both apps together with automatic database setup
make deploy
```

Or directly with docker-compose:
```bash
docker-compose -f infrastructure/docker-compose.yml up --build
```

## What Happens Automatically

1. **Database Initialization**: The database container automatically runs all migrations including the sequence predictions table
2. **Model Service**: Loads both traditional models and the encoder-decoder model
3. **Both Apps**: Start and connect to the shared services
4. **Nginx**: Provides reverse proxy for both apps

## API Endpoints

### **Original App Endpoints**
- `POST /api/v1/predict` - Single digit prediction
- `POST /api/v1/predict-all` - Multi-model prediction
- `POST /api/v1/feedback` - Submit feedback
- `GET /api/v1/stats` - Model statistics
- `GET /api/v1/history` - Prediction history

### **Sequence App Endpoints**
- `POST /api/v1/predict-sequence` - Sequence prediction
- `POST /api/v1/feedback-sequence` - Sequence feedback
- `GET /api/v1/stats-sequence` - Sequence model statistics
- `GET /api/v1/history-sequence` - Sequence prediction history

## Database Schema

Both apps share the same database with separate tables:

### **Original App Tables**
- `predictions` - Single digit predictions
- `feedback_history` - Feedback records

### **Sequence App Tables**
- `sequence_predictions` - Sequence predictions (created automatically)

## Setup Instructions

1. **Start Both Apps** (that's it!):
   ```bash
   make deploy
   ```

2. **Access the Apps**:
   - Original App: http://localhost:8501
   - Sequence App: http://localhost:8502
   - API Documentation: http://localhost:8000/docs

## Troubleshooting

### Port Already in Use
If you get port conflicts, stop any running containers:
```bash
# Stop all containers
docker-compose -f infrastructure/docker-compose.yml down

# Then start again
make deploy
```

### Database Issues
The database automatically runs all migrations on startup. If you need to reset:
```bash
# Remove volumes and restart
docker-compose -f infrastructure/docker-compose.yml down -v
make deploy
```

### Model Loading Issues
The shared model service loads both traditional models and the encoder-decoder model:
- Traditional models: CNN, Transformer1, Transformer2
- Sequence model: Encoder-Decoder

## Development

### Running Individual Apps for Development
```bash
# Original app only
make dev

# Sequence app only (requires separate database setup)
make sequence
```

### API Testing
Test the APIs using the interactive documentation:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Original App  │    │  Sequence App   │
│   (Port 8501)   │    │  (Port 8502)    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
          ┌──────────▼───────────┐
          │   Model Service      │
          │   (Port 8000)        │
          └──────────┬───────────┘
                     │
          ┌──────────▼───────────┐
          │   Database           │
          │   (Port 5432)        │
          │   Auto-migrated      │
          └──────────────────────┘
```

## Notes

- **Single Command**: Just run `make deploy` to start everything
- **Automatic Setup**: Database migrations run automatically
- **Shared Resources**: Both apps share the same database and model service
- **No Conflicts**: All ports and endpoints are properly separated
- **Production Ready**: The setup is designed for production use 