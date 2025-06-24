import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import torch
from models import get_model
from datetime import datetime
from unittest.mock import patch

# Test database URL
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create a test model path
TEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "test_model.pt")

@pytest.fixture(scope="session")
def mock_model():
    """Create a dummy model for testing."""
    return MNISTModel()

@pytest.fixture(scope="session")
def app(mock_model):
    """Create a FastAPI app with mocked model loading."""
    with patch('model.api.main.load_model', return_value=mock_model):
        from models.api.main import app
        return app

@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine

@pytest.fixture(scope="function")
def test_db(test_db_engine):
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(scope="function")
def client(app, test_db):
    """Create a test client with a test database."""
    from models.api.database import init_db
    
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="session")
def test_image():
    """Create a test MNIST image."""
    import numpy as np
    return np.random.rand(28, 28).astype(np.float32)

@pytest.fixture(scope="session")
def test_prediction():
    """Create a test prediction result."""
    return {
        "digit": 5,
        "confidence": 0.95,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@pytest.fixture(scope="function")
def test_prediction_record(test_db):
    """Create a test prediction record in the database."""
    from model.api.models import Prediction
    prediction = Prediction(
        digit=5,
        confidence=0.95,
        timestamp=datetime.utcnow()
    )
    test_db.add(prediction)
    test_db.commit()
    test_db.refresh(prediction)
    return prediction

@pytest.fixture(scope="function")
def test_feedback_record(test_db, test_prediction_record):
    """Create a test feedback record in the database."""
    from models.api.models import FeedbackRequest
    feedback = Feedback(
        prediction_id=test_prediction_record.id,
        is_correct=True,
        correct_digit=5
    )
    test_db.add(feedback)
    test_db.commit()
    test_db.refresh(feedback)
    return feedback

@pytest.fixture(scope="function")
def test_database_with_data(test_db, test_prediction_record, test_feedback_record):
    """Create a test database with sample data."""
    # Add more predictions
    from model.api.models import Prediction
    for i in range(5):
        prediction = Prediction(
            digit=i,
            confidence=0.9,
            timestamp=datetime.utcnow()
        )
        test_db.add(prediction)
    test_db.commit()
    return test_db 