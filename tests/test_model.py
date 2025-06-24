import pytest
import torch
import numpy as np
from models import get_model

@pytest.fixture
def model():
    """Create a test model instance."""
    return MNISTModel()

def test_model_initialization(model):
    """Test the model initialization."""
    assert isinstance(model, MNISTModel)

def test_model_forward_pass(model):
    """Test the model's forward pass."""
    # Create a random input tensor
    input_tensor = torch.randn(1, 1, 28, 28)
    
    # Forward pass
    output = model(input_tensor)
    
    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10)  # batch_size=1, num_classes=10

def test_model_prediction(model):
    """Test the model's prediction method."""
    # Create a random input tensor
    input_tensor = torch.randn(1, 1, 28, 28)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_digit].item()
    
    # Check prediction format
    assert isinstance(predicted_digit, int)
    assert 0 <= predicted_digit <= 9
    assert 0 <= confidence <= 1

def test_model_edge_cases(model):
    """Test the model with edge case inputs."""
    # Test with all zeros
    zeros = torch.zeros((1, 1, 28, 28))
    with torch.no_grad():
        output = model(zeros)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_digit].item()
    
    assert isinstance(predicted_digit, int)
    assert 0 <= predicted_digit <= 9
    assert 0 <= confidence <= 1

    # Test with all ones
    ones = torch.ones((1, 1, 28, 28))
    with torch.no_grad():
        output = model(ones)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_digit].item()
    
    assert isinstance(predicted_digit, int)
    assert 0 <= predicted_digit <= 9
    assert 0 <= confidence <= 1

def test_model_batch_processing(model):
    """Test the model's batch processing capabilities."""
    # Create a batch of test images
    batch_size = 4
    batch_tensor = torch.randn(batch_size, 1, 28, 28)
    
    # Forward pass
    with torch.no_grad():
        output = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Check output shape and properties
    assert output.shape == (batch_size, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size))

def test_model_device_handling(model):
    """Test the model's device handling capabilities."""
    # Test CPU
    model.cpu()
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model.cuda()
        assert next(model.parameters()).device.type == 'cuda'
        # Test prediction on GPU
        input_tensor = torch.randn(1, 1, 28, 28).cuda()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_digit = torch.argmax(probabilities).item()
        assert isinstance(predicted_digit, int)
        assert 0 <= predicted_digit <= 9

def test_model_forward_shape():
    model = MNISTModel()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10)  # 10 classes for MNIST

def test_model_output_range():
    model = MNISTModel()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    # Check that output values are finite (not NaN or Inf)
    assert torch.all(torch.isfinite(output))

def test_model_eval_mode():
    model = MNISTModel()
    model.eval()
    assert not model.training 