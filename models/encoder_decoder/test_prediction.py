#!/usr/bin/env python3
"""
Test script to verify encoder-decoder prediction logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import io
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from encoder_decoder.model import EncoderDecoderMNISTClassifier

def load_mnist_digits():
    """Load actual MNIST digits for testing."""
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get one digit of each class (0-9)
    digits = {}
    for i in range(10):
        for idx in range(len(mnist_dataset)):
            image, label = mnist_dataset[idx]
            if label == i and i not in digits:
                digits[i] = image.squeeze().numpy()  # Remove channel dimension
                break
            if len(digits) == 10:
                break
    
    return digits

def create_test_image(grid_size=2, digit_indices=None):
    """Create a test image with actual MNIST digits."""
    if digit_indices is None:
        digit_indices = [1, 2, 3, 4][:grid_size*grid_size]
    
    # Load actual MNIST digits
    mnist_digits = load_mnist_digits()
    
    # Create a test image (28x28 per digit, arranged in grid)
    image_size = grid_size * 28
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Place MNIST digits in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            digit_idx = i * grid_size + j
            if digit_idx < len(digit_indices):
                digit = digit_indices[digit_idx]
                if digit in mnist_digits:
                    start_x = i * 28
                    start_y = j * 28
                    # Convert to uint8 and place in image
                    digit_image = (mnist_digits[digit] * 255).astype(np.uint8)
                    image[start_x:start_x+28, start_y:start_y+28] = digit_image
    
    return image

def test_prediction():
    """Test the prediction logic."""
    print("Testing Encoder-Decoder Prediction Logic")
    print("=" * 50)
    
    for grid_size in [1, 2]:
        print(f"\n🔍 Testing Grid Size: {grid_size}x{grid_size}")
        
        # Create model
        model = EncoderDecoderMNISTClassifier()
        model.checkpoint_path = f"encoder_decoder/checkpoints/mnist-encoder-decoder-{grid_size}-varlen.pt"
        
        # Load model
        print(f"   Loading model from: {model.checkpoint_path}")
        model.load_model()
        
        # Test with different digit combinations
        test_cases = [
            [1],  # Single digit for 1x1
            [1, 2, 3, 4],  # Four digits for 2x2
        ]
        
        for test_case in test_cases[:grid_size*grid_size]:
            print(f"\n   📝 Testing with digits: {test_case}")
            
            # Create test image with actual MNIST digits
            test_image = create_test_image(grid_size, test_case)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(test_image, mode='L')
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()
            
            # Make prediction
            print(f"   Making prediction...")
            predicted_sequence, confidence = model.predict_sequence(image_bytes, grid_size)
            
            print(f"   ✅ Prediction: {predicted_sequence}")
            print(f"   ✅ Confidence: {confidence:.3f}")
            print(f"   ✅ Expected length: {grid_size * grid_size}")
            print(f"   ✅ Actual length: {len(predicted_sequence)}")
            
            # Check if prediction is reasonable
            if len(predicted_sequence) == grid_size * grid_size:
                print(f"   ✅ Length is correct")
            else:
                print(f"   ❌ Length mismatch")
            
            # Check if all predictions are valid digits (0-9)
            valid_digits = all(0 <= digit <= 9 for digit in predicted_sequence)
            if valid_digits:
                print(f"   ✅ All predictions are valid digits")
            else:
                print(f"   ❌ Invalid digits found: {predicted_sequence}")
            
            # Check if it's not always predicting the same digit
            unique_digits = len(set(predicted_sequence))
            if unique_digits > 1:
                print(f"   ✅ Model predicts different digits: {unique_digits} unique values")
            else:
                print(f"   ⚠️  Model predicts same digit: {predicted_sequence[0]}")
            
            # Check if predictions match expected digits
            if len(predicted_sequence) == len(test_case):
                matches = sum(1 for p, e in zip(predicted_sequence, test_case) if p == e)
                accuracy = matches / len(test_case)
                print(f"   📊 Accuracy: {accuracy:.1%} ({matches}/{len(test_case)} correct)")

if __name__ == "__main__":
    test_prediction() 