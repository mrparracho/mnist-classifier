#!/usr/bin/env python3
"""
Debug script to investigate why the encoder-decoder model always predicts digit 1.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add the models directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoder_decoder.model import EncoderDecoderMNISTClassifier
from torchvision import datasets, transforms


def test_model_with_different_inputs(grid_size: int):
    """Test the model with various input patterns to see if it always predicts 1."""
    
    print(f"🔍 TESTING MODEL BIAS FOR {grid_size}x{grid_size} GRID")
    print("=" * 60)
    
    # Load the model
    checkpoint_path = f"encoder_decoder/checkpoints/mnist-encoder-decoder-{grid_size}-varlen.pt"
    model = EncoderDecoderMNISTClassifier(checkpoint_path=checkpoint_path)
    model.load_model()
    
    # Test 1: All zeros input
    print("\n📊 TEST 1: All zeros input")
    image_size = grid_size * 28
    zeros_input = torch.zeros(1, 1, image_size, image_size)
    
    model.model.eval()
    with torch.no_grad():
        output = model.model(zeros_input, max_length=grid_size * grid_size + 2)
        print(f"   Input: All zeros ({image_size}x{image_size})")
        print(f"   Output tokens: {output[0].tolist()}")
        
        # Extract digits
        digits = []
        for token in output[0]:
            token_id = token.item()
            if token_id == 10:  # start token
                continue
            elif token_id == 11:  # finish token
                break
            elif 0 <= token_id <= 9:
                digits.append(token_id)
        print(f"   Predicted digits: {digits}")
    
    # Test 2: All ones input
    print("\n📊 TEST 2: All ones input")
    ones_input = torch.ones(1, 1, image_size, image_size)
    
    with torch.no_grad():
        output = model.model(ones_input, max_length=grid_size * grid_size + 2)
        print(f"   Input: All ones ({image_size}x{image_size})")
        print(f"   Output tokens: {output[0].tolist()}")
        
        digits = []
        for token in output[0]:
            token_id = token.item()
            if token_id == 10:  # start token
                continue
            elif token_id == 11:  # finish token
                break
            elif 0 <= token_id <= 9:
                digits.append(token_id)
        print(f"   Predicted digits: {digits}")
    
    # Test 3: Random noise input
    print("\n📊 TEST 3: Random noise input")
    noise_input = torch.randn(1, 1, image_size, image_size)
    
    with torch.no_grad():
        output = model.model(noise_input, max_length=grid_size * grid_size + 2)
        print(f"   Input: Random noise ({image_size}x{image_size})")
        print(f"   Output tokens: {output[0].tolist()}")
        
        digits = []
        for token in output[0]:
            token_id = token.item()
            if token_id == 10:  # start token
                continue
            elif token_id == 11:  # finish token
                break
            elif 0 <= token_id <= 9:
                digits.append(token_id)
        print(f"   Predicted digits: {digits}")
    
    # Test 4: Check model parameters for bias
    print("\n📊 TEST 4: Model parameter analysis")
    
    # Check if any parameters are NaN or infinite
    has_nan = False
    has_inf = False
    param_count = 0
    
    for name, param in model.model.named_parameters():
        param_count += 1
        if torch.isnan(param).any():
            has_nan = True
            print(f"   ❌ NaN found in {name}")
        if torch.isinf(param).any():
            has_inf = True
            print(f"   ❌ Inf found in {name}")
    
    print(f"   Total parameters: {param_count}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    
    # Test 5: Check output projection layer bias
    print("\n📊 TEST 5: Output projection analysis")
    
    # Find the output projection layer
    output_proj = None
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 13:  # vocab size
            output_proj = module
            print(f"   Found output projection: {name}")
            break
    
    if output_proj:
        bias = output_proj.bias
        if bias is not None:
            print(f"   Output bias shape: {bias.shape}")
            print(f"   Output bias values: {bias.tolist()}")
            
            # Check if bias favors digit 1 (index 1)
            digit_1_bias = bias[1].item()
            print(f"   Digit 1 bias: {digit_1_bias}")
            
            # Check if this is the highest bias
            max_bias_idx = torch.argmax(bias).item()
            max_bias_val = bias[max_bias_idx].item()
            print(f"   Highest bias: digit {max_bias_idx} = {max_bias_val}")
            
            if max_bias_idx == 1:
                print("   ⚠️  Digit 1 has the highest bias!")
            else:
                print(f"   Digit {max_bias_idx} has the highest bias")


def check_mnist_distribution():
    """Check the distribution of digits in the MNIST dataset."""
    
    print("\n📊 MNIST DATASET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root="encoder_decoder/data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Count digit frequencies
    digit_counts = [0] * 10
    total_samples = len(train_dataset)
    
    print(f"   Total training samples: {total_samples}")
    
    for i in range(min(10000, total_samples)):  # Sample first 10k for speed
        _, label = train_dataset[i]
        digit_counts[label] += 1
    
    print("\n   Digit distribution (first 10k samples):")
    for digit, count in enumerate(digit_counts):
        percentage = (count / 10000) * 100
        print(f"   Digit {digit}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Check if digit 1 is overrepresented
    digit_1_percentage = (digit_counts[1] / 10000) * 100
    if digit_1_percentage > 12:  # More than 12% would be unusual
        print(f"\n   ⚠️  Digit 1 is overrepresented: {digit_1_percentage:.1f}%")
    else:
        print(f"\n   ✅ Digit 1 distribution looks normal: {digit_1_percentage:.1f}%")


def main():
    """Main diagnostic function."""
    
    print("🔍 ENCODER-DECODER MODEL BIAS DIAGNOSTIC")
    print("=" * 80)
    
    # Test different grid sizes
    for grid_size in [1, 2]:
        test_model_with_different_inputs(grid_size)
    
    # Check MNIST distribution
    check_mnist_distribution()
    
    print("\n" + "=" * 80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print("If the model always predicts digit 1 regardless of input, possible causes:")
    print("1. Training data bias (digit 1 overrepresented)")
    print("2. Model architecture issue (output projection bias)")
    print("3. Checkpoint corruption or incorrect training")
    print("4. Model got stuck in local minimum during training")
    print("5. Learning rate too high causing instability")


if __name__ == "__main__":
    main() 