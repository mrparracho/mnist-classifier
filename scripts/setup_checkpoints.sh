#!/bin/bash

# Script to set up checkpoint directories for multi-model structure

set -e

echo "Setting up checkpoint directories for multi-model structure..."

# Create checkpoint directories
echo "Creating checkpoint directories..."
mkdir -p models/cnn_mnist/checkpoints
mkdir -p models/transformer1_mnist/checkpoints
mkdir -p models/transformer2_mnist/checkpoints

# Copy existing CNN checkpoints
if [ -f "model/checkpoints/mnist_model.pt" ]; then
    echo "Copying CNN checkpoint..."
    cp model/checkpoints/mnist_model.pt models/cnn_mnist/checkpoints/cnn_mnist.pt
    echo "✓ CNN checkpoint copied to models/cnn_mnist/checkpoints/cnn_mnist.pt"
else
    echo "⚠ No CNN checkpoint found at model/checkpoints/mnist_model.pt"
fi

# Copy existing ViT checkpoints
if [ -f "ViT/checkpoints/mnist_model.pt" ]; then
    echo "Copying ViT checkpoint..."
    cp ViT/checkpoints/mnist_model.pt models/transformer1_mnist/checkpoints/transformer1_mnist.pt
    cp ViT/checkpoints/mnist_model.pt models/transformer2_mnist/checkpoints/transformer2_mnist.pt
    echo "✓ ViT checkpoint copied to transformer model directories"
else
    echo "⚠ No ViT checkpoint found at ViT/checkpoints/mnist_model.pt"
fi

# Create placeholder files if no checkpoints exist
if [ ! -f "models/cnn_mnist/checkpoints/cnn_mnist.pt" ]; then
    echo "Creating placeholder for CNN checkpoint..."
    touch models/cnn_mnist/checkpoints/cnn_mnist.pt
    echo "# Placeholder file - replace with actual trained model" > models/cnn_mnist/checkpoints/cnn_mnist.pt
fi

if [ ! -f "models/transformer1_mnist/checkpoints/transformer1_mnist.pt" ]; then
    echo "Creating placeholder for Transformer1 checkpoint..."
    touch models/transformer1_mnist/checkpoints/transformer1_mnist.pt
    echo "# Placeholder file - replace with actual trained model" > models/transformer1_mnist/checkpoints/transformer1_mnist.pt
fi

if [ ! -f "models/transformer2_mnist/checkpoints/transformer2_mnist.pt" ]; then
    echo "Creating placeholder for Transformer2 checkpoint..."
    touch models/transformer2_mnist/checkpoints/transformer2_mnist.pt
    echo "# Placeholder file - replace with actual trained model" > models/transformer2_mnist/checkpoints/transformer2_mnist.pt
fi

echo ""
echo "Checkpoint directory structure:"
echo "models/"
echo "├── cnn_mnist/checkpoints/cnn_mnist.pt"
echo "├── transformer1_mnist/checkpoints/transformer1_mnist.pt"
echo "└── transformer2_mnist/checkpoints/transformer2_mnist.pt"
echo ""
echo "✓ Checkpoint setup completed!"
echo ""
echo "Note: If you see placeholder files, you'll need to train the models and replace them with actual checkpoints." 