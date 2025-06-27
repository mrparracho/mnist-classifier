#!/bin/bash

# Script to set up encoder-decoder checkpoint files
# This script creates the directory structure and provides instructions for placing checkpoint files

echo "Setting up encoder-decoder checkpoint directories..."

# Create the checkpoints directory
mkdir -p models/encoder_decoder/checkpoints

# Create placeholder files (you should replace these with actual .pt files)
echo "Creating placeholder files for encoder-decoder checkpoints..."

# Create placeholder files for each grid size
for grid_size in 1 2 3 4; do
    checkpoint_file="models/encoder_decoder/checkpoints/mnist-encoder-decoder-${grid_size}-varlen.pt"
    
    if [ ! -f "$checkpoint_file" ]; then
        echo "Creating placeholder for grid ${grid_size}x${grid_size} checkpoint..."
        echo "# Placeholder for mnist-encoder-decoder-${grid_size}-varlen.pt" > "$checkpoint_file"
        echo "# Replace this file with your actual trained model checkpoint" >> "$checkpoint_file"
        echo "# This should be a PyTorch .pt file containing the model weights" >> "$checkpoint_file"
    else
        echo "Checkpoint file for grid ${grid_size}x${grid_size} already exists: $checkpoint_file"
    fi
done

echo ""
echo "Setup complete!"
echo ""
echo "IMPORTANT: You need to replace the placeholder files with actual trained model checkpoints:"
echo ""
echo "For each grid size, place your trained model checkpoint at:"
echo "  - Grid 1x1: models/encoder_decoder/checkpoints/mnist-encoder-decoder-1-varlen.pt"
echo "  - Grid 2x2: models/encoder_decoder/checkpoints/mnist-encoder-decoder-2-varlen.pt"
echo "  - Grid 3x3: models/encoder_decoder/checkpoints/mnist-encoder-decoder-3-varlen.pt"
echo "  - Grid 4x4: models/encoder_decoder/checkpoints/mnist-encoder-decoder-4-varlen.pt"
echo ""
echo "Each .pt file should contain the trained weights for the encoder-decoder model"
echo "specifically trained for that grid size." 