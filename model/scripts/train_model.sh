#!/bin/bash

# Get the model directory path
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create necessary directories
mkdir -p "$MODEL_DIR/data" "$MODEL_DIR/checkpoints"

# Run the training script
python -m training.train \
    --batch-size 64 \
    --epochs 10 \
    --lr 0.001 \
    --data-dir "$MODEL_DIR/data" \
    --checkpoint-dir "$MODEL_DIR/checkpoints" 