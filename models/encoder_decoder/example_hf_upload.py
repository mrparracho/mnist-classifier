#!/usr/bin/env python3
"""
Example script showing how to train and upload a model to Hugging Face Hub.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    """Example of training with Hugging Face upload."""
    
    # Example command line arguments for training with HF upload
    cmd_args = [
        "python", "models/encoder_decoder/train.py",
        "--epochs", "5",  # Quick training for demo
        "--batch-size", "32",
        "--max-grid-size", "3",  # Small grid for faster training
        "--static-length", "2",  # Fixed 2 digits for simplicity
    ]
    
    print("Example Hugging Face upload command:")
    print(" ".join(cmd_args))
    print()
    
    print("To use Hugging Face upload:")
    print("1. Install Hugging Face dependencies:")
    print("   pip install transformers huggingface-hub")
    print()
    print("2. Set environment variables:")
    print("   export HUGGING_FACE_USER='your-username'")
    print("   export HUGGING_FACE_TOKEN='your-hf-token-here'")
    print()
    print("3. Get your Hugging Face token:")
    print("   - Go to https://huggingface.co/settings/tokens")
    print("   - Create a new token with 'write' permissions")
    print()
    print("4. Run the training command")
    print()
    print("The model will be automatically uploaded to:")
    print("https://huggingface.co/your-username/mnist-encoder-decoder-{grid-size}-{length-mode}")
    print("Examples:")
    print("  - mnist-encoder-decoder-5-varlen (5x5 grid, variable length)")
    print("  - mnist-encoder-decoder-3-fixlen (3x3 grid, fixed length)")
    print()
    print("You can then load it with:")
    print("from models.encoder_decoder.model import EncoderDecoder")
    print("model = EncoderDecoder.from_pretrained('your-username/mnist-encoder-decoder-5-varlen')")
    print()
    print("Alternative: Set environment variables inline:")
    print("HUGGING_FACE_USER=your-username HUGGING_FACE_TOKEN=your-token python models/encoder_decoder/train.py --epochs 5")

if __name__ == "__main__":
    main() 