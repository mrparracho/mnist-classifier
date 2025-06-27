#!/usr/bin/env python3
"""
Universal Model Adapter for multi-digit-scrambled-best.pt
This adapter allows the model to work with any grid size by dynamically resizing inputs.
Implements the BaseModel interface for seamless integration with the app.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from typing import List, Tuple, Optional
import os
import sys

# Add the models directory to the path
sys.path.insert(0, str(os.path.dirname(__file__)))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from base.base_model import BaseModel
from encoder_decoder.other.models import DigitSequenceModel, DigitSequenceModelConfig
from encoder_decoder.other.default_models import DEFAULT_MODEL_PARAMETERS


class UniversalModelAdapter(BaseModel):
    """
    Universal adapter for the multi-digit-scrambled-best.pt model.
    Uses the real DigitSequenceModel and config for inference.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the universal model adapter.
        
        Args:
            checkpoint_path: Path to the multi-digit-scrambled-best.pt model
        """
        if checkpoint_path is None:
            checkpoint_path = "other/multi-digit-scrambled-best.pt"
        
        super().__init__("universal_encoder_decoder", checkpoint_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = DEFAULT_MODEL_PARAMETERS["multi-digit-scrambled"]["model"]
        self.max_sequence_length = self.config.max_sequence_length
        self.base_image_size = self.config.encoder.image_width
        
        # Load the model
        self.load_model()
    
    def create_model(self) -> nn.Module:
        """Create the universal model (required by BaseModel)."""
        return self.load_model()
    
    def load_model(self) -> nn.Module:
        """Load the universal model."""
        if self.model is not None:
            return self.model
        # Instantiate the real model
        model = DigitSequenceModel("multi-digit-scrambled", self.config)
        # Load checkpoint
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path must be provided.")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        return self.model
    
    def preprocess_image(self, image_bytes: bytes, grid_size: int) -> torch.Tensor:
        """
        Preprocess image to target size for the universal model.
        
        Args:
            image_bytes: Raw image bytes
            grid_size: Size of the grid (1-10 supported)
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        target_size = grid_size * 28
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image)  # shape: (1, H, W)
        if target_size != self.base_image_size:
            padded = torch.zeros(self.base_image_size, self.base_image_size)
            start = (self.base_image_size - target_size) // 2
            padded[start:start+target_size, start:start+target_size] = image_tensor[0]
            image_tensor = padded.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        else:
            image_tensor = image_tensor.unsqueeze(0)  # shape: (1, 1, H, W)
        return image_tensor.to(self.device)
    
    def predict_sequence(self, image_bytes: bytes, grid_size: int) -> Tuple[List[int], float]:
        """
        Predict sequence for any grid size using the universal model.
        
        Args:
            image_bytes: Image data as bytes
            grid_size: Size of the grid (1-10 supported)
            
        Returns:
            Tuple of (predicted_sequence, confidence)
        """
        try:
            image_tensor = self.preprocess_image(image_bytes, grid_size)
            # Start token is 10, pad is -1
            seq = torch.full((1, self.max_sequence_length), -1, dtype=torch.long, device=self.device)
            seq[0, 0] = 10  # Start token
            for i in range(1, self.max_sequence_length):
                logits = self.model(image_tensor, seq[:, :i])
                next_token = torch.argmax(logits[0, -1]).item()
                seq[0, i] = next_token
                if next_token == 10:
                    break
            # Remove start token, stop at first 10 (end token)
            out = seq[0, 1:].tolist()
            digits = []
            for token in out:
                if token == 10:
                    break
                if 0 <= token <= 9:
                    digits.append(token)
            expected_length = grid_size * grid_size
            if len(digits) < expected_length:
                digits.extend([0] * (expected_length - len(digits)))
            elif len(digits) > expected_length:
                digits = digits[:expected_length]
            confidence = 1.0 if len(digits) == expected_length else 0.5
            return digits, confidence
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return [0] * (grid_size * grid_size), 0.0
    
    def get_preprocessing_transform(self):
        """Get preprocessing transform for the universal model."""
        return transforms.Compose([
            transforms.Resize((self.base_image_size, self.base_image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def create_universal_model_adapter(checkpoint_path: str = "other/multi-digit-scrambled-best.pt") -> UniversalModelAdapter:
    """
    Factory function to create a universal model adapter.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        UniversalModelAdapter instance
    """
    return UniversalModelAdapter(checkpoint_path)


if __name__ == "__main__":
    # Test the universal model adapter
    adapter = create_universal_model_adapter()
    
    # Test with different grid sizes
    for grid_size in [1, 2, 3, 4, 5]:
        print(f"\nTesting grid size {grid_size}x{grid_size}...")
        
        # Create a dummy image (all zeros)
        dummy_image = torch.zeros(28 * grid_size, 28 * grid_size)
        dummy_image_bytes = io.BytesIO()
        
        # Convert to PIL and save as bytes
        from PIL import Image
        Image.fromarray(dummy_image.numpy()).save(dummy_image_bytes, format='PNG')
        
        # Test prediction
        sequence, confidence = adapter.predict_sequence(dummy_image_bytes.getvalue(), grid_size)
        print(f"   Predicted sequence: {sequence}")
        print(f"   Confidence: {confidence:.3f}") 