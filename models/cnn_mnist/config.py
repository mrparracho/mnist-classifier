"""
Configuration for CNN MNIST model.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CNNMNISTConfig:
    """Configuration for CNN MNIST model."""
    
    # Model architecture
    input_channels: int = 1
    conv1_channels: int = 32
    conv2_channels: int = 64
    fc1_features: int = 128
    num_classes: int = 10
    dropout_rate: float = 0.5
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10
    
    # Data preprocessing
    image_size: int = 28
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081
    
    # Model metadata
    model_name: str = "cnn_mnist"
    display_name: str = "CNN MNIST"
    description: str = "Convolutional Neural Network for MNIST digit classification"
    model_type: str = "cnn"
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_channels": self.input_channels,
            "conv1_channels": self.conv1_channels,
            "conv2_channels": self.conv2_channels,
            "fc1_features": self.fc1_features,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "image_size": self.image_size,
            "normalize_mean": self.normalize_mean,
            "normalize_std": self.normalize_std,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "description": self.description,
            "model_type": self.model_type,
            "version": self.version
        } 