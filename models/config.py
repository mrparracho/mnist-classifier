"""
Centralized configuration for all MNIST models.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    display_name: str
    description: str
    model_type: str
    version: str
    checkpoint_path: str
    is_active: bool = True
    config: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Registry for all available models."""
    
    # Default model configurations
    _models: Dict[str, ModelConfig] = {
        "cnn_mnist": ModelConfig(
            name="cnn_mnist",
            display_name="CNN MNIST",
            description="Convolutional Neural Network for MNIST digit classification",
            model_type="cnn",
            version="1.0.0",
            checkpoint_path=os.getenv("CNN_MNIST_CHECKPOINT_PATH", "cnn_mnist/checkpoints/cnn_mnist.pt"),
            config={
                "input_channels": 1,
                "conv1_channels": 32,
                "conv2_channels": 64,
                "fc1_features": 128,
                "num_classes": 10,
                "dropout_rate": 0.5,
                "image_size": 28,
                "normalize_mean": 0.1307,
                "normalize_std": 0.3081
            }
        ),
        "transformer1_mnist": ModelConfig(
            name="transformer1_mnist",
            display_name="Transformer1 MNIST",
            description="Vision Transformer with encoder layers for MNIST classification",
            model_type="transformer",
            version="1.0.0",
            checkpoint_path=os.getenv("TRANSFORMER1_MNIST_CHECKPOINT_PATH", "transformer1_mnist/checkpoints/transformer1_mnist.pt"),
            config={
                "image_size": 28,
                "patch_size": 7,
                "embed_dim": 128,
                "num_layers": 3,
                "num_classes": 10,
                "normalize_mean": 0.1307,
                "normalize_std": 0.3081
            }
        ),
        "transformer2_mnist": ModelConfig(
            name="transformer2_mnist",
            display_name="Transformer2 MNIST",
            description="Vision Transformer with encoder layers and MLP for MNIST classification",
            model_type="transformer",
            version="1.0.0",
            checkpoint_path=os.getenv("TRANSFORMER2_MNIST_CHECKPOINT_PATH", "transformer2_mnist/checkpoints/transformer2_mnist.pt"),
            config={
                "image_size": 28,
                "patch_size": 7,
                "embed_dim": 128,
                "num_layers": 3,
                "num_classes": 10,
                "normalize_mean": 0.1307,
                "normalize_std": 0.3081
            }
        ),
        "encoder_decoder": ModelConfig(
            name="encoder_decoder",
            display_name="Encoder-Decoder",
            description="Encoder-Decoder model for sequence MNIST digit prediction",
            model_type="encoder_decoder",
            version="1.0.0",
            checkpoint_path=os.getenv("ENCODER_DECODER_CHECKPOINT_PATH", "encoder_decoder/checkpoints/encoder_decoder.pt"),
            config={
                "patch_size": 7,
                "encoder_embed_dim": 64,
                "decoder_embed_dim": 64,
                "num_layers": 4,
                "num_heads": 8,
                "dropout": 0.1,
                "normalize_mean": 0.1307,
                "normalize_std": 0.3081,
                "grid_checkpoints": {
                    1: "encoder_decoder/checkpoints/mnist-encoder-decoder-1-varlen.pt",
                    2: "encoder_decoder/checkpoints/mnist-encoder-decoder-2-varlen.pt",
                    3: "encoder_decoder/checkpoints/mnist-encoder-decoder-3-varlen.pt",
                    4: "encoder_decoder/checkpoints/mnist-encoder-decoder-4-varlen.pt"
                }
            }
        ),
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig: Model configuration
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
        
        return cls._models[model_name]
    
    @classmethod
    def get_all_models(cls) -> List[ModelConfig]:
        """
        Get all model configurations.
        
        Returns:
            List[ModelConfig]: List of all model configurations
        """
        return list(cls._models.values())
    
    @classmethod
    def get_active_models(cls) -> List[ModelConfig]:
        """
        Get all active model configurations.
        
        Returns:
            List[ModelConfig]: List of active model configurations
        """
        return [model for model in cls._models.values() if model.is_active]
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """
        Get list of all model names.
        
        Returns:
            List[str]: List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_active_model_names(cls) -> List[str]:
        """
        Get list of active model names.
        
        Returns:
            List[str]: List of active model names
        """
        return [model.name for model in cls._models.values() if model.is_active]
    
    @classmethod
    def register_model(cls, model_config: ModelConfig):
        """
        Register a new model configuration.
        
        Args:
            model_config: Model configuration to register
        """
        cls._models[model_config.name] = model_config
    
    @classmethod
    def unregister_model(cls, model_name: str):
        """
        Unregister a model configuration.
        
        Args:
            model_name: Name of the model to unregister
        """
        if model_name in cls._models:
            del cls._models[model_name]
    
    @classmethod
    def update_model_config(cls, model_name: str, **kwargs):
        """
        Update configuration for a specific model.
        
        Args:
            model_name: Name of the model
            **kwargs: Configuration updates
        """
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_config = cls._models[model_name]
        for key, value in kwargs.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")

    @classmethod
    def get_encoder_decoder_checkpoint_path(cls, grid_size: int) -> str:
        """
        Get the checkpoint path for encoder-decoder model based on grid size.
        
        Args:
            grid_size: Size of the grid (1-4)
            
        Returns:
            str: Checkpoint path for the specific grid size
        """
        config = cls.get_model_config("encoder_decoder")
        if config.config is None:
            raise ValueError("Encoder-decoder model config is not properly configured")
        
        grid_checkpoints = config.config.get("grid_checkpoints", {})
        
        if grid_size not in grid_checkpoints:
            raise ValueError(f"Grid size {grid_size} not supported. Available: {list(grid_checkpoints.keys())}")
        
        return grid_checkpoints[grid_size]


# Convenience functions
def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    return ModelRegistry.get_model_config(model_name)


def get_all_models() -> List[ModelConfig]:
    """Get all model configurations."""
    return ModelRegistry.get_all_models()


def get_active_models() -> List[ModelConfig]:
    """Get all active model configurations."""
    return ModelRegistry.get_active_models()


def get_model_names() -> List[str]:
    """Get list of all model names."""
    return ModelRegistry.get_model_names()


def get_active_model_names() -> List[str]:
    """Get list of active model names."""
    return ModelRegistry.get_active_model_names() 