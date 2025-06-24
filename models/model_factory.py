"""
Model factory for managing multiple MNIST model implementations.
"""

import os
from typing import Dict, List, Optional, Type
from base.base_model import BaseModel
from cnn_mnist.model import CNNMNISTClassifier
from transformer1_mnist.model import Transformer1MNISTClassifier
from transformer2_mnist.model import Transformer2MNISTClassifier
from config import get_model_config


class ModelFactory:
    """
    Factory class for creating and managing MNIST model instances.
    """
    
    # Registry of available models
    _models: Dict[str, Type[BaseModel]] = {
        "cnn_mnist": CNNMNISTClassifier,
        "transformer1_mnist": Transformer1MNISTClassifier,
        "transformer2_mnist": Transformer2MNISTClassifier,
    }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List[str]: List of available model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_metadata(cls, model_name: str) -> Optional[Dict]:
        """
        Get metadata for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Optional[Dict]: Model metadata or None if model doesn't exist
        """
        try:
            config = get_model_config(model_name)
            return {
                "name": config.name,
                "display_name": config.display_name,
                "description": config.description,
                "model_type": config.model_type,
                "version": config.version,
                "checkpoint_path": config.checkpoint_path,
                "is_active": config.is_active,
                "config": config.config
            }
        except ValueError:
            return None
    
    @classmethod
    def get_all_model_metadata(cls) -> Dict[str, Dict]:
        """
        Get metadata for all available models.
        
        Returns:
            Dict[str, Dict]: Dictionary of model metadata
        """
        metadata = {}
        for model_name in cls.get_available_models():
            metadata[model_name] = cls.get_model_metadata(model_name)
        return metadata
    
    @classmethod
    def create_model(cls, model_name: str, checkpoint_path: Optional[str] = None) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            checkpoint_path: Optional path to model checkpoint
            
        Returns:
            BaseModel: Model instance
            
        Raises:
            ValueError: If model name is not recognized
        """
        if model_name not in cls._models:
            available_models = ", ".join(cls.get_available_models())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
        
        # Use provided checkpoint path or get from config
        if checkpoint_path is None:
            try:
                config = get_model_config(model_name)
                checkpoint_path = config.checkpoint_path
            except ValueError:
                # Fallback to default path if config not found
                checkpoint_path = f"models/{model_name}/checkpoints/{model_name}.pt"
        
        # Create model instance
        model_class = cls._models[model_name]
        return model_class(checkpoint_path=checkpoint_path)
    
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel], metadata: Dict):
        """
        Register a new model with the factory.
        
        Args:
            model_name: Name of the model
            model_class: Model class that inherits from BaseModel
            metadata: Model metadata dictionary
        """
        cls._models[model_name] = model_class
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """
        Check if a model name is valid.
        
        Args:
            model_name: Name to validate
            
        Returns:
            bool: True if model name is valid, False otherwise
        """
        return model_name in cls._models


# Convenience functions
def get_model(model_name: str, checkpoint_path: Optional[str] = None) -> BaseModel:
    """
    Get a model instance by name.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        BaseModel: Model instance
    """
    return ModelFactory.create_model(model_name, checkpoint_path)


def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List[str]: List of available model names
    """
    return ModelFactory.get_available_models()


def get_model_metadata(model_name: str) -> Optional[Dict]:
    """
    Get metadata for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Optional[Dict]: Model metadata or None if model doesn't exist
    """
    return ModelFactory.get_model_metadata(model_name) 