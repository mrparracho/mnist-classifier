"""
Abstract base class for all MNIST model implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms


class BaseModel(ABC):
    """
    Abstract base class defining the common interface for all MNIST models.
    """
    
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name identifier for the model
            checkpoint_path: Path to the model checkpoint file
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def create_model(self) -> nn.Module:
        """
        Create and return the model architecture.
        
        Returns:
            nn.Module: The model architecture
        """
        pass
    
    @abstractmethod
    def get_preprocessing_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for this model.
        
        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        pass
    
    def load_model(self) -> nn.Module:
        """
        Load the model and its trained weights.
        
        Returns:
            nn.Module: The loaded model
        """
        if self.model is None:
            self.model = self.create_model()
            
        if self.checkpoint_path:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # If checkpoint contains model_state_dict key (training format)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    # If checkpoint contains epoch key but no model_state_dict (older format)
                    elif 'epoch' in checkpoint:
                        # Remove non-model keys and use the rest as state_dict
                        state_dict = {k: v for k, v in checkpoint.items() 
                                    if k not in ['epoch', 'optimizer_state_dict', 'accuracy', 'loss']}
                    else:
                        # Assume the entire dict is the state_dict
                        state_dict = checkpoint
                else:
                    # Direct state_dict format
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {self.checkpoint_path}: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        return self.model
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image bytes for model inference.
        
        Args:
            image_bytes: Raw bytes of the uploaded image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            
            # Get model-specific preprocessing
            transform = self.get_preprocessing_transform()
            
            # Apply transformations and add batch dimension
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Image preprocessing error: {e}")
    
    def predict(self, image_bytes: bytes) -> Tuple[int, float, List[float]]:
        """
        Make a prediction on the given image.
        
        Args:
            image_bytes: Raw bytes of the uploaded image
            
        Returns:
            Tuple[int, float, List[float]]: (predicted_class, confidence, probabilities)
        """
        if self.model is None:
            self.load_model()
        
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image_bytes)
            
            # Forward pass
            if self.model is not None:
                logits = self.model(image_tensor)
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
                
                # Convert probabilities to list
                prob_list = probabilities.squeeze().cpu().numpy().tolist()
                
                return int(predicted_class), float(confidence), prob_list
            else:
                raise RuntimeError("Model not loaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        return {
            "name": self.model_name,
            "type": self.__class__.__name__,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "is_loaded": self.model is not None
        } 