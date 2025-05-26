import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image):
    """
    Preprocess the image for model inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28 if necessary
        if image.size != (28, 28):
            image = image.resize((28, 28))
        
        # Define transformations (same as training)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image format: {e}"
        ) 