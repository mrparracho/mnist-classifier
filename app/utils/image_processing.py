import io
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def preprocess_canvas_image(image_data: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> bytes:
    """Process canvas image data for model input.
    
    Args:
        image_data: Numpy array containing RGBA image data from canvas.
        target_size: Target size for the processed image (width, height).
        
    Returns:
        Processed image as bytes in PNG format.
    """
    # Convert the RGBA image to grayscale
    pil_image = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    pil_image = pil_image.convert('L')
    
    # Resize to target size (MNIST format)
    pil_image = pil_image.resize(target_size)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def validate_canvas_image(image_data: np.ndarray) -> Optional[str]:
    """Validate canvas image data.
    
    Args:
        image_data: Numpy array containing image data from canvas.
        
    Returns:
        Error message if validation fails, None otherwise.
    """
    if image_data is None:
        return "No image data provided"
    
    if not isinstance(image_data, np.ndarray):
        return "Invalid image data type"
    
    if len(image_data.shape) != 3 or image_data.shape[2] != 4:
        return "Invalid image format - expected RGBA"
    
    # Check if the image is empty (all transparent or black)
    alpha = image_data[:, :, 3]
    if np.all(alpha == 0):
        return "Empty canvas - please draw a digit"
    
    return None

def get_image_center_of_mass(image_data: np.ndarray) -> Tuple[float, float]:
    """Calculate the center of mass of the drawn digit.
    
    Args:
        image_data: Numpy array containing RGBA image data.
        
    Returns:
        Tuple of (x, y) coordinates of the center of mass.
    """
    # Use alpha channel to find drawn pixels
    alpha = image_data[:, :, 3]
    
    # Get coordinates of non-zero (drawn) pixels
    y_coords, x_coords = np.nonzero(alpha)
    
    # Calculate center of mass
    if len(x_coords) > 0 and len(y_coords) > 0:
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        return center_x, center_y
    
    # Return image center if no drawn pixels found
    height, width = image_data.shape[:2]
    return width / 2, height / 2 