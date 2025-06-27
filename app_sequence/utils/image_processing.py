import io
import numpy as np
from PIL import Image
import streamlit as st
from typing import Tuple, Optional

def preprocess_canvas_image(image_data: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> bytes:
    """
    Preprocess canvas image data for model input.
    
    Args:
        image_data: Canvas image data as numpy array
        target_size: Target size for the image (width, height). For correct model input, use (grid_size*28, grid_size*28).
        
    Returns:
        Preprocessed image as bytes
    """
    try:
        # Use RGB channels, convert to grayscale
        if image_data.shape[2] >= 3:
            rgb_data = image_data[:, :, :3]
            gray_data = np.dot(rgb_data, [0.299, 0.587, 0.114])
        else:
            gray_data = image_data[:, :, 0]

        # No inversion, no normalization, just scale to 0-255
        gray_data = np.clip(gray_data, 0, 1)
        gray_data = (gray_data * 255).astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(gray_data, mode='L')
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

        # Debug: Save intermediate image for inspection
        debug_path = "/tmp/debug_preprocessed.png"
        pil_image.save(debug_path)
        print(f"DEBUG: Saved intermediate image to {debug_path}")
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def validate_canvas_image(image_data: np.ndarray) -> Optional[str]:
    """
    Validate canvas image data.
    
    Args:
        image_data: Canvas image data as numpy array
        
    Returns:
        Error message if validation fails, None if valid
    """
    if image_data is None:
        return "No image data available"
    
    # Check if image has content (not just background)
    if image_data.shape[2] == 4:  # RGBA
        alpha_channel = image_data[:, :, 3]
        if np.max(alpha_channel) < 10:  # Very low alpha values
            return "Please draw something on the canvas"
    else:
        # For RGB, check if there's any non-zero content
        if np.max(image_data) < 10:
            return "Please draw something on the canvas"
    
    return None 