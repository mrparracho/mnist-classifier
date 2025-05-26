import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

class DrawingCanvas:
    def __init__(self):
        self.canvas_width = 280
        self.canvas_height = 280
        self.stroke_width = 20
        self.stroke_color = "#FFFFFF"
        self.bg_color = "#000000"
        
    def get_image(self):
        """Get the current image from the canvas and preprocess it for the model."""
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            background_color=self.bg_color,
            width=self.canvas_width,
            height=self.canvas_height,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            # Convert the RGBA image to grayscale
            img_data = canvas_result.image_data
            pil_image = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            pil_image = pil_image.convert('L')
            
            # Resize to 28x28 (MNIST format)
            pil_image = pil_image.resize((28, 28))
            
            # Invert colors (MNIST has white digits on black background)
            pil_image = ImageOps.invert(pil_image)
            
            # Convert to numpy array and normalize
            img_array = np.array(pil_image)
            img_array = img_array.astype('float32') / 255.0
            
            return img_array
        
        return None 