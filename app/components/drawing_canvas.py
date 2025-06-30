import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np

class DrawingCanvas:
    def __init__(self, grid_size=1):
        self.canvas_width = 280
        self.canvas_height = 280
        # MNIST stroke width calculation:
        # MNIST digits have ~2-3 pixel stroke width in 28x28 images
        # Canvas scale factor: 280/28 = 10
        # Optimal stroke width: 2.5 * 10 = 25 pixels
        self.base_stroke_width = 25  # Matches MNIST training data characteristics
        self.stroke_color = "#FFFFFF"
        self.bg_color = "#000000"
        self.grid_size = grid_size
        
    def get_stroke_width(self):
        """Calculate stroke width proportionally based on grid size to match MNIST training data."""
        # MNIST stroke width is ~2-3 pixels in 28x28 images
        # For larger grids, reduce stroke width proportionally to maintain MNIST-like characteristics
        if self.grid_size == 1:
            return self.base_stroke_width  # 25px for 1x1 grid
        elif self.grid_size == 2:
            return int(self.base_stroke_width * 0.6)  # ~15px for 2x2 grid
        elif self.grid_size == 3:
            return int(self.base_stroke_width * 0.4)  # ~10px for 3x3 grid
        elif self.grid_size == 4:
            return int(self.base_stroke_width * 0.3)  # ~8px for 4x4 grid
        else:
            return max(5, int(self.base_stroke_width / self.grid_size))  # Fallback
        
    def get_image(self):
        """Get the current image from the canvas and preprocess it for the model."""
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=self.get_stroke_width(),
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