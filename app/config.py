import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class APIConfig:
    """API configuration settings."""
    base_url: str = os.getenv("MODEL_API_URL", "http://model-service:8000")
    retries: int = 3
    timeout: int = 10

@dataclass
class CanvasConfig:
    """Canvas configuration settings."""
    width: int = 280
    height: int = 280
    stroke_width: int = 20
    stroke_color: str = "#FFFFFF"
    bg_color: str = "#000000"
    target_size: Tuple[int, int] = (28, 28)

@dataclass
class AppConfig:
    """Application configuration settings."""
    title: str = "MNIST Digit Classifier"
    page_icon: str = "✍️"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Style paths
    style_path: str = "styles/main.css"
    
    # Component configurations
    api: APIConfig = APIConfig()
    canvas: CanvasConfig = CanvasConfig()

# Create global config instance
config = AppConfig() 