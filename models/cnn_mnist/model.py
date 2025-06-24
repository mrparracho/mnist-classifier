"""
CNN MNIST model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional

from base.base_model import BaseModel


class CNNMNISTModel(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 2 Convolutional layers with max pooling
    - 2 Fully connected layers
    - Output layer with 10 classes (digits 0-9)
    """
    
    def __init__(self):
        super(CNNMNISTModel, self).__init__()
        # First convolutional layer
        # Input: 1x28x28, Output: 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Second convolutional layer
        # Input: 32 feature maps, Output: 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First fully connected layer
        # After two 2x2 max pooling operations, the image size is reduced to 7x7
        # Input: 64 feature maps of size 7x7, Output: 128 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # Second fully connected layer
        # Input: 128 features, Output: 10 classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First convolution block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolution block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        
        # First fully connected layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer
        x = self.fc2(x)
        
        return x


class CNNMNISTClassifier(BaseModel):
    """
    CNN MNIST classifier wrapper that implements the BaseModel interface.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the CNN MNIST classifier.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        super().__init__("cnn_mnist", checkpoint_path)
    
    def create_model(self) -> nn.Module:
        """
        Create and return the CNN model architecture.
        
        Returns:
            nn.Module: The CNN model architecture
        """
        return CNNMNISTModel()
    
    def get_preprocessing_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for CNN model.
        
        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) 