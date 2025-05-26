import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 2 Convolutional layers with max pooling
    - 2 Fully connected layers
    - Output layer with 10 classes (digits 0-9)
    """
    
    def __init__(self):
        super(MNISTModel, self).__init__()
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
    
    def predict(self, x):
        """
        Make a prediction with probabilities
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        with torch.no_grad():
            # Forward pass
            logits = self(x)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted class (digit with highest probability)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            # Get the confidence (probability of the predicted class)
            confidence = torch.max(probabilities, dim=1)[0]
            
            return predicted_class, confidence


def get_model():
    """Factory function to create and initialize the model."""
    model = MNISTModel()
    return model
