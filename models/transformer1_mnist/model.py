"""
Transformer1 MNIST model implementation.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional

from base.base_model import BaseModel


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Define layers and parameters
        self.patch_embedding = nn.Linear(patch_size * patch_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x):
        # Patch tokenization
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches_flat = patches.reshape(x.shape[0], self.num_patches, self.patch_size * self.patch_size)
        
        # Linear projection
        embedded_patches = self.patch_embedding(patches_flat)
        
        # Add positional embeddings
        embedded_patches_with_pos = embedded_patches + self.position_embedding
        
        # Add class token
        embedded_patches_with_class = torch.cat([self.class_token.expand(x.shape[0], -1, -1), embedded_patches_with_pos], dim=1)
        
        return embedded_patches_with_class


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        
        # Define the linear layers
        self.W_q = nn.Linear(embed_dim, embed_dim) 
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, residual_stream):
        # Compute Q, K, V
        K = residual_stream @ self.W_k.weight.T
        Q = residual_stream @ self.W_q.weight.T
        V = residual_stream @ self.W_v.weight.T
        
        # Compute attention scores
        A = Q @ K.transpose(-1, -2)
        A = A / (self.embed_dim ** 0.5)  # scale by sqrt(d_k)
        A = torch.softmax(A, dim=-1)
        
        # Apply attention
        H = A @ V
        H = self.W_o(H)
        
        return H


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Define the linear classifier
        self.Linear_classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, H):
        # Extract only the class token (first token)
        class_token_output = H[:, 0, :]
        
        # Apply classifier only to class token
        logits = self.Linear_classifier(class_token_output)
        
        return logits


class Transformer1Model(nn.Module):
    """Simple transformer model with encoder layers and a classification head"""
    def __init__(self, image_size, patch_size, embed_dim, num_layers, num_classes):
        super(Transformer1Model, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # Create multiple encoder layers
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim=embed_dim) for _ in range(num_layers)
        ])
        self.classification_head = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        
        # Pass through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        x = self.classification_head(x)
        return x


class Transformer1MNISTClassifier(BaseModel):
    """
    Transformer1 MNIST classifier wrapper that implements the BaseModel interface.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the Transformer1 MNIST classifier.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        super().__init__("transformer1_mnist", checkpoint_path)
    
    def create_model(self) -> nn.Module:
        """
        Create and return the Transformer1 model architecture.
        
        Returns:
            nn.Module: The Transformer1 model architecture
        """
        return Transformer1Model(
            image_size=28,
            patch_size=7,
            embed_dim=128,
            num_layers=3,
            num_classes=10
        )
    
    def get_preprocessing_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for Transformer1 model.
        
        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) 