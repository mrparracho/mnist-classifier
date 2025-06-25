"""
Transformer2 MNIST model implementation.
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


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Define the linear layers
        self.W_q = nn.Linear(embed_dim, embed_dim) 
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        


    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        K = x @ self.W_k.weight.T
        Q = x @ self.W_q.weight.T
        V = x @ self.W_v.weight.T
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        A = Q @ K.transpose(-1, -2)
        A = A / (self.head_dim ** 0.5)  # scale by sqrt(d_k)
        A = torch.softmax(A, dim=-1)
        
        # Apply attention
        H = A @ V
        
        # Reshape back
        H = H.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        H = self.W_o(H)
        
        return H


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        
        # Define the linear layers
        self.LayerNorm1 = nn.LayerNorm(embed_dim)
        self.mlp_up = nn.Linear(embed_dim, 4*embed_dim)
        self.mlp_down = nn.Linear(4*embed_dim, embed_dim)
        self.LayerNorm2 = nn.LayerNorm(embed_dim)

        # Add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual_stream):
        # MLP with residual connections
        x = self.dropout(x)
        x = x + residual_stream
        x = self.LayerNorm1(x)
        x = self.mlp_up(x)
        x = torch.relu(x)
        x = residual_stream + self.mlp_down(x)  # residual connection
        x = self.LayerNorm2(x)

        return x


class TransformerBlock(nn.Module):
    """Complete Transformer block with attention, MLP, residual connections, and normalization"""
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim

        # Add dropout
        self.dropout = nn.Dropout(dropout)

        # Attention layer
        self.attention = Attention(embed_dim, num_heads)
        
        # MLP layer
        self.mlp = MLP(embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # dropout
        x = self.dropout(x)

        # Add residual stream
        residual_stream = x

        # Layer normalization
        x = self.norm1(x)

        # Attention with residual connection and normalization  
        x = self.attention(x)

        # MLP with residual connection and normalization
        x = self.mlp(x, residual_stream)
        x = self.norm2(x)
        
        return x


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


class Transformer2Model(nn.Module):
    """Transformer model with Multiple Transformer Blocks and a Classification head"""
    def __init__(self, image_size, patch_size, embed_dim, num_layers, num_classes, num_heads=1, dropout=0.1):
        super(Transformer2Model, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.classification_head = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        
        # Pass through each transformer block
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Pass through classification head
        x = self.classification_head(x)
        return x


class Transformer2MNISTClassifier(BaseModel):
    """
    Transformer2 MNIST classifier wrapper that implements the BaseModel interface.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the Transformer2 MNIST classifier.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        super().__init__("transformer2_mnist", checkpoint_path)
    
    def create_model(self) -> nn.Module:
        """
        Create and return the Transformer2 model architecture.
        
        Returns:
            nn.Module: The Transformer2 model architecture
        """
        return Transformer2Model(
            image_size=28,
            patch_size=7,
            embed_dim=128,
            num_layers=12, # 12 encoder layers
            num_classes=10,
            num_heads=12,  # 12 attention heads
            dropout=0.1
        )
    
    def get_preprocessing_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for Transformer2 model.
        
        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])