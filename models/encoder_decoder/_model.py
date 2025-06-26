"""
Encoder-Decoder MNIST model implementation with Multi-Head Latent Attention (MLA) and RoPE.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from typing import Optional

from base.base_model import BaseModel
from .mla_attention import MLAEncoderSelfAttention, MLADecoderSelfAttention, MLADecoderCrossAttention


# Old RoPE implementation removed - MLA handles its own positional encoding


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, encoder_embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Define layers and parameters
        self.patch_embedding = nn.Linear(patch_size * patch_size, encoder_embed_dim)
        # Remove position embedding - RoPE will handle positions
        self.class_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
    
    def forward(self, x):
        # Patch tokenization
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches_flat = patches.reshape(x.shape[0], self.num_patches, self.patch_size * self.patch_size)
        
        # Linear projection
        embedded_patches = self.patch_embedding(patches_flat)
        
        # Add class token (no position embeddings needed with RoPE)
        embedded_patches_with_class = torch.cat([self.class_token.expand(x.shape[0], -1, -1), embedded_patches], dim=1)
        
        return embedded_patches_with_class

class DecoderEmbedding(nn.Module):
    def __init__(self, decoder_embed_dim, max_seq_len):
        super().__init__()
        # Vocabulary: 0-9 (digits) + <start> + <finish> + <pad> = 13 tokens
        self.vocab_size = 13
        self.token_embedding = nn.Embedding(self.vocab_size, decoder_embed_dim)
        # Remove position embedding - RoPE will handle positions
        self.decoder_embed_dim = decoder_embed_dim
        
        # Define token indices
        self.start_token = 10  # <start>
        self.finish_token = 11  # <finish>
        self.pad_token = 12     # <pad>
        
    def forward(self, x):
        # x is a tensor of integers [batch_size, seq_len]
        # Only token embeddings - no position embeddings with RoPE
        return self.token_embedding(x)
    
    def get_start_token(self, batch_size, device):
        """Get start token for a batch"""
        return torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=device)
    
    def get_pad_token(self, batch_size, seq_len, device):
        """Get padding tokens for a batch"""
        return torch.full((batch_size, seq_len), self.pad_token, dtype=torch.long, device=device)

class OutputProjection(nn.Module):
    def __init__(self, decoder_embed_dim):
        super().__init__()
        # Output vocabulary: 0-9 (digits) + <start> + <finish> + <pad> = 13 tokens
        self.vocab_size = 13
        self.projection = nn.Linear(decoder_embed_dim, self.vocab_size)
        
    def forward(self, x):
        return self.projection(x)


class EncoderSelfAttention(nn.Module):
    """Wrapper for MLA Encoder Self-Attention to maintain interface compatibility"""
    def __init__(self, encoder_embed_dim, num_heads=1, dropout=0.1):
        super(EncoderSelfAttention, self).__init__()
        
        # Calculate appropriate compression ratios based on model size
        kv_lora_rank = max(encoder_embed_dim // 2, 64)  # 2x compression for encoder
        qk_rope_head_dim = min(32, encoder_embed_dim // num_heads // 2)  # Adaptive RoPE dim
        
        # Use MLA implementation
        self.mla_attention = MLAEncoderSelfAttention(
            embed_dim=encoder_embed_dim,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dropout=dropout
        )

    def forward(self, x):
        return self.mla_attention(x)

class DecoderAttention(nn.Module):
    """Wrapper for MLA Decoder Cross-Attention to maintain interface compatibility"""
    def __init__(self, encoder_output_embed_dim, decoder_embed_dim, num_heads=1, dropout=0.1):
        super(DecoderAttention, self).__init__()
        
        # Aggressive compression for cross-attention (encoder is static)
        kv_lora_rank = max(encoder_output_embed_dim // 4, 32)  # 4x compression
        
        # Use MLA implementation
        self.mla_attention = MLADecoderCrossAttention(
            encoder_embed_dim=encoder_output_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            dropout=dropout
        )

    def forward(self, x, encoder_output):
        return self.mla_attention(x, encoder_output)


class DecoderSelfAttention(nn.Module):
    """Wrapper for MLA Decoder Self-Attention to maintain interface compatibility"""
    def __init__(self, decoder_embed_dim, num_heads=1, dropout=0.1):
        super(DecoderSelfAttention, self).__init__()
        
        # Aggressive compression for decoder self-attention (highest memory usage during generation)
        kv_lora_rank = max(decoder_embed_dim // 3, 42)  # 3x compression
        qk_rope_head_dim = min(32, decoder_embed_dim // num_heads // 2)  # Adaptive RoPE dim
        
        # Use MLA implementation
        self.mla_attention = MLADecoderSelfAttention(
            embed_dim=decoder_embed_dim,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dropout=dropout
        )

    def forward(self, x, mask=None):
        return self.mla_attention(x, mask)


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        
        self.mlp_up = nn.Linear(embed_dim, 4*embed_dim)
        self.mlp_down = nn.Linear(4*embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Better than ReLU

    def forward(self, x):
        x = self.mlp_up(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.mlp_down(x)
        return x


class EncoderBlock(nn.Module):
    """Complete Encoder block with attention, MLP, residual connections, and normalization"""
    def __init__(self, encoder_embed_dim, num_heads=1, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.encoder_embed_dim = encoder_embed_dim

        # Attention layer
        self.attention = EncoderSelfAttention(encoder_embed_dim, num_heads)
        
        # MLP layer
        self.mlp = MLP(encoder_embed_dim, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(encoder_embed_dim)
        self.norm2 = nn.LayerNorm(encoder_embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention with residual connection and normalization
        residual = x
        x = self.norm1(x)
        attn_output = self.attention(x)
        x = residual + self.dropout(attn_output)
        
        # MLP with residual connection and normalization
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        
        return x


class DecoderBlock(nn.Module):
    """Complete Decoder block with self-attention, cross-attention, MLP, residual connections, and normalization"""
    def __init__(self, encoder_output_embed_dim, decoder_embed_dim, num_heads=1, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_output_embed_dim = encoder_output_embed_dim

        # Self-attention layer
        self.self_attention = DecoderSelfAttention(decoder_embed_dim, num_heads, dropout)
        
        # Cross-attention layer
        self.cross_attention = DecoderAttention(encoder_output_embed_dim=encoder_output_embed_dim, decoder_embed_dim=decoder_embed_dim, num_heads=num_heads, dropout=dropout)
        
        # MLP layer
        self.mlp = MLP(decoder_embed_dim, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(decoder_embed_dim)
        self.norm2 = nn.LayerNorm(decoder_embed_dim)
        self.norm3 = nn.LayerNorm(decoder_embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):
        # Self-attention with residual connection and normalization
        residual = x
        x = self.norm1(x)
        self_attn_output = self.self_attention(x, mask)
        x = residual + self.dropout(self_attn_output)
        
        # Cross-attention with residual connection and normalization
        residual = x
        x = self.norm2(x)
        cross_attn_output = self.cross_attention(x, encoder_output)
        x = residual + self.dropout(cross_attn_output)
        
        # MLP with residual connection and normalization
        residual = x
        x = self.norm3(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, image_size, patch_size, encoder_embed_dim, decoder_embed_dim, num_layers, num_heads=1, dropout=0.1, max_seq_len=102):
        super(EncoderDecoder, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, encoder_embed_dim=encoder_embed_dim)
        self.decoder_embedding = DecoderEmbedding(decoder_embed_dim=decoder_embed_dim, max_seq_len=max_seq_len)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(encoder_embed_dim=encoder_embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(encoder_output_embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = OutputProjection(decoder_embed_dim=decoder_embed_dim)

    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for positions that can attend to each other

    def forward(self, images, target_sequence=None, max_length=20, temperature=1.0):
        # Encode images
        encoded = self.patch_embedding(images)
        for encoder_block in self.encoder_blocks:
            encoded = encoder_block(encoded)
        
        # Decode sequence
        if target_sequence is not None:
            # Training mode
            decoded = self.decoder_embedding(target_sequence)
            
            # Create causal mask for training
            seq_len = target_sequence.size(1)
            mask = self.create_causal_mask(seq_len, target_sequence.device)
            
            for decoder_block in self.decoder_blocks:
                decoded = decoder_block(decoded, encoded, mask)
            output = self.output_projection(decoded)
            return output
        else:
            # Inference mode - autoregressive generation
            batch_size = images.size(0)
            device = images.device
            
            # Start with start token
            current_sequence = self.decoder_embedding.get_start_token(batch_size, device)
            
            # Track which sequences have finished
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for step in range(max_length):
                # Process full sequence
                decoded = self.decoder_embedding(current_sequence)
                
                # Create causal mask for current sequence length
                seq_len = current_sequence.size(1)
                mask = self.create_causal_mask(seq_len, device)
                
                # Pass through decoder blocks
                for decoder_block in self.decoder_blocks:
                    decoded = decoder_block(decoded, encoded, mask)
                
                logits = self.output_projection(decoded)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature scaling for better sampling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Get next token - use argmax for now, but could add sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # Add tokens to sequences that haven't finished
                # For finished sequences, keep adding pad tokens
                for i in range(batch_size):
                    if finished_sequences[i]:
                        next_token[i] = self.decoder_embedding.pad_token
                
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
                
                # Update finished sequences - a sequence is finished when it generates a finish token
                new_finishes = (next_token.squeeze(-1) == self.decoder_embedding.finish_token)
                finished_sequences = finished_sequences | new_finishes
                
                # Early stopping: if all sequences are finished or we've generated enough digits
                if finished_sequences.all():
                    break
                    
                # Also stop if we've generated too many digits (safety check)
                digit_count = 0
                for token in current_sequence[0]:  # Check first sequence as example
                    if 0 <= token <= 9:
                        digit_count += 1
                if digit_count >= 8:  # Don't generate more than 8 digits
                    break
            
            return current_sequence


class EncoderDecoderMNISTClassifier(BaseModel):
    """
    Encoder-Decoder wrapper that implements the BaseModel interface.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the Encoder-Decoder.
        
        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        super().__init__("encoder-decoder-mnist", checkpoint_path)
    
    def create_model(self) -> nn.Module:
        """
        Create and return the Encoder-Decoder model architecture.
        
        Returns:
            nn.Module: The Encoder-Decoder model architecture
        """
        return EncoderDecoder(
            image_size=28,
            patch_size=7,
            encoder_embed_dim=64,
            decoder_embed_dim=64,
            num_layers=4, # 4 encoder layers
            num_heads=8,  # 8 attention heads
            dropout=0.1
        )
    
    def get_preprocessing_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for Encoder-Decoder model.
        
        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])