"""
Encoder-Decoder MNIST model implementation.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional

from base.base_model import BaseModel


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, encoder_embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Define layers and parameters
        self.patch_embedding = nn.Linear(patch_size * patch_size, encoder_embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches, encoder_embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
    
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

class DecoderEmbedding(nn.Module):
    def __init__(self, decoder_embed_dim, max_seq_len):
        super().__init__()
        # Vocabulary: 0-9 (digits) + <start> + <finish> + <pad> = 13 tokens
        self.vocab_size = 13
        self.token_embedding = nn.Embedding(self.vocab_size, decoder_embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_seq_len, decoder_embed_dim))
        self.decoder_embed_dim = decoder_embed_dim
        
        # Define token indices
        self.start_token = 10  # <start>
        self.finish_token = 11  # <finish>
        self.pad_token = 12     # <pad>
        
    def forward(self, x):
        # x is a tensor of integers [batch_size, seq_len]
        seq_len = x.size(1)
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding[:seq_len, :]
        return token_embeddings + position_embeddings
    
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
    def __init__(self, encoder_embed_dim, num_heads=1, dropout=0.1):
        super(EncoderSelfAttention, self).__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.num_heads = num_heads
        self.head_dim = encoder_embed_dim // num_heads
        
        # Ensure encoder_embed_dim is divisible by num_heads
        assert encoder_embed_dim % num_heads == 0, "encoder_embed_dim must be divisible by num_heads"
        
        # Define the linear layers
        self.W_q = nn.Linear(encoder_embed_dim, encoder_embed_dim) 
        self.W_k = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.W_v = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.W_o = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, encoder_embed_dim = x.shape

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
        A = self.dropout(A)
        
        # Apply attention
        H = A @ V
        
        # Reshape back
        H = H.transpose(1, 2).contiguous().view(batch_size, seq_len, encoder_embed_dim)
        H = self.W_o(H)
        
        return H

class DecoderAttention(nn.Module):
    def __init__(self, encoder_output_embed_dim, decoder_embed_dim, num_heads=1, dropout=0.1):
        super(DecoderAttention, self).__init__()
        self.encoder_output_embed_dim = encoder_output_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_heads = num_heads
        self.head_dim = decoder_embed_dim // num_heads
        
        # Ensure decoder_embed_dim is divisible by num_heads
        assert decoder_embed_dim % num_heads == 0, "decoder_embed_dim must be divisible by num_heads"
        
        # Define the linear layers
        self.W_q = nn.Linear(decoder_embed_dim, decoder_embed_dim) 
        self.W_k = nn.Linear(encoder_output_embed_dim, decoder_embed_dim)  # Project to decoder dim
        self.W_v = nn.Linear(encoder_output_embed_dim, decoder_embed_dim)  # Project to decoder dim
        self.W_o = nn.Linear(decoder_embed_dim, decoder_embed_dim)  # Output to decoder dim
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        batch_size, seq_len, decoder_embed_dim = x.shape
        encoder_seq_len = encoder_output.shape[1]

        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, encoder_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, encoder_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        A = Q @ K.transpose(-1, -2)
        A = A / (self.head_dim ** 0.5)  # scale by sqrt(d_k)
        A = torch.softmax(A, dim=-1)
        A = self.dropout(A)
        
        # Apply attention
        H = A @ V
        
        # Reshape back
        H = H.transpose(1, 2).contiguous().view(batch_size, seq_len, decoder_embed_dim)
        H = self.W_o(H)
        
        return H


class DecoderSelfAttention(nn.Module):
    def __init__(self, decoder_embed_dim, num_heads=1, dropout=0.1):
        super(DecoderSelfAttention, self).__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.num_heads = num_heads
        self.head_dim = decoder_embed_dim // num_heads
        
        # Ensure decoder_embed_dim is divisible by num_heads
        assert decoder_embed_dim % num_heads == 0, "decoder_embed_dim must be divisible by num_heads"
        
        # Define the linear layers
        self.W_q = nn.Linear(decoder_embed_dim, decoder_embed_dim) 
        self.W_k = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.W_v = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.W_o = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, decoder_embed_dim = x.shape

        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        A = Q @ K.transpose(-1, -2)
        A = A / (self.head_dim ** 0.5)  # scale by sqrt(d_k)
        
        # Apply causal mask for autoregressive generation
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        
        A = torch.softmax(A, dim=-1)
        A = self.dropout(A)
        
        # Apply attention
        H = A @ V
        
        # Reshape back
        H = H.transpose(1, 2).contiguous().view(batch_size, seq_len, decoder_embed_dim)
        H = self.W_o(H)
        
        return H


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
            # Inference mode - autoregressive generation with improved stopping
            batch_size = images.size(0)
            device = images.device
            
            # Start with start token
            current_sequence = self.decoder_embedding.get_start_token(batch_size, device)
            
            # Track which sequences have finished and their final lengths
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
            final_sequences = []
            
            for step in range(max_length):
                decoded = self.decoder_embedding(current_sequence)
                
                # Create causal mask for current sequence length
                seq_len = current_sequence.size(1)
                mask = self.create_causal_mask(seq_len, device)
                
                for decoder_block in self.decoder_blocks:
                    decoded = decoder_block(decoded, encoded, mask)
                logits = self.output_projection(decoded)
                
                # Apply temperature scaling for better sampling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get next token - use argmax for now, but could add sampling
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # Only add tokens to sequences that haven't finished
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
                
                # Update finished sequences - a sequence is finished when it generates a finish token
                new_finishes = (next_token.squeeze(-1) == self.decoder_embedding.finish_token)
                finished_sequences = finished_sequences | new_finishes
                
                # Early stopping: if all sequences are finished
                if finished_sequences.all():
                    break
            
            # FIXED: Return sequences without padding - trim each sequence to its finish token
            result_sequences = []
            for i in range(batch_size):
                seq = current_sequence[i]
                # Find finish token position
                finish_positions = (seq == self.decoder_embedding.finish_token).nonzero(as_tuple=True)[0]
                if len(finish_positions) > 0:
                    # Include finish token, trim everything after
                    finish_pos = finish_positions[0].item()
                    trimmed_seq = seq[:finish_pos + 1]
                else:
                    # No finish token found, use full sequence
                    trimmed_seq = seq
                result_sequences.append(trimmed_seq)
            
            # Pad result sequences to same length for batch processing (minimal padding)
            max_result_len = max(len(seq) for seq in result_sequences)
            padded_results = []
            for seq in result_sequences:
                if len(seq) < max_result_len:
                    padding = torch.full((max_result_len - len(seq),), self.decoder_embedding.pad_token, 
                                       dtype=seq.dtype, device=seq.device)
                    padded_seq = torch.cat([seq, padding])
                else:
                    padded_seq = seq
                padded_results.append(padded_seq)
            
            return torch.stack(padded_results)


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
            num_layers=2, # 4 encoder layers
            num_heads=2,  # 8 attention heads
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