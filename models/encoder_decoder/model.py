"""
Encoder-Decoder MNIST model implementation.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional, List
import re

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
        # Grid size will be extracted when checkpoint_path is set
        self.grid_size: Optional[int] = None
        # Model configuration will be extracted from checkpoint
        self.model_config: Optional[dict] = None
    
    def _extract_config_from_checkpoint(self) -> dict:
        """
        Extract configuration from the checkpoint file using exact training parameters.
        
        Returns:
            dict: Complete model configuration
        """
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path is required")
        
        # Extract grid size from filename
        match = re.search(r'mnist-encoder-decoder-(\d+)-varlen\.pt', self.checkpoint_path)
        if not match:
            raise ValueError(f"Could not extract grid size from checkpoint path: {self.checkpoint_path}")
        
        grid_size = int(match.group(1))
        if not (1 <= grid_size <= 4):
            raise ValueError(f"Invalid grid size {grid_size} from checkpoint path")
        
        # Use exact training parameters from the logs and training code
        # These are the hardcoded values used during training
        config = {
            'grid_size': grid_size,
            'image_size': grid_size * 28,  # e.g., 2*28=56 for 2x2 grid
            'max_seq_len': (grid_size * grid_size) + 2,  # +2 for start/finish tokens
            'patch_size': 14,  # From training logs: --patch-size: 14
            'encoder_embed_dim': 128,  # From training code: hardcoded value
            'decoder_embed_dim': 128,  # From training code: hardcoded value
            'num_layers': 8,  # From training code: hardcoded value
            'num_heads': 8,  # From training code: hardcoded value
            'dropout': 0.1,  # From training code: hardcoded value
            'normalize_mean': 0.1307,  # MNIST standard
            'normalize_std': 0.3081,   # MNIST standard
        }
        
        print(f"DEBUG: Using exact training parameters:")
        for key, value in config.items():
            print(f"DEBUG:   {key}: {value}")
        
        return config
    
    def _ensure_config(self):
        """Ensure configuration is loaded from checkpoint."""
        if self.model_config is None:
            self.model_config = self._extract_config_from_checkpoint()
            self.grid_size = self.model_config['grid_size']
    
    def create_model(self) -> nn.Module:
        """
        Create and return the Encoder-Decoder model architecture for the specific grid size.
        
        Returns:
            nn.Module: The Encoder-Decoder model architecture
        """
        # Ensure configuration is loaded from checkpoint
        self._ensure_config()
        assert self.model_config is not None  # Type assertion for linter
        
        # Use configuration extracted from checkpoint
        config = self.model_config
        
        print(f"DEBUG: Creating model with extracted config:")
        print(f"DEBUG:   image_size={config['image_size']}, max_seq_len={config['max_seq_len']}")
        print(f"DEBUG:   patch_size={config['patch_size']}, embed_dim={config['encoder_embed_dim']}")
        
        return EncoderDecoder(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            encoder_embed_dim=config['encoder_embed_dim'],
            decoder_embed_dim=config['decoder_embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        )
    
    def get_preprocessing_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for Encoder-Decoder model.
        
        Returns:
            transforms.Compose: The preprocessing pipeline
        """
        # Ensure configuration is loaded from checkpoint
        self._ensure_config()
        assert self.model_config is not None  # Type assertion for linter
        
        # Use configuration extracted from checkpoint
        config = self.model_config
        
        return transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize((config['normalize_mean'],), (config['normalize_std'],))
        ])
    
    def predict_sequence(self, image_bytes: bytes, grid_size: int) -> tuple[List[int], float]:
        """
        Predict a sequence of digits for a given grid size.
        
        Args:
            image_bytes: Image data as bytes
            grid_size: Size of the grid (1-4) - should match the model's grid size
            
        Returns:
            tuple: (predicted_sequence, confidence)
        """
        try:
            # Ensure grid size is set
            self._ensure_config()
            assert self.grid_size is not None  # Type assertion for linter
            
            # Verify grid size matches the model's grid size
            if grid_size != self.grid_size:
                print(f"WARNING: Requested grid size {grid_size} doesn't match model's grid size {self.grid_size}")
                print(f"Using model's grid size {self.grid_size}")
                grid_size = self.grid_size
            
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
            
            # Check if model loaded successfully
            if self.model is None:
                raise RuntimeError("Failed to load model")
            
            # Preprocess the image
            image_tensor = self.preprocess_image(image_bytes)
            
            # Add batch dimension if needed
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device
            device = next(self.model.parameters()).device
            image_tensor = image_tensor.to(device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                # Generate sequence with appropriate max_length based on grid size
                max_length = grid_size * grid_size + 2  # +2 for start and finish tokens
                
                print(f"DEBUG: Generating sequence for grid {grid_size}x{grid_size}, max_length={max_length}")
                print(f"DEBUG: Input image shape: {image_tensor.shape}")
                
                # Generate the sequence
                output_sequences = self.model(
                    image_tensor, 
                    target_sequence=None, 
                    max_length=max_length,
                    temperature=1.0
                )
                
                # Process the output sequence
                sequence = output_sequences[0]  # Take first batch item
                
                print(f"DEBUG: Raw sequence tokens: {sequence.tolist()}")
                print(f"DEBUG: Start token: 10")
                print(f"DEBUG: Finish token: 11")
                print(f"DEBUG: Pad token: 12")
                
                # Convert tokens to digits (remove start token, stop at finish token)
                predicted_sequence = []
                start_token = 10  # <start> token
                finish_token = 11  # <finish> token
                pad_token = 12     # <pad> token
                
                for token in sequence:
                    token_id = token.item()
                    if token_id == start_token:
                        print(f"DEBUG: Skipping start token {token_id}")
                        continue  # Skip start token
                    elif token_id == finish_token:
                        print(f"DEBUG: Found finish token {token_id}, stopping")
                        break  # Stop at finish token
                    elif token_id == pad_token:
                        print(f"DEBUG: Skipping pad token {token_id}")
                        continue  # Skip pad tokens
                    elif 0 <= token_id <= 9:
                        print(f"DEBUG: Adding digit token {token_id}")
                        predicted_sequence.append(token_id)
                    else:
                        print(f"DEBUG: Skipping invalid token {token_id}")
                        # Invalid token, skip
                        continue
                
                print(f"DEBUG: Predicted sequence before padding: {predicted_sequence}")
                
                # Ensure we have the right number of digits
                expected_length = grid_size * grid_size
                if len(predicted_sequence) < expected_length:
                    # Pad with zeros if we don't have enough digits
                    padding_needed = expected_length - len(predicted_sequence)
                    print(f"DEBUG: Padding with {padding_needed} zeros")
                    predicted_sequence.extend([0] * padding_needed)
                elif len(predicted_sequence) > expected_length:
                    # Truncate if we have too many digits
                    print(f"DEBUG: Truncating from {len(predicted_sequence)} to {expected_length}")
                    predicted_sequence = predicted_sequence[:expected_length]
                
                print(f"DEBUG: Final predicted sequence: {predicted_sequence}")
                
                # Calculate confidence based on model's prediction quality
                confidence = 0.0
                if len(predicted_sequence) > 0:
                    # Simple heuristic: longer sequences that complete properly have higher confidence
                    if len(predicted_sequence) == expected_length:
                        confidence = 0.8  # Good confidence for complete sequences
                    else:
                        confidence = 0.4  # Lower confidence for incomplete sequences
                else:
                    confidence = 0.1  # Very low confidence for empty sequences
                
                return predicted_sequence, confidence
                
        except Exception as e:
            print(f"DEBUG: Exception in predict_sequence: {e}")
            import traceback
            traceback.print_exc()
            # Return a default sequence in case of error
            default_sequence = [0] * (grid_size * grid_size)
            return default_sequence, 0.0

    def load_model(self) -> nn.Module:
        """
        Load the model and its trained weights.
        
        Returns:
            nn.Module: The loaded model
        """
        if self.model is None:
            self.model = self.create_model()
            
        if self.checkpoint_path:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # If checkpoint contains model_state_dict key (training format)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    # If checkpoint contains epoch key but no model_state_dict (older format)
                    elif 'epoch' in checkpoint:
                        # Remove non-model keys and use the rest as state_dict
                        state_dict = {k: v for k, v in checkpoint.items() 
                                    if k not in ['epoch', 'optimizer_state_dict', 'accuracy', 'loss']}
                    else:
                        # Assume the entire dict is the state_dict
                        state_dict = checkpoint
                else:
                    # Direct state_dict format
                    state_dict = checkpoint
                
                # Try to load the state dict with strict=False to handle partial matches
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    print(f"DEBUG: Successfully loaded checkpoint with strict=True")
                except RuntimeError as e:
                    print(f"DEBUG: Strict loading failed: {e}")
                    print(f"DEBUG: Attempting partial loading...")
                    
                    # Try partial loading
                    try:
                        # Load what we can and ignore the rest
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        print(f"DEBUG: Partial loading successful")
                        print(f"DEBUG: Missing keys: {len(missing_keys)}")
                        print(f"DEBUG: Unexpected keys: {len(unexpected_keys)}")
                        
                        if len(missing_keys) > 0:
                            print(f"DEBUG: Missing keys (first 5): {missing_keys[:5]}")
                        if len(unexpected_keys) > 0:
                            print(f"DEBUG: Unexpected keys (first 5): {unexpected_keys[:5]}")
                            
                    except Exception as partial_e:
                        print(f"DEBUG: Partial loading also failed: {partial_e}")
                        print(f"DEBUG: Using model with random weights")
                        
            except Exception as e:
                print(f"DEBUG: Checkpoint loading failed: {e}")
                print(f"DEBUG: Using model with random weights")
        
        self.model.to(self.device)
        self.model.eval()
        return self.model