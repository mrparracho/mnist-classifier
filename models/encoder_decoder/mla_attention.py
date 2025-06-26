#!/usr/bin/env python3
"""
Multi-Head Latent Attention (MLA) implementation for Encoder-Decoder MNIST model.
Based on DeepSeek-V2 paper: https://arxiv.org/abs/2405.04434

This implementation provides memory-efficient attention through low-rank compression
of Key-Value matrices while maintaining or improving model performance.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class MLAPositionalEmbedding(nn.Module):
    """RoPE-compatible positional embedding for MLA with decoupled encoding"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(-2)
            
        # Create position indices
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Compute frequencies for each position
        inv_freq = torch.as_tensor(self.inv_freq, device=t.device)
        freqs = torch.outer(t, inv_freq)  # [seq_len, dim//2]
        
        # Create rotation matrix components
        cos_freqs = torch.cos(freqs)  # [seq_len, dim//2]
        sin_freqs = torch.sin(freqs)  # [seq_len, dim//2]
        
        return cos_freqs, sin_freqs


def apply_mla_rope(x, cos_freqs, sin_freqs):
    """Apply RoPE to input tensor x for MLA"""
    # Handle different input shapes
    if x.dim() == 4 and x.size(1) != x.size(-1):  # [batch, heads, seq_len, dim]
        x = x.transpose(1, 2)  # Convert to [batch, seq_len, heads, dim]
        transposed = True
    else:
        transposed = False
    
    # Split into even/odd dimensions
    x1 = x[..., ::2]   # Even dimensions
    x2 = x[..., 1::2]  # Odd dimensions
    
    # Expand cos/sin freqs to match x dimensions
    cos_freqs = cos_freqs.unsqueeze(0).unsqueeze(-2)  # [1, seq_len, 1, dim//2]
    sin_freqs = sin_freqs.unsqueeze(0).unsqueeze(-2)  # [1, seq_len, 1, dim//2]
    
    # Apply rotation
    rotated_x1 = x1 * cos_freqs - x2 * sin_freqs
    rotated_x2 = x1 * sin_freqs + x2 * cos_freqs
    
    # Interleave back
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    
    # Restore original shape if needed
    if transposed:
        rotated_x = rotated_x.transpose(1, 2)
    
    return rotated_x


class MLAEncoderSelfAttention(nn.Module):
    """Multi-Head Latent Attention for Encoder Self-Attention"""
    
    def __init__(self, embed_dim, num_heads=16, kv_lora_rank=None, q_lora_rank=None, 
                 qk_rope_head_dim=32, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # MLA compression dimensions
        if kv_lora_rank is None:
            # Conservative compression for encoder (2x compression)
            self.kv_lora_rank = max(embed_dim // 2, 64)
        else:
            self.kv_lora_rank = kv_lora_rank
            
        if q_lora_rank is None:
            # Lighter compression for Q (1.5x compression)
            self.q_lora_rank = max(embed_dim * 2 // 3, 96)
        else:
            self.q_lora_rank = q_lora_rank
        
        # RoPE configuration
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = self.head_dim - qk_rope_head_dim
        
        assert self.qk_nope_head_dim > 0, f"RoPE dim {qk_rope_head_dim} too large for head dim {self.head_dim}"
        
        # Query projection layers (low-rank)
        self.q_a_proj = nn.Linear(embed_dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, num_heads * self.head_dim, bias=False)
        
        # KV joint compression with decoupled RoPE
        self.kv_a_proj_with_rope = nn.Linear(
            embed_dim, 
            self.kv_lora_rank + qk_rope_head_dim,  # Include RoPE head
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        
        # KV decompression (keys and values together)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.head_dim),  # K_nope + V
            bias=False
        )
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        # RoPE and dropout
        self.rope = MLAPositionalEmbedding(qk_rope_head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Query projection with low-rank factorization
        q_compressed = self.q_a_proj(x)  # [B, S, q_lora_rank]
        q_compressed = self.q_a_layernorm(q_compressed)
        q = self.q_b_proj(q_compressed)  # [B, S, num_heads * head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Split Q into non-positional and positional parts
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # KV joint compression with RoPE extraction
        compressed_kv_with_rope = self.kv_a_proj_with_rope(x)  # [B, S, kv_lora_rank + rope_dim]
        compressed_kv, k_pe = torch.split(
            compressed_kv_with_rope, 
            [self.kv_lora_rank, self.qk_rope_head_dim], 
            dim=-1
        )
        
        # KV decompression
        compressed_kv_norm = self.kv_a_layernorm(compressed_kv)
        kv_decompressed = self.kv_b_proj(compressed_kv_norm)  # [B, S, num_heads * (nope_dim + head_dim)]
        kv_decompressed = kv_decompressed.view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.head_dim
        ).transpose(1, 2)
        
        # Split into K_nope and V
        k_nope, v = torch.split(
            kv_decompressed, 
            [self.qk_nope_head_dim, self.head_dim], 
            dim=-1
        )
        
        # Prepare K_pe for RoPE (shared across heads - MQA style for positional part)
        k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        # Apply RoPE to positional parts
        cos_freqs, sin_freqs = self.rope(x, seq_len)
        q_pe = apply_mla_rope(q_pe, cos_freqs, sin_freqs)
        k_pe = apply_mla_rope(k_pe, cos_freqs, sin_freqs)
        
        # Expand K_pe to all heads (like MQA for positional component)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        
        # Concatenate non-positional and positional parts
        q_full = torch.cat([q_nope, q_pe], dim=-1)  # [B, num_heads, S, head_dim]
        k_full = torch.cat([k_nope, k_pe], dim=-1)  # [B, num_heads, S, head_dim]
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, S, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        
        return output


class MLADecoderSelfAttention(nn.Module):
    """Multi-Head Latent Attention for Decoder Self-Attention"""
    
    def __init__(self, embed_dim, num_heads=16, kv_lora_rank=None, q_lora_rank=None, 
                 qk_rope_head_dim=32, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # More aggressive compression for decoder self-attention (3x compression)
        if kv_lora_rank is None:
            self.kv_lora_rank = max(embed_dim // 3, 48)
        else:
            self.kv_lora_rank = kv_lora_rank
            
        if q_lora_rank is None:
            self.q_lora_rank = max(embed_dim // 2, 64)
        else:
            self.q_lora_rank = q_lora_rank
        
        # RoPE configuration
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = self.head_dim - qk_rope_head_dim
        
        assert self.qk_nope_head_dim > 0, f"RoPE dim {qk_rope_head_dim} too large for head dim {self.head_dim}"
        
        # Query projection layers (low-rank)
        self.q_a_proj = nn.Linear(embed_dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, num_heads * self.head_dim, bias=False)
        
        # KV joint compression with decoupled RoPE
        self.kv_a_proj_with_rope = nn.Linear(
            embed_dim, 
            self.kv_lora_rank + qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        
        # KV decompression
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.head_dim),  # K_nope + V
            bias=False
        )
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        # RoPE and dropout
        self.rope = MLAPositionalEmbedding(qk_rope_head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Query projection with low-rank factorization
        q_compressed = self.q_a_proj(x)
        q_compressed = self.q_a_layernorm(q_compressed)
        q = self.q_b_proj(q_compressed)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Split Q into non-positional and positional parts
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # KV joint compression with RoPE extraction
        compressed_kv_with_rope = self.kv_a_proj_with_rope(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv_with_rope, 
            [self.kv_lora_rank, self.qk_rope_head_dim], 
            dim=-1
        )
        
        # KV decompression
        compressed_kv_norm = self.kv_a_layernorm(compressed_kv)
        kv_decompressed = self.kv_b_proj(compressed_kv_norm)
        kv_decompressed = kv_decompressed.view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.head_dim
        ).transpose(1, 2)
        
        # Split into K_nope and V
        k_nope, v = torch.split(
            kv_decompressed, 
            [self.qk_nope_head_dim, self.head_dim], 
            dim=-1
        )
        
        # Prepare K_pe for RoPE (shared across heads)
        k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        # Apply RoPE to positional parts
        cos_freqs, sin_freqs = self.rope(x, seq_len)
        q_pe = apply_mla_rope(q_pe, cos_freqs, sin_freqs)
        k_pe = apply_mla_rope(k_pe, cos_freqs, sin_freqs)
        
        # Expand K_pe to all heads
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        
        # Concatenate non-positional and positional parts
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe], dim=-1)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        
        return output


class MLADecoderCrossAttention(nn.Module):
    """Multi-Head Latent Attention for Decoder Cross-Attention"""
    
    def __init__(self, encoder_embed_dim, decoder_embed_dim, num_heads=16, kv_lora_rank=None, dropout=0.1):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_heads = num_heads
        self.head_dim = decoder_embed_dim // num_heads
        
        # Aggressive compression for cross-attention (encoder is static, can compress heavily)
        if kv_lora_rank is None:
            self.kv_lora_rank = max(encoder_embed_dim // 4, 32)  # 4x compression
        else:
            self.kv_lora_rank = kv_lora_rank
        
        # Query projection (from decoder)
        self.q_proj = nn.Linear(decoder_embed_dim, num_heads * self.head_dim, bias=False)
        
        # KV compression for encoder
        self.kv_a_proj = nn.Linear(encoder_embed_dim, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            num_heads * 2 * self.head_dim,  # K + V
            bias=False
        )
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, decoder_embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, encoder_output):
        batch_size, q_seq_len, _ = query.shape
        enc_seq_len = encoder_output.shape[1]
        
        # Query projection
        q = self.q_proj(query)  # [B, q_seq_len, num_heads * head_dim]
        q = q.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV compression for encoder
        compressed_kv = self.kv_a_proj(encoder_output)  # [B, enc_seq_len, kv_lora_rank]
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        
        # Decompress KV
        kv = self.kv_b_proj(compressed_kv)  # [B, enc_seq_len, num_heads * 2 * head_dim]
        kv = kv.view(batch_size, enc_seq_len, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        k, v = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)
        
        # Compute cross-attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, q_seq_len, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        
        return output


# Helper function to calculate memory compression
def calculate_mla_compression(embed_dim, num_heads, num_layers, kv_lora_rank=None, seq_len=100):
    """Calculate memory compression ratio for MLA vs standard attention"""
    
    head_dim = embed_dim // num_heads
    
    # Standard attention memory per token per layer
    standard_kv_size = num_heads * head_dim * 2  # K + V
    standard_total = standard_kv_size * num_layers * seq_len
    
    # MLA memory per token per layer
    if kv_lora_rank is None:
        kv_lora_rank = embed_dim // 3  # Default 3x compression
    
    mla_kv_size = kv_lora_rank  # Compressed representation
    mla_total = mla_kv_size * num_layers * seq_len
    
    compression_ratio = standard_total / mla_total
    memory_saved_percent = (1 - mla_total / standard_total) * 100
    
    return {
        'standard_memory': standard_total,
        'mla_memory': mla_total,
        'compression_ratio': compression_ratio,
        'memory_saved_percent': memory_saved_percent
    }


# Example usage and testing
if __name__ == "__main__":
    # Test MLA Encoder Self-Attention
    batch_size, seq_len, embed_dim = 4, 32, 128
    num_heads = 16
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create MLA encoder self-attention
    mla_encoder_attn = MLAEncoderSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kv_lora_rank=64,  # 2x compression
        qk_rope_head_dim=16
    )
    
    # Test forward pass
    output = mla_encoder_attn(x)
    print(f"MLA Encoder Self-Attention output shape: {output.shape}")
    
    # Test MLA Decoder Self-Attention
    mla_decoder_self_attn = MLADecoderSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kv_lora_rank=42,  # 3x compression
        qk_rope_head_dim=16
    )
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    output = mla_decoder_self_attn(x, mask=mask)
    print(f"MLA Decoder Self-Attention output shape: {output.shape}")
    
    # Test MLA Cross-Attention
    encoder_output = torch.randn(batch_size, 20, embed_dim)  # Different sequence length
    decoder_query = torch.randn(batch_size, seq_len, embed_dim)
    
    mla_cross_attn = MLADecoderCrossAttention(
        encoder_embed_dim=embed_dim,
        decoder_embed_dim=embed_dim,
        num_heads=num_heads,
        kv_lora_rank=32  # 4x compression for cross-attention
    )
    
    cross_output = mla_cross_attn(decoder_query, encoder_output)
    print(f"MLA Cross-Attention output shape: {cross_output.shape}")
    
    # Calculate memory savings
    compression_stats = calculate_mla_compression(
        embed_dim=128, 
        num_heads=16, 
        num_layers=16, 
        kv_lora_rank=42,  # 3x compression
        seq_len=100
    )
    
    print(f"\nMemory Analysis:")
    print(f"Standard attention memory: {compression_stats['standard_memory']:,}")
    print(f"MLA memory: {compression_stats['mla_memory']:,}")
    print(f"Compression ratio: {compression_stats['compression_ratio']:.2f}x smaller")
    print(f"Memory saved: {compression_stats['memory_saved_percent']:.1f}%") 