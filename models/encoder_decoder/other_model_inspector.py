#!/usr/bin/env python3
"""
Utility to inspect model parameters from the other/ folder by analyzing the actual model state dict.
This helps determine the real configuration used during training without relying on hardcoded values.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import re

# Add the models directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoder_decoder.model import EncoderDecoder


def analyze_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the state dict to determine the actual model configuration.
    
    Args:
        state_dict: The model's state dictionary
        
    Returns:
        Dictionary containing inferred configuration parameters
    """
    config = {}
    
    # Handle nested structure where model state dict is under 'model' key
    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        actual_state_dict = state_dict['model']
        print(f"   Found nested structure: model state dict has {len(actual_state_dict)} parameters")
    else:
        actual_state_dict = state_dict
    
    # Extract encoder embedding dimension from patch embedding
    patch_embed_keys = [k for k in actual_state_dict.keys() if 'patch_embedding' in k]
    if patch_embed_keys:
        # Look for the linear layer weight in patch embedding
        for key in patch_embed_keys:
            if 'patch_embedding.weight' in key:
                weight = actual_state_dict[key]
                if hasattr(weight, 'shape'):
                    # patch_embedding.weight shape is [embed_dim, patch_size * patch_size]
                    embed_dim = weight.shape[0]
                    patch_size_squared = weight.shape[1]
                    patch_size = int(patch_size_squared ** 0.5)
                    config['encoder_embed_dim'] = embed_dim
                    config['patch_size'] = patch_size
                    break
    
    # Extract decoder embedding dimension from decoder embedding
    decoder_embed_keys = [k for k in actual_state_dict.keys() if 'decoder_embedding' in k]
    if decoder_embed_keys:
        for key in decoder_embed_keys:
            if 'decoder_embedding.weight' in key:
                weight = actual_state_dict[key]
                if hasattr(weight, 'shape'):
                    # decoder_embedding.weight shape is [max_seq_len, decoder_embed_dim]
                    decoder_embed_dim = weight.shape[1]
                    max_seq_len = weight.shape[0]
                    config['decoder_embed_dim'] = decoder_embed_dim
                    config['max_seq_len'] = max_seq_len
                    break
    
    # Count encoder blocks
    encoder_block_keys = [k for k in actual_state_dict.keys() if 'encoder_blocks' in k]
    if encoder_block_keys:
        # Extract block indices from keys like "encoder_blocks.0.attention.W_q.weight"
        block_indices = set()
        for key in encoder_block_keys:
            match = re.search(r'encoder_blocks\.(\d+)\.', key)
            if match:
                block_indices.add(int(match.group(1)))
        config['num_layers'] = len(block_indices)
    
    # Count decoder blocks
    decoder_block_keys = [k for k in actual_state_dict.keys() if 'decoder_blocks' in k]
    if decoder_block_keys:
        block_indices = set()
        for key in decoder_block_keys:
            match = re.search(r'decoder_blocks\.(\d+)\.', key)
            if match:
                block_indices.add(int(match.group(1)))
        # Verify decoder layers match encoder layers
        if 'num_layers' in config:
            if len(block_indices) != config['num_layers']:
                print(f"WARNING: Encoder layers ({config['num_layers']}) != Decoder layers ({len(block_indices)})")
    
    # Extract number of heads from attention weights
    attention_keys = [k for k in actual_state_dict.keys() if 'attention' in k and 'W_q.weight' in k]
    if attention_keys:
        for key in attention_keys:
            weight = actual_state_dict[key]
            if hasattr(weight, 'shape'):
                # W_q.weight shape is [embed_dim, embed_dim]
                embed_dim = weight.shape[0]
                # Assuming num_heads divides embed_dim evenly
                possible_heads = [1, 2, 4, 8, 16, 32]
                for num_heads in possible_heads:
                    if embed_dim % num_heads == 0:
                        config['num_heads'] = num_heads
                        break
    
    # Try to infer image size from patch embedding position embeddings
    pos_embed_keys = [k for k in actual_state_dict.keys() if 'position_embedding' in k]
    if pos_embed_keys:
        for key in pos_embed_keys:
            weight = actual_state_dict[key]
            if hasattr(weight, 'shape'):
                # position_embedding shape is [num_patches, embed_dim]
                num_patches = weight.shape[0]
                if 'patch_size' in config:
                    # num_patches = (image_size // patch_size) ** 2
                    image_size = int((num_patches ** 0.5) * config['patch_size'])
                    config['image_size'] = image_size
                    # Infer grid size
                    if image_size % 28 == 0:
                        grid_size = image_size // 28
                        config['grid_size'] = grid_size
                    break
    
    # Extract output projection dimensions
    output_keys = [k for k in actual_state_dict.keys() if 'output_projection' in k]
    if output_keys:
        for key in output_keys:
            if 'output_projection.weight' in key:
                weight = actual_state_dict[key]
                if hasattr(weight, 'shape'):
                    # output_projection.weight shape is [vocab_size, decoder_embed_dim]
                    vocab_size = weight.shape[0]
                    config['vocab_size'] = vocab_size
                    break
    
    # Extract training information if available
    if 'training' in state_dict and isinstance(state_dict['training'], dict):
        training_info = state_dict['training']
        config['training_info'] = {}
        for key, value in training_info.items():
            if isinstance(value, (int, float, str)):
                config['training_info'][key] = value
            elif hasattr(value, 'item'):  # torch tensor
                config['training_info'][key] = value.item()
    
    return config


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def display_model_analysis(checkpoint_path: str, verbose: bool = False):
    """
    Display comprehensive analysis of a model checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        verbose: Whether to show detailed information
    """
    print("=" * 80)
    print(f"MODEL ANALYSIS: {os.path.basename(checkpoint_path)}")
    print("=" * 80)
    
    # Basic file info
    if os.path.exists(checkpoint_path):
        file_size = os.path.getsize(checkpoint_path)
        print(f"\n📁 FILE INFORMATION:")
        print(f"   Path: {checkpoint_path}")
        print(f"   Size: {format_number(file_size)} bytes ({file_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"❌ File not found: {checkpoint_path}")
        return False
    
    # Load state dict
    print(f"\n🔄 LOADING STATE DICT...")
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        print(f"   ✅ State dict loaded successfully")
        
        if isinstance(state_dict, dict):
            print(f"   Keys: {len(state_dict)} parameters")
            if verbose:
                print(f"   Parameter names: {list(state_dict.keys())[:10]}...")
        else:
            print(f"   ⚠️  State dict is not a dictionary: {type(state_dict)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to load state dict: {e}")
        return False
    
    # Analyze configuration
    print(f"\n🔍 ANALYZING CONFIGURATION...")
    config = analyze_state_dict(state_dict)
    
    if config:
        print(f"   ✅ Configuration extracted successfully")
        print(f"\n⚙️  INFERRED CONFIGURATION:")
        for key, value in sorted(config.items()):
            print(f"   {key:<20}: {value}")
    else:
        print(f"   ⚠️  Could not extract configuration")
        return False
    
    # Calculate model size
    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        actual_state_dict = state_dict['model']
        total_params = sum(p.numel() for p in actual_state_dict.values() if hasattr(p, 'numel'))
    else:
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
    
    model_size_bytes = total_params * 4  # Assuming float32
    model_size_mb = model_size_bytes / 1024 / 1024
    
    print(f"\n📊 MODEL STATISTICS:")
    print(f"   Total Parameters: {format_number(total_params)}")
    print(f"   Model Size: {format_number(model_size_bytes)} bytes ({model_size_mb:.1f} MB)")
    
    # Validate configuration
    print(f"\n✅ CONFIGURATION VALIDATION:")
    validation_errors = []
    
    # Check if all required keys are present
    required_keys = ['encoder_embed_dim', 'decoder_embed_dim', 'num_layers', 'patch_size']
    for key in required_keys:
        if key not in config:
            validation_errors.append(f"Missing {key}")
        else:
            print(f"   {key:<20}: ✅ {config[key]}")
    
    # Check for consistency
    if 'image_size' in config and 'grid_size' in config:
        expected_image_size = config['grid_size'] * 28
        if config['image_size'] != expected_image_size:
            validation_errors.append(f"Image size mismatch: {config['image_size']} != {expected_image_size}")
        else:
            print(f"   Image size consistency: ✅ {config['image_size']} = {config['grid_size']} * 28")
    
    if 'max_seq_len' in config and 'grid_size' in config:
        expected_max_seq_len = config['grid_size'] * config['grid_size'] + 2
        if config['max_seq_len'] != expected_max_seq_len:
            validation_errors.append(f"Max seq len mismatch: {config['max_seq_len']} != {expected_max_seq_len}")
        else:
            print(f"   Max seq len consistency: ✅ {config['max_seq_len']} = {config['grid_size']}^2 + 2")
    
    if validation_errors:
        print(f"\n⚠️  VALIDATION WARNINGS:")
        for error in validation_errors:
            print(f"   {error}")
    
    # Show parameter breakdown if verbose
    if verbose:
        print(f"\n🔍 DETAILED PARAMETER BREAKDOWN:")
        param_groups = {}
        for key in state_dict.keys():
            group = key.split('.')[0] if '.' in key else key
            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append(key)
        
        for group, keys in sorted(param_groups.items()):
            group_params = sum(state_dict[k].numel() for k in keys)
            print(f"   {group:<20}: {format_number(group_params)} params ({len(keys)} layers)")
    
    print(f"\n" + "=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model parameters from other/ folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python other_model_inspector.py --checkpoint other/single-digit-v1.pt
  python other_model_inspector.py --checkpoint other/best2by2.pt --verbose
  python other_model_inspector.py --list-all
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to specific checkpoint file'
    )
    
    parser.add_argument(
        '--list-all',
        action='store_true',
        help='List and analyze all checkpoints in other/ folder'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed parameter breakdown'
    )
    
    args = parser.parse_args()
    
    if args.list_all:
        # Analyze all checkpoints in other/ folder
        other_dir = Path(__file__).parent / "other"
        if not other_dir.exists():
            print(f"❌ Directory not found: {other_dir}")
            return False
        
        checkpoint_files = list(other_dir.glob("*.pt"))
        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {other_dir}")
            return False
        
        print(f"🔍 FOUND {len(checkpoint_files)} CHECKPOINT FILES:")
        for i, checkpoint_file in enumerate(checkpoint_files, 1):
            print(f"\n[{i}/{len(checkpoint_files)}] Analyzing {checkpoint_file.name}...")
            success = display_model_analysis(str(checkpoint_file), args.verbose)
            if not success:
                print(f"   ❌ Failed to analyze {checkpoint_file.name}")
        
        return True
    
    elif args.checkpoint:
        # Analyze specific checkpoint
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            return False
        
        return display_model_analysis(args.checkpoint, args.verbose)
    
    else:
        parser.error("Either --checkpoint or --list-all must be specified")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 