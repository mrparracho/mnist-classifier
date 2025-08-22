#!/usr/bin/env python3
"""
CLI utility to load and inspect encoder-decoder model properties.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Add the models directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoder_decoder.model import EncoderDecoderMNISTClassifier
from config import get_model_config


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of model parameters and architecture."""
    total_params = 0
    trainable_params = 0
    layer_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if params > 0:
                total_params += params
                trainable_params += trainable
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params,
                    'trainable': trainable,
                    'shape': str(list(module.parameters())[0].shape) if list(module.parameters()) else 'N/A'
                })
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layers': layer_info
    }


def display_model_properties(model_instance: EncoderDecoderMNISTClassifier, grid_size: int):
    """Display comprehensive model properties."""
    print("=" * 80)
    print(f"ENCODER-DECODER MODEL INSPECTOR")
    print("=" * 80)
    
    # Basic model info
    print(f"\n📋 BASIC INFORMATION:")
    print(f"   Model Type: {type(model_instance).__name__}")
    print(f"   Grid Size: {grid_size}x{grid_size}")
    print(f"   Checkpoint Path: {model_instance.checkpoint_path}")
    print(f"   Device: {model_instance.device}")
    
    # Configuration info
    print(f"\n⚙️  CONFIGURATION:")
    try:
        config = get_model_config("encoder_decoder").config
        if config:
            print(f"   Patch Size: {config.get('patch_size', 'N/A')}")
            print(f"   Encoder Embed Dim: {config.get('encoder_embed_dim', 'N/A')}")
            print(f"   Decoder Embed Dim: {config.get('decoder_embed_dim', 'N/A')}")
            print(f"   Num Layers: {config.get('num_layers', 'N/A')}")
            print(f"   Num Heads: {config.get('num_heads', 'N/A')}")
            print(f"   Dropout: {config.get('dropout', 'N/A')}")
            print(f"   Normalize Mean: {config.get('normalize_mean', 'N/A')}")
            print(f"   Normalize Std: {config.get('normalize_std', 'N/A')}")
        else:
            print("   ⚠️  Configuration not found")
    except Exception as e:
        print(f"   ⚠️  Error loading config: {e}")
    
    # Architecture info
    print(f"\n🏗️  ARCHITECTURE:")
    image_size = grid_size * 28
    max_seq_len = grid_size * grid_size + 2
    print(f"   Input Image Size: {image_size}x{image_size} pixels")
    print(f"   Max Sequence Length: {max_seq_len} tokens")
    print(f"   Expected Output: {grid_size * grid_size} digits")
    
    # Load the model
    print(f"\n🔄 LOADING MODEL...")
    try:
        model = model_instance.load_model()
        print(f"   ✅ Model loaded successfully")
        
        # Model summary
        summary = get_model_summary(model)
        
        print(f"\n📊 MODEL STATISTICS:")
        print(f"   Total Parameters: {format_number(summary['total_params'])}")
        print(f"   Trainable Parameters: {format_number(summary['trainable_params'])}")
        model_size_bytes = summary['total_params'] * 4
        model_size_mb = model_size_bytes / 1024 / 1024
        print(f"   Model Size: {format_number(model_size_bytes)} bytes ({model_size_mb:.1f} MB)")
        
        # Layer breakdown
        print(f"\n🔍 LAYER BREAKDOWN:")
        for layer in summary['layers'][:10]:  # Show first 10 layers
            print(f"   {layer['name']:<30} {layer['type']:<20} {format_number(layer['params']):>10} params")
        
        if len(summary['layers']) > 10:
            print(f"   ... and {len(summary['layers']) - 10} more layers")
        
        # Checkpoint info
        if model_instance.checkpoint_path and os.path.exists(model_instance.checkpoint_path):
            checkpoint_size = os.path.getsize(model_instance.checkpoint_path)
            print(f"\n💾 CHECKPOINT INFORMATION:")
            print(f"   File Size: {format_number(checkpoint_size)} bytes ({checkpoint_size / 1024 / 1024:.1f} MB)")
            
            # Load checkpoint metadata
            try:
                checkpoint = torch.load(model_instance.checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    print(f"   Checkpoint Keys: {list(checkpoint.keys())}")
                    if 'epoch' in checkpoint:
                        print(f"   Training Epoch: {checkpoint['epoch']}")
                    if 'accuracy' in checkpoint:
                        print(f"   Training Accuracy: {checkpoint['accuracy']:.4f}")
                    if 'loss' in checkpoint:
                        print(f"   Training Loss: {checkpoint['loss']:.4f}")
            except Exception as e:
                print(f"   ⚠️  Error reading checkpoint metadata: {e}")
        
        # Test prediction capability
        print(f"\n🧪 PREDICTION TEST:")
        try:
            # Create a dummy image (all zeros)
            dummy_image = torch.zeros(1, 1, image_size, image_size)
            model.eval()
            
            with torch.no_grad():
                # Test forward pass
                output = model(dummy_image, max_length=max_seq_len)
                print(f"   ✅ Forward pass successful")
                print(f"   Output Shape: {output.shape}")
                print(f"   Output Type: {output.dtype}")
        except Exception as e:
            print(f"   ❌ Prediction test failed: {e}")
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    print(f"\n" + "=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inspect encoder-decoder model properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_inspector.py --grid-size 1
  python model_inspector.py --grid-size 2 --checkpoint-path custom/path/model.pt
  python model_inspector.py --list-checkpoints
        """
    )
    
    parser.add_argument(
        '--grid-size', 
        type=int, 
        choices=[1, 2, 3, 4],
        help='Grid size to inspect (1-4)'
    )
    
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='Custom checkpoint path (overrides default)'
    )
    
    parser.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List available checkpoint files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        print("📁 AVAILABLE CHECKPOINT FILES:")
        checkpoint_dir = Path(__file__).parent / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob("*.pt"):
                size = checkpoint_file.stat().st_size
                print(f"   {checkpoint_file.name:<40} {size:>10,} bytes ({size/1024/1024:.1f} MB)")
        else:
            print(f"   Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Validate arguments
    if not args.grid_size and not args.list_checkpoints:
        parser.error("Either --grid-size or --list-checkpoints must be specified")
    
    if args.grid_size:
        # Determine checkpoint path
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            # Use default path based on grid size
            checkpoint_path = f"encoder_decoder/checkpoints/mnist-encoder-decoder-{args.grid_size}-varlen.pt"
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print(f"💡 Use --list-checkpoints to see available files")
            return False
        
        # Create model instance
        try:
            model_instance = EncoderDecoderMNISTClassifier(checkpoint_path=checkpoint_path)
            success = display_model_properties(model_instance, args.grid_size)
            return success
        except Exception as e:
            print(f"❌ Error creating model instance: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 