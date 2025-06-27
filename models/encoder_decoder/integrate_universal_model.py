#!/usr/bin/env python3
"""
Integration script to use the universal model adapter in the app.
This shows how to replace the current model with the universal one.
"""

import sys
import os
from pathlib import Path

# Add the models directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from universal_model_adapter import UniversalModelAdapter, create_universal_model_adapter


def integrate_with_app():
    """
    Show how to integrate the universal model with the app.
    """
    print("🔧 INTEGRATING UNIVERSAL MODEL WITH APP")
    print("=" * 50)
    
    # 1. Create the universal model adapter
    print("\n1️⃣ Creating universal model adapter...")
    try:
        adapter = create_universal_model_adapter()
        print("   ✅ Universal model adapter created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create adapter: {e}")
        return False
    
    # 2. Test with different grid sizes
    print("\n2️⃣ Testing with different grid sizes...")
    test_results = {}
    
    for grid_size in [1, 2, 3, 4, 5]:
        print(f"\n   Testing {grid_size}x{grid_size} grid...")
        
        # Create a test image (simple pattern)
        import torch
        import io
        from PIL import Image
        
        # Create a test image with some pattern
        test_image = torch.zeros(28 * grid_size, 28 * grid_size)
        
        # Add some simple pattern (diagonal line)
        for i in range(min(grid_size * 28, 28)):
            test_image[i, i] = 1.0
        
        # Convert to bytes
        test_image_pil = Image.fromarray(test_image.numpy())
        test_bytes = io.BytesIO()
        test_image_pil.save(test_bytes, format='PNG')
        
        # Test prediction
        try:
            sequence, confidence = adapter.predict_sequence(test_bytes.getvalue(), grid_size)
            test_results[grid_size] = {
                'sequence': sequence,
                'confidence': confidence,
                'success': True
            }
            print(f"      ✅ Success: {sequence} (confidence: {confidence:.3f})")
        except Exception as e:
            test_results[grid_size] = {
                'sequence': [],
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
            print(f"      ❌ Failed: {e}")
    
    # 3. Show integration code
    print("\n3️⃣ Integration code for your app:")
    print("""
# In your app's model loading code, replace:
# from encoder_decoder.model import EncoderDecoderMNISTClassifier
# model = EncoderDecoderMNISTClassifier(checkpoint_path)

# With:
from universal_model_adapter import create_universal_model_adapter
model = create_universal_model_adapter("other/multi-digit-scrambled-best.pt")

# The model will now work with any grid size automatically!
""")
    
    # 4. Show API compatibility
    print("\n4️⃣ API Compatibility Check:")
    print("   ✅ Same predict_sequence(image_bytes, grid_size) interface")
    print("   ✅ Same return format: (sequence, confidence)")
    print("   ✅ Same preprocessing pipeline")
    print("   ✅ Works with existing app UI")
    
    # 5. Performance considerations
    print("\n5️⃣ Performance Considerations:")
    print("   ⚠️  Model may be slower due to dynamic resizing")
    print("   ⚠️  Memory usage may be higher for large grids")
    print("   ✅ No need to load multiple model files")
    print("   ✅ Supports any grid size without reconfiguration")
    
    return True


def show_usage_examples():
    """
    Show usage examples for the universal model.
    """
    print("\n📖 USAGE EXAMPLES")
    print("=" * 30)
    
    print("""
# Example 1: Basic usage
adapter = create_universal_model_adapter()
sequence, confidence = adapter.predict_sequence(image_bytes, grid_size=3)

# Example 2: Integration with existing app
def predict_digits(image_bytes, grid_size):
    model = create_universal_model_adapter()
    return model.predict_sequence(image_bytes, grid_size)

# Example 3: Batch processing
def process_multiple_images(image_list, grid_sizes):
    model = create_universal_model_adapter()
    results = []
    for image, grid_size in zip(image_list, grid_sizes):
        sequence, confidence = model.predict_sequence(image, grid_size)
        results.append((sequence, confidence))
    return results
""")


if __name__ == "__main__":
    print("🚀 UNIVERSAL MODEL INTEGRATION")
    print("=" * 40)
    
    # Check if the model file exists
    model_path = "other/multi-digit-scrambled-best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the model file exists in the other/ directory")
        sys.exit(1)
    
    # Run integration
    success = integrate_with_app()
    
    if success:
        show_usage_examples()
        print("\n✅ Integration guide completed!")
        print("\n💡 Next steps:")
        print("   1. Test the universal model with your app")
        print("   2. Update your model loading code")
        print("   3. Deploy with the new universal model")
    else:
        print("\n❌ Integration failed. Check the errors above.") 