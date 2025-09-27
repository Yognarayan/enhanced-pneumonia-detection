"""
Test script for the Chest X-ray Pneumonia Detection Explainability System

Run this script to validate your installation and test the explainability pipeline.
"""

import sys
import os
import numpy as np
from PIL import Image

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('tensorflow', 'tf'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('PIL', 'Image'),
        ('shap', 'shap'),
        ('lime', 'lime'),
        ('cv2', 'cv2'),
        ('skimage', 'skimage'),
        ('gradio', 'gr'),
        ('pandas', 'pd')
    ]
    
    missing_packages = []
    
    for package, alias in required_packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\\n✅ All dependencies are available!")
        return True


def test_model_loading():
    """Test loading the pneumonia detection model."""
    print("\\n🔍 Testing model loading...")
    
    try:
        from huggingface_hub import hf_hub_download
        import tensorflow as tf
        
        print("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="ayushirathour/chest-xray-pneumonia-detection",
            filename="best_chest_xray_model.h5"
        )
        
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None


def test_explainability_imports():
    """Test importing explainability modules."""
    print("\\n🔍 Testing explainability imports...")
    
    try:
        from explain.gradcam import GradCAM
        from explain.lime_explain import LIMEExplainer  
        from explain.shap_explain import SHAPExplainer
        from explain.dashboard import ProbabilityDashboard
        from explain.report import HTMLReportGenerator
        from explain.pipeline import ExplainabilityPipeline
        
        print("✅ All explainability modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Explainability import failed: {e}")
        return False


def create_test_image():
    """Create a synthetic test image for validation."""
    print("\\n🔍 Creating test image...")
    
    try:
        # Create a synthetic chest X-ray-like image
        img_array = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        center_x, center_y = 112, 112
        for i in range(224):
            for j in range(224):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < 80:  # Lung region
                    img_array[i, j] = img_array[i, j] * 0.7  # Darker
        
        # Save test image
        test_img = Image.fromarray(img_array)
        test_path = "./test_chest_xray.png"
        test_img.save(test_path)
        
        print(f"✅ Test image created: {test_path}")
        return test_path
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return None


def test_pipeline(model, test_image_path):
    """Test the complete explainability pipeline."""
    print("\\n🔍 Testing explainability pipeline...")
    
    if not model or not test_image_path:
        print("❌ Cannot test pipeline - missing model or test image")
        return False
    
    try:
        from explain.pipeline import generate_complete_explanation
        from PIL import Image
        
        def test_preprocess(pil_img):
            """Test preprocessing function."""
            img = pil_img.convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        
        # Test basic model prediction
        print("Testing model prediction...")
        test_img = Image.open(test_image_path)
        processed = test_preprocess(test_img)
        prediction = model.predict(processed, verbose=0)
        print(f"✅ Model prediction: {prediction[0][0]:.4f}")
        
        # Test explainability pipeline (with SHAP disabled for speed)
        print("Testing explainability pipeline...")
        
        output_dir = "./test_explanations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Note: We'll disable SHAP for the test to make it faster
        from explain.pipeline import ExplainabilityPipeline
        
        pipeline = ExplainabilityPipeline(model, test_preprocess)
        
        # Test individual components
        print("  - Testing Grad-CAM...")
        from explain.gradcam import generate_gradcam_explanation
        
        original_img = np.array(test_img)
        gradcam_files = generate_gradcam_explanation(
            model, processed, original_img, "TEST", output_dir
        )
        print(f"    ✅ Grad-CAM files: {gradcam_files}")
        
        print("  - Testing probability dashboard...")
        from explain.dashboard import generate_probability_dashboard
        
        probabilities = np.array([1 - prediction[0][0], prediction[0][0]])
        dashboard_file = generate_probability_dashboard(
            probabilities, "TEST", output_dir
        )
        print(f"    ✅ Dashboard file: {dashboard_file}")
        
        print("✅ Pipeline test completed successfully!")
        print(f"   Check results in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up test files."""
    print("\\n🧹 Cleaning up test files...")
    
    files_to_remove = [
        "./test_chest_xray.png"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Removed: {file_path}")


def main():
    """Run all tests."""
    print("🚀 Starting Chest X-ray Pneumonia Detection Explainability Test Suite")
    print("=" * 70)
    
    # Test 1: Check dependencies
    deps_ok = check_dependencies()
    
    # Test 2: Model loading
    model = test_model_loading() if deps_ok else None
    
    # Test 3: Explainability imports
    explain_ok = test_explainability_imports() if deps_ok else False
    
    # Test 4: Create test image
    test_image = create_test_image() if deps_ok else None
    
    # Test 5: Pipeline test
    pipeline_ok = False
    if deps_ok and explain_ok and model and test_image:
        pipeline_ok = test_pipeline(model, test_image)
    
    # Summary
    print("\\n" + "=" * 70)
    print("🎯 Test Results Summary:")
    print(f"   Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"   Model Loading: {'✅ PASS' if model else '❌ FAIL'}")
    print(f"   Explainability: {'✅ PASS' if explain_ok else '❌ FAIL'}")
    print(f"   Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if deps_ok and model and explain_ok and pipeline_ok:
        print("\\n🎉 All tests passed! Your installation is ready to use.")
        print("\\n📋 Next steps:")
        print("   1. Run the web interface: python pheumonia/pneumonia.py")
        print("   2. Try CLI: python -m explain.run --create-sample-csv sample.csv")
        print("   3. Check the README.md for detailed usage examples")
    else:
        print("\\n⚠️  Some tests failed. Please check the errors above and:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check your internet connection for model download")
        print("   3. Verify TensorFlow installation")
    
    # Cleanup
    cleanup()


if __name__ == "__main__":
    main()