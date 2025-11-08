#!/usr/bin/env python3
"""
Test script to verify the Traffic Anomaly Detection System installation
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'plotly': 'Plotly',
        'pywt': 'PyWavelets',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow'
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {str(e)}")
            failed.append(name)
    
    return len(failed) == 0

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    base_dir = Path(__file__).parent
    required_dirs = ['models', 'temp', 'outputs', 'utils', 'pages']
    
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            all_exist = False
    
    return all_exist

def test_files():
    """Test if required files exist"""
    print("\nTesting required files...")
    
    base_dir = Path(__file__).parent
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'utils/config.py',
        'utils/model_utils.py',
        'utils/video_utils.py',
        'pages/image_detection.py'
    ]
    
    all_exist = True
    
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (missing)")
            all_exist = False
    
    return all_exist

def test_model():
    """Test if model file exists"""
    print("\nTesting model file...")
    
    base_dir = Path(__file__).parent
    model_path = base_dir / 'models' / 'best_model.h5'
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Model file found ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  ⚠️  Model file not found: {model_path}")
        print("      The application will not work without the model file")
        print("      Please place 'best_model.h5' in the models/ directory")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from utils.model_utils import load_trained_model
        
        model = load_trained_model()
        
        if model is not None:
            print("  ✅ Model loaded successfully")
            
            # Test model architecture
            try:
                input_shape = model.input_shape
                output_shape = model.output_shape
                print(f"     Input shape: {input_shape}")
                print(f"     Output shape: {output_shape}")
                
                if output_shape[-1] == 4:
                    print("     ✅ Correct number of output classes (4)")
                else:
                    print(f"     ⚠️  Expected 4 output classes, got {output_shape[-1]}")
            except Exception as e:
                print(f"     ⚠️  Could not verify model architecture: {str(e)}")
            
            return True
        else:
            print("  ❌ Model loading failed")
            return False
    
    except Exception as e:
        print(f"  ❌ Error loading model: {str(e)}")
        return False

def test_dummy_prediction():
    """Test if a dummy prediction works"""
    print("\nTesting dummy prediction...")
    
    try:
        import numpy as np
        from utils.model_utils import load_trained_model, predict_single_image
        
        model = load_trained_model()
        
        if model is None:
            print("  ⚠️  Skipping (model not loaded)")
            return False
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Predict
        result = predict_single_image(model, dummy_image)
        
        print(f"  ✅ Prediction successful")
        print(f"     Predicted class: {result['class']}")
        print(f"     Confidence: {result['confidence']:.2f}%")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Prediction failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("  Traffic Anomaly Detection System - Installation Test")
    print("=" * 60)
    
    results = {
        "Package Imports": test_imports(),
        "Directory Structure": test_directories(),
        "Required Files": test_files(),
        "Model File": test_model(),
        "Model Loading": test_model_loading(),
        "Dummy Prediction": test_dummy_prediction()
    }
    
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    
    all_passed = all(results.values())
    
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the application, run:")
        print("  streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Add model file: Place 'best_model.h5' in models/ directory")
        print("  • Check file structure: Ensure all files are present")
    
    print("\n" + "=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())