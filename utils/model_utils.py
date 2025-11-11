"""
Model utilities for loading and inference
"""

import numpy as np
import cv2
import pywt
from tensorflow.keras.models import load_model
import streamlit as st
from pathlib import Path
from utils.config import AppConfig

config = AppConfig()

@st.cache_resource
def load_trained_model():
    """
    Load the trained model with caching
    Returns the loaded Keras model
    """
    try:
        if config.MODEL_PATH.exists():
                model = load_model(str(config.MODEL_PATH), compile=False)
            return model        else:
            st.error(f"Model not found at {config.MODEL_PATH}")
            st.info("Please place your trained model file 'improved_model.h5' in the 'models' directory")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_wavelet_features(image, wavelet='haar', level=2):
    """
    Extract wavelet features from image
    
    Args:
        image: Grayscale image array
        wavelet: Wavelet type (default: 'haar')
        level: Decomposition level (default: 2)
    
    Returns:
        Array of wavelet features
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    features = []
    
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            for c in coeff:
                features.append(np.mean(c))
                features.append(np.std(c))
        else:
            features.append(np.mean(coeff))
            features.append(np.std(coeff))
    
    return np.array(features)

def preprocess_image(image, size=None):
    """
    Preprocess image for model input
    
    Args:
        image: Input image (BGR format)
        size: Target size tuple (width, height)
    
    Returns:
        Preprocessed image ready for model
    """
    if size is None:
        size = config.IMAGE_SIZE
    
    # Resize
    img_resized = cv2.resize(image, size)
    
    # Normalize to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batched = np.expand_dims(img_normalized, axis=0)
    
    return img_batched

def predict_single_image(model, image):
    """
    Predict anomaly class for a single image
    
    Args:
        model: Trained Keras model
        image: Input image (BGR format)
    
    Returns:
        Dictionary with prediction results
    """
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed, verbose=0)
    
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100
    
    result = {
        'class': config.CLASS_NAMES[predicted_class_idx],
        'class_index': predicted_class_idx,
        'confidence': confidence,
        'probabilities': {
            config.CLASS_NAMES[i]: float(prediction[0][i] * 100)
            for i in range(len(config.CLASS_NAMES))
        },
        'raw_prediction': prediction[0],
        'severity': config.get_severity(config.CLASS_NAMES[predicted_class_idx]),
        'emoji': config.get_emoji(config.CLASS_NAMES[predicted_class_idx])
    }
    
    return result

def predict_batch_images(model, images):
    """
    Predict anomaly classes for multiple images
    
    Args:
        model: Trained Keras model
        images: List of input images (BGR format)
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image in images:
        result = predict_single_image(model, image)
        results.append(result)
    
    return results

def get_model_summary():
    """
    Get model architecture summary
    
    Returns:
        String representation of model architecture
    """
    model = load_trained_model()
    if model is None:
        return "Model not loaded"
    
    from io import StringIO
    import sys
    
    # Capture model.summary() output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    model.summary()
    
    summary_string = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    return summary_string

def validate_image(image):
    """
    Validate image is suitable for processing
    
    Args:
        image: Input image array
    
    Returns:
        Tuple (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if len(image.shape) != 3:
        return False, "Image must be 3-channel (RGB/BGR)"
    
    if image.shape[2] != 3:
        return False, "Image must have 3 channels"
    
    min_size = 32
    if image.shape[0] < min_size or image.shape[1] < min_size:
        return False, f"Image too small (minimum {min_size}x{min_size})"
    
    return True, "Valid"
