"""
Configuration settings for Traffic Anomaly Detection System
"""

import os
from pathlib import Path

class AppConfig:
    """Application configuration"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "models"
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Ensure directories exist
    MODEL_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Model settings
    MODEL_NAME = "best_model.h5"
    MODEL_PATH = MODEL_DIR / MODEL_NAME
    IMAGE_SIZE = (128, 128)
    
    # Class names and colors
    CLASS_NAMES = ['accident', 'dense_traffic', 'fire', 'sparse_traffic']
    CLASS_COLORS = {
        'accident': (0, 0, 255),        # Red
        'fire': (0, 100, 255),          # Orange  
        'dense_traffic': (0, 255, 255), # Yellow
        'sparse_traffic': (0, 255, 0)   # Green
    }
    
    CLASS_EMOJIS = {
        'accident': '🚨',
        'fire': '🔥',
        'dense_traffic': '⚠️',
        'sparse_traffic': '✅'
    }
    
    CLASS_SEVERITY = {
        'accident': 'critical',
        'fire': 'critical',
        'dense_traffic': 'warning',
        'sparse_traffic': 'normal'
    }
    
    # Video processing settings
    VIDEO_BUFFER_SIZE = 30
    SKIP_FRAMES = 2  # Process every Nth frame
    CONFIDENCE_THRESHOLD = 0.7
    TEMPORAL_WINDOW = 10  # Frames for temporal smoothing
    SMOOTHING_ALPHA = 0.7  # Exponential moving average
    
    # Wavelet settings
    WAVELET_TYPE = 'haar'
    WAVELET_LEVEL = 2
    
    # Training settings (for reference)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
    SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'flv']
    
    # Performance settings
    MAX_BATCH_SIZE = 50  # Maximum images for batch processing
    MAX_VIDEO_SIZE_MB = 500  # Maximum video size in MB
    
    # Alert settings
    ALERT_COOLDOWN = 5  # Seconds between alerts
    NOTIFICATION_ENABLED = True
    
    @classmethod
    def get_color_bgr(cls, class_name):
        """Get BGR color tuple for a class"""
        return cls.CLASS_COLORS.get(class_name, (255, 255, 255))
    
    @classmethod
    def get_color_rgb(cls, class_name):
        """Get RGB color tuple for a class"""
        bgr = cls.CLASS_COLORS.get(class_name, (255, 255, 255))
        return (bgr[2], bgr[1], bgr[0])
    
    @classmethod
    def get_emoji(cls, class_name):
        """Get emoji for a class"""
        return cls.CLASS_EMOJIS.get(class_name, '📍')
    
    @classmethod
    def get_severity(cls, class_name):
        """Get severity level for a class"""
        return cls.CLASS_SEVERITY.get(class_name, 'normal')