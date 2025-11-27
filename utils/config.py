"""
Configuration settings for Traffic Anomaly Detection System
Includes Edge Processing and RTA Dubai Integration settings
"""

import os
from pathlib import Path


class EdgeConfig:
    """Edge processing configuration for Raspberry Pi deployment"""

    # Device identification
    DEVICE_ID = os.environ.get('EDGE_DEVICE_ID', 'edge-device-001')
    CAMERA_ID = os.environ.get('CAMERA_ID', 'cam-001')
    LOCATION_ID = os.environ.get('LOCATION_ID', 'intersection-001')

    # Keyframe extraction settings
    SCENE_THRESHOLD = 30.0           # Scene change detection threshold
    MOTION_THRESHOLD = 5.0           # Motion detection threshold
    MIN_KEYFRAME_INTERVAL = 30       # Minimum frames between keyframes
    MAX_KEYFRAME_INTERVAL = 150      # Maximum frames between keyframes

    # Compression settings
    DEFAULT_COMPRESSION = 'high'     # full, high, medium, low, minimal
    DEFAULT_FORMAT = 'jpeg'          # jpeg or webp

    # Network monitoring
    NETWORK_CHECK_INTERVAL = 5.0     # Seconds between network checks
    NETWORK_TIMEOUT = 5.0            # Network timeout in seconds
    PING_HOSTS = ['8.8.8.8', '1.1.1.1']

    # Latency thresholds (ms)
    LATENCY_EXCELLENT = 50.0
    LATENCY_GOOD = 100.0
    LATENCY_FAIR = 200.0
    LATENCY_POOR = 500.0

    # Bandwidth thresholds (Kbps)
    BANDWIDTH_EXCELLENT = 10000.0    # 10 Mbps
    BANDWIDTH_GOOD = 5000.0          # 5 Mbps
    BANDWIDTH_FAIR = 1000.0          # 1 Mbps
    BANDWIDTH_POOR = 256.0           # 256 Kbps

    # Offline storage
    OFFLINE_DB_PATH = 'data/offline_storage.db'
    MAX_OFFLINE_STORAGE_MB = 500
    MAX_OFFLINE_AGE_HOURS = 72
    SYNC_BATCH_SIZE = 50

    # Transmission settings
    MAX_QUEUE_SIZE = 5000
    BATCH_INTERVAL_SECONDS = 5.0
    RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 2.0


class RTAConfig:
    """RTA Dubai API configuration"""

    # API endpoints
    BASE_URL = os.environ.get('RTA_API_URL', 'https://api.rta.ae/traffic/v1')
    AUTH_URL = os.environ.get('RTA_AUTH_URL', 'https://auth.rta.ae/oauth2/token')
    HEALTH_CHECK_PATH = '/health'

    # Authentication
    CLIENT_ID = os.environ.get('RTA_CLIENT_ID', '')
    CLIENT_SECRET = os.environ.get('RTA_CLIENT_SECRET', '')
    API_KEY = os.environ.get('RTA_API_KEY', '')

    # Camera details
    INTERSECTION_NAME = os.environ.get('INTERSECTION_NAME', 'Traffic Intersection')
    COORDINATES = (
        float(os.environ.get('CAMERA_LAT', '25.2048')),   # Dubai default lat
        float(os.environ.get('CAMERA_LON', '55.2708'))    # Dubai default lon
    )

    # Request settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    BURST_LIMIT = 10

    # Heartbeat
    HEARTBEAT_INTERVAL = 60  # seconds

    # TLS/SSL
    VERIFY_SSL = True
    CERT_PATH = None

    # Fallback servers
    FALLBACK_SERVERS = [
        'https://backup1.rta.ae/traffic/v1',
        'https://backup2.rta.ae/traffic/v1'
    ]


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
    MODEL_NAME = "improved_model.h5"
    MODEL_PATH = MODEL_DIR / MODEL_NAME
    IMAGE_SIZE = (128, 128)  # Model input size
    
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