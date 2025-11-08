"""
Pages package for Traffic Anomaly Detection System
"""

from . import image_detection
from . import video_detection
from . import batch_analysis
from . import realtime_stream
from . import analytics_dashboard

__all__ = [
    'image_detection',
    'video_detection',
    'batch_analysis',
    'realtime_stream',
    'analytics_dashboard'
]