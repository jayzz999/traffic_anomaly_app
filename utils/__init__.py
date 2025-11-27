"""
Utilities package for Traffic Anomaly Detection System
Includes Edge Processing and RTA Dubai Integration modules
"""

from .config import AppConfig, EdgeConfig, RTAConfig
from .model_utils import (
    load_trained_model,
    preprocess_image,
    predict_single_image,
    predict_batch_images,
    extract_wavelet_features,
    validate_image,
    get_model_summary,
)
from .video_utils import VideoAnomalyDetector
from .visualization import (
    create_probability_bar_chart,
    create_pie_chart,
    create_timeline_chart,
    create_confidence_trend,
    create_batch_summary_chart,
    create_heatmap,
    create_confusion_matrix,
    format_alert_message,
)

# Edge Processing modules
from .edge_processor import (
    EdgeProcessor,
    KeyframeExtractor,
    ImageCompressor,
    CompressionLevel,
    ProcessedFrame,
)
from .network_monitor import (
    NetworkMonitor,
    NetworkStatus,
    NetworkMetrics,
    NetworkConfig,
)
from .adaptive_transmitter import (
    AdaptiveTransmitter,
    TransmissionManager,
    TransmissionMode,
    TransmissionPolicy,
)
from .rta_client import (
    RTAClient,
    RTAIntegration,
    RTAEvent,
    RTAEventType,
    RTASeverity,
    RTAConfig as RTAClientConfig,
)
from .offline_storage import (
    OfflineStorage,
    OfflineSyncManager,
    StorageConfig,
)

__all__ = [
    # Configuration
    "AppConfig",
    "EdgeConfig",
    "RTAConfig",
    # Model utilities
    "load_trained_model",
    "preprocess_image",
    "predict_single_image",
    "predict_batch_images",
    "extract_wavelet_features",
    "validate_image",
    "get_model_summary",
    # Video processing
    "VideoAnomalyDetector",
    # Visualization
    "create_probability_bar_chart",
    "create_pie_chart",
    "create_timeline_chart",
    "create_confidence_trend",
    "create_batch_summary_chart",
    "create_heatmap",
    "create_confusion_matrix",
    "format_alert_message",
    # Edge Processing
    "EdgeProcessor",
    "KeyframeExtractor",
    "ImageCompressor",
    "CompressionLevel",
    "ProcessedFrame",
    # Network Monitoring
    "NetworkMonitor",
    "NetworkStatus",
    "NetworkMetrics",
    "NetworkConfig",
    # Adaptive Transmission
    "AdaptiveTransmitter",
    "TransmissionManager",
    "TransmissionMode",
    "TransmissionPolicy",
    # RTA Integration
    "RTAClient",
    "RTAIntegration",
    "RTAEvent",
    "RTAEventType",
    "RTASeverity",
    "RTAClientConfig",
    # Offline Storage
    "OfflineStorage",
    "OfflineSyncManager",
    "StorageConfig",
]
