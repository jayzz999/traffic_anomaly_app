"""
Edge Processing Module for Raspberry Pi
Handles keyframe extraction, image compression, and adaptive processing
for network-resilient traffic monitoring with RTA Dubai integration.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import hashlib
import io
import base64
from enum import Enum


class CompressionLevel(Enum):
    """Compression levels for adaptive transmission"""
    FULL = "full"           # Original quality
    HIGH = "high"           # 80% quality, full resolution
    MEDIUM = "medium"       # 60% quality, 75% resolution
    LOW = "low"             # 40% quality, 50% resolution
    MINIMAL = "minimal"     # 20% quality, 25% resolution, grayscale


@dataclass
class ProcessedFrame:
    """Container for processed frame data"""
    frame_id: str
    timestamp: datetime
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    compression_level: CompressionLevel
    is_keyframe: bool
    data: bytes
    format: str  # 'jpeg' or 'webp'
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_bytes: int = 0
    compressed_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio"""
        if self.original_bytes == 0:
            return 0.0
        return 1 - (self.compressed_bytes / self.original_bytes)

    def to_base64(self) -> str:
        """Convert frame data to base64 string"""
        return base64.b64encode(self.data).decode('utf-8')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp.isoformat(),
            'original_size': self.original_size,
            'processed_size': self.processed_size,
            'compression_level': self.compression_level.value,
            'is_keyframe': self.is_keyframe,
            'data_base64': self.to_base64(),
            'format': self.format,
            'metadata': self.metadata,
            'compression_ratio': self.compression_ratio
        }


class KeyframeExtractor:
    """
    Extracts keyframes from video stream based on scene change detection
    and motion analysis for efficient bandwidth usage.
    """

    def __init__(
        self,
        scene_threshold: float = 30.0,
        motion_threshold: float = 5.0,
        min_keyframe_interval: int = 30,  # frames
        max_keyframe_interval: int = 150,  # frames
    ):
        self.scene_threshold = scene_threshold
        self.motion_threshold = motion_threshold
        self.min_keyframe_interval = min_keyframe_interval
        self.max_keyframe_interval = max_keyframe_interval

        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None
        self.frames_since_keyframe: int = 0
        self.keyframe_count: int = 0

    def reset(self):
        """Reset extractor state"""
        self.prev_frame_gray = None
        self.prev_hist = None
        self.frames_since_keyframe = 0

    def _compute_histogram(self, frame_gray: np.ndarray) -> np.ndarray:
        """Compute normalized histogram for scene change detection"""
        hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _compute_motion_score(self, frame_gray: np.ndarray) -> float:
        """Compute motion score using frame differencing"""
        if self.prev_frame_gray is None:
            return 0.0

        diff = cv2.absdiff(frame_gray, self.prev_frame_gray)
        motion_score = np.mean(diff)
        return float(motion_score)

    def _compute_scene_change_score(self, frame_gray: np.ndarray) -> float:
        """Compute scene change score using histogram comparison"""
        current_hist = self._compute_histogram(frame_gray)

        if self.prev_hist is None:
            self.prev_hist = current_hist
            return 0.0

        # Compare histograms using correlation
        correlation = cv2.compareHist(self.prev_hist, current_hist, cv2.HISTCMP_CORREL)
        scene_change_score = (1 - correlation) * 100  # Convert to percentage

        self.prev_hist = current_hist
        return float(scene_change_score)

    def is_keyframe(self, frame: np.ndarray, force: bool = False) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if frame should be extracted as keyframe.

        Args:
            frame: Input frame (BGR format)
            force: Force keyframe extraction

        Returns:
            Tuple of (is_keyframe, scores_dict)
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute scores
        motion_score = self._compute_motion_score(frame_gray)
        scene_change_score = self._compute_scene_change_score(frame_gray)

        scores = {
            'motion_score': motion_score,
            'scene_change_score': scene_change_score,
            'frames_since_keyframe': self.frames_since_keyframe
        }

        # Update previous frame
        self.prev_frame_gray = frame_gray.copy()
        self.frames_since_keyframe += 1

        # Determine if keyframe
        is_kf = False

        if force:
            is_kf = True
        elif self.frames_since_keyframe >= self.max_keyframe_interval:
            # Force keyframe after max interval
            is_kf = True
        elif self.frames_since_keyframe >= self.min_keyframe_interval:
            # Check for scene change or significant motion
            if scene_change_score > self.scene_threshold:
                is_kf = True
            elif motion_score > self.motion_threshold:
                is_kf = True

        if is_kf:
            self.frames_since_keyframe = 0
            self.keyframe_count += 1

        return is_kf, scores


class ImageCompressor:
    """
    Adaptive image compression for bandwidth-constrained transmission.
    Supports multiple compression levels and formats.
    """

    COMPRESSION_PARAMS = {
        CompressionLevel.FULL: {
            'quality': 95,
            'scale': 1.0,
            'grayscale': False
        },
        CompressionLevel.HIGH: {
            'quality': 80,
            'scale': 1.0,
            'grayscale': False
        },
        CompressionLevel.MEDIUM: {
            'quality': 60,
            'scale': 0.75,
            'grayscale': False
        },
        CompressionLevel.LOW: {
            'quality': 40,
            'scale': 0.5,
            'grayscale': False
        },
        CompressionLevel.MINIMAL: {
            'quality': 20,
            'scale': 0.25,
            'grayscale': True
        }
    }

    def __init__(self, default_format: str = 'jpeg'):
        """
        Initialize compressor.

        Args:
            default_format: Default image format ('jpeg' or 'webp')
        """
        self.default_format = default_format

    def compress(
        self,
        frame: np.ndarray,
        level: CompressionLevel = CompressionLevel.HIGH,
        format: Optional[str] = None
    ) -> Tuple[bytes, Tuple[int, int]]:
        """
        Compress frame according to specified level.

        Args:
            frame: Input frame (BGR format)
            level: Compression level
            format: Output format ('jpeg' or 'webp')

        Returns:
            Tuple of (compressed_bytes, new_size)
        """
        params = self.COMPRESSION_PARAMS[level]
        format = format or self.default_format

        # Scale image
        if params['scale'] < 1.0:
            new_width = int(frame.shape[1] * params['scale'])
            new_height = int(frame.shape[0] * params['scale'])
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale if needed
        if params['grayscale']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR for consistent encoding
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Encode
        if format == 'webp':
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, params['quality']]
            _, buffer = cv2.imencode('.webp', frame, encode_params)
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, params['quality']]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)

        return buffer.tobytes(), (frame.shape[1], frame.shape[0])

    def estimate_size(self, frame: np.ndarray, level: CompressionLevel) -> int:
        """Estimate compressed size without full compression"""
        params = self.COMPRESSION_PARAMS[level]
        # Rough estimation based on quality and scale
        original_size = frame.shape[0] * frame.shape[1] * frame.shape[2]
        estimated = original_size * params['scale']**2 * (params['quality'] / 100) * 0.1
        if params['grayscale']:
            estimated *= 0.33
        return int(estimated)


class EdgeProcessor:
    """
    Main edge processing class for Raspberry Pi deployment.
    Combines keyframe extraction, compression, and adaptive transmission.
    """

    def __init__(
        self,
        camera_id: str,
        location_id: str,
        scene_threshold: float = 30.0,
        motion_threshold: float = 5.0,
        min_keyframe_interval: int = 30,
        max_keyframe_interval: int = 150,
        default_compression: CompressionLevel = CompressionLevel.HIGH,
        default_format: str = 'jpeg'
    ):
        """
        Initialize edge processor.

        Args:
            camera_id: Unique camera identifier
            location_id: Location/intersection identifier for RTA
            scene_threshold: Threshold for scene change detection
            motion_threshold: Threshold for motion detection
            min_keyframe_interval: Minimum frames between keyframes
            max_keyframe_interval: Maximum frames between keyframes
            default_compression: Default compression level
            default_format: Default image format
        """
        self.camera_id = camera_id
        self.location_id = location_id
        self.default_compression = default_compression

        self.keyframe_extractor = KeyframeExtractor(
            scene_threshold=scene_threshold,
            motion_threshold=motion_threshold,
            min_keyframe_interval=min_keyframe_interval,
            max_keyframe_interval=max_keyframe_interval
        )

        self.compressor = ImageCompressor(default_format=default_format)

        # Statistics
        self.total_frames_processed: int = 0
        self.total_keyframes_extracted: int = 0
        self.total_bytes_original: int = 0
        self.total_bytes_compressed: int = 0
        self.processing_history: deque = deque(maxlen=1000)

    def _generate_frame_id(self, frame: np.ndarray, timestamp: datetime) -> str:
        """Generate unique frame ID"""
        frame_hash = hashlib.md5(frame.tobytes()[:1000]).hexdigest()[:8]
        return f"{self.camera_id}_{timestamp.strftime('%Y%m%d%H%M%S%f')}_{frame_hash}"

    def process_frame(
        self,
        frame: np.ndarray,
        compression_level: Optional[CompressionLevel] = None,
        force_keyframe: bool = False,
        anomaly_detected: bool = False,
        anomaly_class: Optional[str] = None,
        anomaly_confidence: Optional[float] = None
    ) -> Optional[ProcessedFrame]:
        """
        Process a single frame for transmission.

        Args:
            frame: Input frame (BGR format)
            compression_level: Override default compression level
            force_keyframe: Force this frame as keyframe
            anomaly_detected: Whether anomaly was detected in this frame
            anomaly_class: Detected anomaly class
            anomaly_confidence: Confidence of detection

        Returns:
            ProcessedFrame if frame should be transmitted, None otherwise
        """
        timestamp = datetime.now()
        self.total_frames_processed += 1

        # Check if this is a keyframe
        is_keyframe, scores = self.keyframe_extractor.is_keyframe(
            frame,
            force=force_keyframe or anomaly_detected
        )

        if not is_keyframe:
            return None

        self.total_keyframes_extracted += 1

        # Determine compression level
        level = compression_level or self.default_compression

        # If anomaly detected, use higher quality for critical events
        if anomaly_detected and anomaly_class in ['accident', 'fire']:
            level = CompressionLevel.HIGH

        # Calculate original size
        original_bytes = frame.shape[0] * frame.shape[1] * frame.shape[2]
        self.total_bytes_original += original_bytes

        # Compress frame
        compressed_data, new_size = self.compressor.compress(frame, level)
        compressed_bytes = len(compressed_data)
        self.total_bytes_compressed += compressed_bytes

        # Generate frame ID
        frame_id = self._generate_frame_id(frame, timestamp)

        # Build metadata
        metadata = {
            'camera_id': self.camera_id,
            'location_id': self.location_id,
            'keyframe_scores': scores,
            'anomaly_detected': anomaly_detected
        }

        if anomaly_detected:
            metadata['anomaly_class'] = anomaly_class
            metadata['anomaly_confidence'] = anomaly_confidence

        # Create processed frame
        processed = ProcessedFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            original_size=(frame.shape[1], frame.shape[0]),
            processed_size=new_size,
            compression_level=level,
            is_keyframe=True,
            data=compressed_data,
            format=self.compressor.default_format,
            metadata=metadata,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes
        )

        # Store in history
        self.processing_history.append({
            'frame_id': frame_id,
            'timestamp': timestamp,
            'compression_ratio': processed.compression_ratio,
            'is_anomaly': anomaly_detected
        })

        return processed

    def process_video_stream(
        self,
        video_source,
        model=None,
        predict_func=None,
        compression_level: Optional[CompressionLevel] = None,
        skip_frames: int = 2,
        max_frames: Optional[int] = None
    ):
        """
        Generator that processes video stream and yields keyframes.

        Args:
            video_source: OpenCV VideoCapture object or video path
            model: Anomaly detection model
            predict_func: Function to predict anomaly from frame
            compression_level: Compression level override
            skip_frames: Process every Nth frame
            max_frames: Maximum frames to process

        Yields:
            ProcessedFrame objects for keyframes
        """
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source

        frame_count = 0
        processed_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames
                if frame_count % skip_frames != 0:
                    continue

                if max_frames and processed_count >= max_frames:
                    break

                # Detect anomaly if model provided
                anomaly_detected = False
                anomaly_class = None
                anomaly_confidence = None

                if model is not None and predict_func is not None:
                    try:
                        prediction = predict_func(model, frame)
                        anomaly_class = prediction.get('class')
                        anomaly_confidence = prediction.get('confidence', 0)

                        # Consider it an anomaly if critical class detected
                        if anomaly_class in ['accident', 'fire'] and anomaly_confidence > 0.7:
                            anomaly_detected = True
                        elif anomaly_class == 'dense_traffic' and anomaly_confidence > 0.8:
                            anomaly_detected = True
                    except Exception:
                        pass

                # Process frame
                processed = self.process_frame(
                    frame,
                    compression_level=compression_level,
                    anomaly_detected=anomaly_detected,
                    anomaly_class=anomaly_class,
                    anomaly_confidence=anomaly_confidence
                )

                if processed is not None:
                    processed_count += 1
                    yield processed

        finally:
            if isinstance(video_source, str):
                cap.release()

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_compression = 0.0
        if self.total_bytes_original > 0:
            avg_compression = 1 - (self.total_bytes_compressed / self.total_bytes_original)

        keyframe_rate = 0.0
        if self.total_frames_processed > 0:
            keyframe_rate = self.total_keyframes_extracted / self.total_frames_processed

        return {
            'camera_id': self.camera_id,
            'location_id': self.location_id,
            'total_frames_processed': self.total_frames_processed,
            'total_keyframes_extracted': self.total_keyframes_extracted,
            'keyframe_extraction_rate': keyframe_rate,
            'total_bytes_original': self.total_bytes_original,
            'total_bytes_compressed': self.total_bytes_compressed,
            'average_compression_ratio': avg_compression,
            'bandwidth_saved_bytes': self.total_bytes_original - self.total_bytes_compressed
        }

    def reset_statistics(self):
        """Reset processing statistics"""
        self.total_frames_processed = 0
        self.total_keyframes_extracted = 0
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0
        self.processing_history.clear()
        self.keyframe_extractor.reset()
