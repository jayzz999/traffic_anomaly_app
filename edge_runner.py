#!/usr/bin/env python3
"""
Edge Runner for Raspberry Pi
Main entry point for edge deployment with RTA Dubai integration.
Handles camera input, anomaly detection, and adaptive transmission.
"""

import os
import sys
import time
import signal
import logging
import argparse
from typing import Optional
from datetime import datetime
from pathlib import Path

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU on Raspberry Pi

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import AppConfig, EdgeConfig, RTAConfig
from utils.model_utils import load_trained_model, preprocess_image, predict_single_image
from utils.edge_processor import EdgeProcessor, CompressionLevel
from utils.network_monitor import NetworkMonitor, NetworkConfig, NetworkStatus
from utils.adaptive_transmitter import AdaptiveTransmitter, TransmissionMode
from utils.rta_client import RTAClient, RTAConfig as RTAClientConfig, RTAIntegration
from utils.offline_storage import OfflineStorage, OfflineSyncManager, StorageConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('edge_runner.log')
    ]
)
logger = logging.getLogger(__name__)


class EdgeRunner:
    """
    Main edge runner class for Raspberry Pi deployment.
    Coordinates all edge processing components.
    """

    def __init__(
        self,
        camera_source: str = "0",
        camera_id: Optional[str] = None,
        location_id: Optional[str] = None,
        rta_api_url: Optional[str] = None,
        rta_api_key: Optional[str] = None
    ):
        """
        Initialize edge runner.

        Args:
            camera_source: Camera source (device ID, RTSP URL, or video file)
            camera_id: Unique camera identifier
            location_id: Location/intersection identifier
            rta_api_url: RTA API base URL
            rta_api_key: RTA API key
        """
        self.camera_source = camera_source
        self.camera_id = camera_id or EdgeConfig.CAMERA_ID
        self.location_id = location_id or EdgeConfig.LOCATION_ID
        self.rta_api_url = rta_api_url or RTAConfig.BASE_URL
        self.rta_api_key = rta_api_key or RTAConfig.API_KEY

        # State
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._model = None

        # Statistics
        self._stats = {
            'start_time': None,
            'frames_processed': 0,
            'anomalies_detected': 0,
            'frames_transmitted': 0,
            'current_fps': 0.0
        }

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all edge processing components"""
        logger.info("Initializing edge runner components...")

        # Load ML model
        logger.info("Loading anomaly detection model...")
        try:
            self._model = load_trained_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Initialize edge processor
        self.edge_processor = EdgeProcessor(
            camera_id=self.camera_id,
            location_id=self.location_id,
            scene_threshold=EdgeConfig.SCENE_THRESHOLD,
            motion_threshold=EdgeConfig.MOTION_THRESHOLD,
            min_keyframe_interval=EdgeConfig.MIN_KEYFRAME_INTERVAL,
            max_keyframe_interval=EdgeConfig.MAX_KEYFRAME_INTERVAL,
            default_format=EdgeConfig.DEFAULT_FORMAT
        )

        # Initialize network monitor
        network_config = NetworkConfig(
            primary_server=self.rta_api_url,
            check_interval_seconds=EdgeConfig.NETWORK_CHECK_INTERVAL,
            timeout_seconds=EdgeConfig.NETWORK_TIMEOUT,
            ping_hosts=EdgeConfig.PING_HOSTS,
            latency_excellent_ms=EdgeConfig.LATENCY_EXCELLENT,
            latency_good_ms=EdgeConfig.LATENCY_GOOD,
            latency_fair_ms=EdgeConfig.LATENCY_FAIR,
            latency_poor_ms=EdgeConfig.LATENCY_POOR,
            bandwidth_excellent_kbps=EdgeConfig.BANDWIDTH_EXCELLENT,
            bandwidth_good_kbps=EdgeConfig.BANDWIDTH_GOOD,
            bandwidth_fair_kbps=EdgeConfig.BANDWIDTH_FAIR,
            bandwidth_poor_kbps=EdgeConfig.BANDWIDTH_POOR
        )

        self.network_monitor = NetworkMonitor(
            config=network_config,
            on_status_change=self._on_network_status_change,
            on_offline=self._on_network_offline,
            on_reconnect=self._on_network_reconnect
        )

        # Initialize RTA client
        rta_config = RTAClientConfig(
            base_url=self.rta_api_url,
            api_key=self.rta_api_key,
            camera_id=self.camera_id,
            location_id=self.location_id,
            intersection_name=RTAConfig.INTERSECTION_NAME,
            coordinates=RTAConfig.COORDINATES
        )
        self.rta_client = RTAClient(rta_config)

        # Initialize offline storage
        storage_config = StorageConfig(
            db_path=EdgeConfig.OFFLINE_DB_PATH,
            max_storage_mb=EdgeConfig.MAX_OFFLINE_STORAGE_MB,
            max_age_hours=EdgeConfig.MAX_OFFLINE_AGE_HOURS,
            batch_size=EdgeConfig.SYNC_BATCH_SIZE
        )
        self.offline_storage = OfflineStorage(config=storage_config)

        # Initialize sync manager
        self.sync_manager = OfflineSyncManager(
            storage=self.offline_storage,
            send_frames_func=self._send_frames_to_rta,
            send_events_func=self._send_events_to_rta,
            config=storage_config
        )

        # Initialize adaptive transmitter
        self.transmitter = AdaptiveTransmitter(
            network_monitor=self.network_monitor,
            send_func=self._send_frames_to_rta,
            on_mode_change=self._on_transmission_mode_change
        )

        logger.info("All components initialized successfully")

    def _on_network_status_change(self, old_status: NetworkStatus, new_status: NetworkStatus):
        """Handle network status change"""
        logger.info(f"Network status changed: {old_status.value} -> {new_status.value}")

        # Update sync manager network availability
        is_available = new_status not in [NetworkStatus.OFFLINE, NetworkStatus.CRITICAL]
        self.sync_manager.set_network_available(is_available)

    def _on_network_offline(self):
        """Handle network going offline"""
        logger.warning("Network went offline - switching to local storage mode")

    def _on_network_reconnect(self):
        """Handle network reconnection"""
        logger.info("Network reconnected - initiating sync")
        self.sync_manager.force_sync()

    def _on_transmission_mode_change(self, old_mode: TransmissionMode, new_mode: TransmissionMode):
        """Handle transmission mode change"""
        logger.info(f"Transmission mode changed: {old_mode.value} -> {new_mode.value}")

    def _send_frames_to_rta(self, frames: list) -> bool:
        """Send frames to RTA server"""
        try:
            success, _ = self.rta_client.upload_frames(frames)
            if success:
                self._stats['frames_transmitted'] += len(frames)
            return success
        except Exception as e:
            logger.error(f"Failed to send frames to RTA: {e}")
            return False

    def _send_events_to_rta(self, events: list) -> bool:
        """Send events to RTA server"""
        try:
            for event in events:
                success, _ = self.rta_client.report_event(event)
                if not success:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to send events to RTA: {e}")
            return False

    def _open_camera(self) -> bool:
        """Open camera source"""
        try:
            # Determine camera source type
            if self.camera_source.isdigit():
                source = int(self.camera_source)
            elif self.camera_source.startswith(('rtsp://', 'http://', 'https://')):
                source = self.camera_source
            else:
                source = self.camera_source  # File path

            self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error(f"Failed to open camera source: {self.camera_source}")
                return False

            # Configure camera
            if isinstance(source, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info(f"Camera opened successfully: {self.camera_source}")
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def _process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the anomaly detection pipeline.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Prediction result dictionary
        """
        try:
            # Preprocess for model
            processed = preprocess_image(frame, AppConfig.IMAGE_SIZE)

            # Run inference
            prediction = predict_single_image(self._model, processed)

            return prediction

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {'class': 'unknown', 'confidence': 0.0}

    def _handle_anomaly(self, frame: np.ndarray, prediction: dict):
        """Handle detected anomaly"""
        anomaly_class = prediction.get('class')
        confidence = prediction.get('confidence', 0)

        self._stats['anomalies_detected'] += 1

        logger.warning(
            f"ANOMALY DETECTED: {anomaly_class} "
            f"(confidence: {confidence:.2%})"
        )

        # Create and report RTA event
        if self.network_monitor.is_online:
            try:
                event = self.rta_client.create_event_from_detection(
                    anomaly_class=anomaly_class,
                    confidence=confidence,
                    metadata={'frame_count': self._stats['frames_processed']}
                )
                success, _ = self.rta_client.report_event(event)

                if not success:
                    # Store offline if failed
                    self.offline_storage.store_event(event.to_dict())

            except Exception as e:
                logger.error(f"Failed to report anomaly: {e}")
        else:
            # Store offline
            event = self.rta_client.create_event_from_detection(
                anomaly_class=anomaly_class,
                confidence=confidence
            )
            self.offline_storage.store_event(event.to_dict())

    def run(self):
        """Main run loop"""
        logger.info("Starting edge runner...")
        self._stats['start_time'] = datetime.now()

        # Open camera
        if not self._open_camera():
            logger.error("Failed to open camera. Exiting.")
            return

        # Start background services
        self.network_monitor.start_monitoring()
        self.transmitter.start()
        self.sync_manager.start()

        # Send initial heartbeat
        self.rta_client.send_heartbeat()

        self._running = True
        frame_count = 0
        last_fps_time = time.time()
        fps_frame_count = 0

        logger.info("Edge runner started. Processing frames...")

        try:
            while self._running:
                ret, frame = self._cap.read()

                if not ret:
                    if isinstance(self.camera_source, str) and not self.camera_source.isdigit():
                        # Video file ended
                        logger.info("Video file ended")
                        break
                    else:
                        # Camera error, try to reconnect
                        logger.warning("Camera read failed, attempting reconnect...")
                        time.sleep(1)
                        self._open_camera()
                        continue

                frame_count += 1
                fps_frame_count += 1
                self._stats['frames_processed'] = frame_count

                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self._stats['current_fps'] = fps_frame_count / (current_time - last_fps_time)
                    fps_frame_count = 0
                    last_fps_time = current_time

                # Skip frames based on configuration
                if frame_count % AppConfig.SKIP_FRAMES != 0:
                    continue

                # Process frame for anomaly detection
                prediction = self._process_frame(frame)
                anomaly_class = prediction.get('class')
                confidence = prediction.get('confidence', 0)

                # Check for anomaly
                is_anomaly = False
                if anomaly_class in ['accident', 'fire'] and confidence > 0.7:
                    is_anomaly = True
                    self._handle_anomaly(frame, prediction)
                elif anomaly_class == 'dense_traffic' and confidence > 0.8:
                    is_anomaly = True
                    self._handle_anomaly(frame, prediction)

                # Get current compression level based on network status
                compression_level = self.transmitter.get_compression_level()

                # Process frame for transmission
                processed_frame = self.edge_processor.process_frame(
                    frame,
                    compression_level=compression_level,
                    anomaly_detected=is_anomaly,
                    anomaly_class=anomaly_class,
                    anomaly_confidence=confidence
                )

                if processed_frame:
                    # Queue for transmission
                    is_priority = is_anomaly and anomaly_class in ['accident', 'fire']

                    if self.network_monitor.is_online:
                        self.transmitter.queue_frame(processed_frame, is_priority=is_priority)
                    else:
                        # Store offline
                        self.offline_storage.store_frame(
                            processed_frame.to_dict(),
                            is_priority=is_priority
                        )

                # Periodic heartbeat
                if frame_count % 1000 == 0:
                    self.rta_client.send_heartbeat()

                # Log status periodically
                if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    self._log_status()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()

    def _log_status(self):
        """Log current status"""
        network_stats = self.network_monitor.get_statistics()
        transmit_stats = self.transmitter.get_statistics()
        edge_stats = self.edge_processor.get_statistics()
        storage_stats = self.offline_storage.get_statistics()

        logger.info(
            f"Status: "
            f"Frames={self._stats['frames_processed']}, "
            f"FPS={self._stats['current_fps']:.1f}, "
            f"Anomalies={self._stats['anomalies_detected']}, "
            f"Transmitted={self._stats['frames_transmitted']}, "
            f"Network={network_stats.get('current_status', 'unknown')}, "
            f"Queue={transmit_stats.get('priority_queue_size', 0) + transmit_stats.get('normal_queue_size', 0)}, "
            f"Offline={storage_stats.get('pending_frames', 0)}"
        )

    def stop(self):
        """Stop the edge runner"""
        logger.info("Stopping edge runner...")
        self._running = False

        # Stop components
        self.transmitter.stop()
        self.sync_manager.stop()
        self.network_monitor.stop_monitoring()

        # Release camera
        if self._cap:
            self._cap.release()

        # Final status
        self._log_status()
        logger.info("Edge runner stopped")

    def get_status(self) -> dict:
        """Get complete status"""
        return {
            'runner': self._stats,
            'network': self.network_monitor.get_statistics(),
            'transmission': self.transmitter.get_statistics(),
            'edge_processing': self.edge_processor.get_statistics(),
            'offline_storage': self.offline_storage.get_statistics(),
            'rta_client': self.rta_client.get_statistics()
        }


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    if hasattr(signal_handler, 'runner'):
        signal_handler.runner.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Edge Runner for Traffic Anomaly Detection with RTA Integration'
    )

    parser.add_argument(
        '--camera', '-c',
        default='0',
        help='Camera source (device ID, RTSP URL, or video file path)'
    )
    parser.add_argument(
        '--camera-id',
        default=None,
        help='Unique camera identifier'
    )
    parser.add_argument(
        '--location-id',
        default=None,
        help='Location/intersection identifier'
    )
    parser.add_argument(
        '--rta-url',
        default=None,
        help='RTA API base URL'
    )
    parser.add_argument(
        '--rta-key',
        default=None,
        help='RTA API key'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run edge runner
    runner = EdgeRunner(
        camera_source=args.camera,
        camera_id=args.camera_id,
        location_id=args.location_id,
        rta_api_url=args.rta_url,
        rta_api_key=args.rta_key
    )

    signal_handler.runner = runner

    try:
        runner.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
