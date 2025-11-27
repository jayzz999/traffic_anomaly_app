#!/usr/bin/env python3
"""
Edge Device Simulator
Run on your Mac to test the edge processing system before deploying to Raspberry Pi.
Uses real video files or your webcam - no fake data.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Reduce TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import AppConfig, EdgeConfig
from utils.model_utils import load_trained_model, preprocess_image, predict_single_image
from utils.edge_processor import EdgeProcessor, CompressionLevel
from utils.network_monitor import NetworkMonitor, NetworkConfig, NetworkStatus
from utils.offline_storage import OfflineStorage, StorageConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeSimulator:
    """
    Simulates edge device processing on your Mac.
    Uses real camera/video input and real model inference.
    """

    def __init__(
        self,
        source: str = "0",
        output_dir: str = "simulation_output",
        save_frames: bool = True,
        show_preview: bool = True
    ):
        """
        Initialize simulator.

        Args:
            source: Camera ID (0, 1, etc.) or path to video file
            output_dir: Directory to save results
            save_frames: Whether to save processed keyframes
            show_preview: Whether to show live preview window
        """
        self.source = source
        self.output_dir = Path(output_dir)
        self.save_frames = save_frames
        self.show_preview = show_preview

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "keyframes").mkdir(exist_ok=True)
        (self.output_dir / "anomalies").mkdir(exist_ok=True)

        # State
        self._running = False
        self._cap = None
        self._model = None

        # Statistics
        self.stats = {
            'start_time': None,
            'frames_read': 0,
            'frames_processed': 0,
            'keyframes_extracted': 0,
            'anomalies_detected': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'detections': {
                'accident': 0,
                'fire': 0,
                'dense_traffic': 0,
                'sparse_traffic': 0
            }
        }

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize processing components"""
        logger.info("Initializing simulator components...")

        # Load ML model
        logger.info("Loading anomaly detection model...")
        self._model = load_trained_model()
        logger.info("Model loaded successfully")

        # Initialize edge processor
        self.edge_processor = EdgeProcessor(
            camera_id="simulator-cam",
            location_id="mac-simulation",
            scene_threshold=EdgeConfig.SCENE_THRESHOLD,
            motion_threshold=EdgeConfig.MOTION_THRESHOLD,
            min_keyframe_interval=15,  # More frequent for testing
            max_keyframe_interval=60
        )

        # Initialize network monitor (to show real network status)
        network_config = NetworkConfig(
            primary_server="https://httpbin.org",
            check_interval_seconds=10.0
        )
        self.network_monitor = NetworkMonitor(config=network_config)

        # Initialize offline storage (for testing storage functionality)
        storage_config = StorageConfig(
            db_path=str(self.output_dir / "simulation.db"),
            max_storage_mb=100
        )
        self.offline_storage = OfflineStorage(config=storage_config)

        logger.info("All components initialized")

    def _open_source(self) -> bool:
        """Open video source"""
        try:
            # Determine source type
            if self.source.isdigit():
                source = int(self.source)
                logger.info(f"Opening webcam {source}...")
            else:
                source = self.source
                if not os.path.exists(source):
                    logger.error(f"Video file not found: {source}")
                    return False
                logger.info(f"Opening video file: {source}")

            self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error("Failed to open video source")
                return False

            # Get video info
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30

            logger.info(f"Video source opened: {width}x{height} @ {fps:.1f} FPS")
            return True

        except Exception as e:
            logger.error(f"Error opening source: {e}")
            return False

    def _process_frame(self, frame: np.ndarray) -> dict:
        """Run anomaly detection on frame"""
        try:
            # predict_single_image handles preprocessing internally
            prediction = predict_single_image(self._model, frame)
            # Convert confidence from percentage (0-100) to decimal (0-1)
            prediction['confidence'] = prediction['confidence'] / 100.0
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'class': 'unknown', 'confidence': 0.0}

    def _annotate_frame(self, frame: np.ndarray, prediction: dict, is_keyframe: bool) -> np.ndarray:
        """Add annotations to frame for display"""
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # Get prediction info
        pred_class = prediction.get('class', 'unknown')
        confidence = prediction.get('confidence', 0)

        # Color based on class
        colors = {
            'accident': (0, 0, 255),      # Red
            'fire': (0, 100, 255),         # Orange
            'dense_traffic': (0, 255, 255), # Yellow
            'sparse_traffic': (0, 255, 0)   # Green
        }
        color = colors.get(pred_class, (255, 255, 255))

        # Draw prediction box
        cv2.rectangle(annotated, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (350, 120), color, 2)

        # Add text
        cv2.putText(annotated, f"Class: {pred_class.upper()}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Confidence: {confidence:.1%}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Keyframe indicator
        if is_keyframe:
            cv2.putText(annotated, "KEYFRAME",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Anomaly alert
        if pred_class in ['accident', 'fire'] and confidence > 0.7:
            cv2.rectangle(annotated, (0, 0), (width, height), (0, 0, 255), 10)
            cv2.putText(annotated, "! ANOMALY DETECTED !",
                        (width//2 - 150, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Stats overlay
        cv2.rectangle(annotated, (width - 250, 10), (width - 10, 80), (0, 0, 0), -1)
        cv2.putText(annotated, f"Frames: {self.stats['frames_processed']}",
                    (width - 240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Keyframes: {self.stats['keyframes_extracted']}",
                    (width - 240, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Anomalies: {self.stats['anomalies_detected']}",
                    (width - 240, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def _save_keyframe(self, frame: np.ndarray, processed_frame, prediction: dict, is_anomaly: bool):
        """Save keyframe to disk"""
        if not self.save_frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pred_class = prediction.get('class', 'unknown')

        # Choose directory
        if is_anomaly:
            save_dir = self.output_dir / "anomalies"
        else:
            save_dir = self.output_dir / "keyframes"

        # Save original frame
        filename = f"{timestamp}_{pred_class}_{prediction.get('confidence', 0):.0%}.jpg"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), frame)

        # Also save the compressed version
        compressed_path = save_dir / f"{timestamp}_compressed.jpg"
        with open(compressed_path, 'wb') as f:
            f.write(processed_frame.data)

        logger.debug(f"Saved keyframe: {filename}")

    def run(self):
        """Main simulation loop"""
        logger.info("=" * 60)
        logger.info("  EDGE DEVICE SIMULATOR")
        logger.info("=" * 60)

        if not self._open_source():
            return

        # Start network monitoring
        self.network_monitor.start_monitoring()

        self.stats['start_time'] = datetime.now()
        self._running = True

        logger.info("")
        logger.info("Starting simulation...")
        logger.info("Press 'q' to quit, 's' to save current frame, 'n' to check network")
        logger.info("")

        frame_count = 0
        last_log_time = time.time()

        try:
            while self._running:
                ret, frame = self._cap.read()

                if not ret:
                    if not self.source.isdigit():
                        logger.info("End of video file")
                        break
                    else:
                        logger.warning("Failed to read frame")
                        continue

                frame_count += 1
                self.stats['frames_read'] = frame_count

                # Skip frames for performance (process every 2nd frame)
                if frame_count % 2 != 0:
                    continue

                self.stats['frames_processed'] += 1

                # Run anomaly detection
                prediction = self._process_frame(frame)
                pred_class = prediction.get('class', 'unknown')
                confidence = prediction.get('confidence', 0)

                # Update detection stats
                if pred_class in self.stats['detections']:
                    self.stats['detections'][pred_class] += 1

                # Check for anomaly
                is_anomaly = False
                if pred_class in ['accident', 'fire'] and confidence > 0.7:
                    is_anomaly = True
                    self.stats['anomalies_detected'] += 1
                    logger.warning(f"ANOMALY: {pred_class} detected with {confidence:.1%} confidence!")

                # Get compression level based on network
                network_status = self.network_monitor.status
                compression_map = {
                    NetworkStatus.EXCELLENT: CompressionLevel.HIGH,
                    NetworkStatus.GOOD: CompressionLevel.HIGH,
                    NetworkStatus.FAIR: CompressionLevel.MEDIUM,
                    NetworkStatus.POOR: CompressionLevel.LOW,
                    NetworkStatus.CRITICAL: CompressionLevel.MINIMAL,
                    NetworkStatus.OFFLINE: CompressionLevel.LOW
                }
                compression = compression_map.get(network_status, CompressionLevel.MEDIUM)

                # Process through edge processor
                processed_frame = self.edge_processor.process_frame(
                    frame,
                    compression_level=compression,
                    force_keyframe=is_anomaly,
                    anomaly_detected=is_anomaly,
                    anomaly_class=pred_class,
                    anomaly_confidence=confidence
                )

                is_keyframe = processed_frame is not None

                if is_keyframe:
                    self.stats['keyframes_extracted'] += 1
                    self.stats['total_original_bytes'] += processed_frame.original_bytes
                    self.stats['total_compressed_bytes'] += processed_frame.compressed_bytes

                    # Save to offline storage
                    self.offline_storage.store_frame(
                        processed_frame.to_dict(),
                        is_priority=is_anomaly
                    )

                    # Save to disk
                    self._save_keyframe(frame, processed_frame, prediction, is_anomaly)

                # Show preview
                if self.show_preview:
                    annotated = self._annotate_frame(frame, prediction, is_keyframe)
                    cv2.imshow("Edge Simulator", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested")
                        break
                    elif key == ord('s'):
                        # Force save current frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(str(self.output_dir / f"manual_{timestamp}.jpg"), frame)
                        logger.info("Frame saved manually")
                    elif key == ord('n'):
                        # Show network status
                        metrics = self.network_monitor.metrics
                        if metrics:
                            logger.info(f"Network: {metrics.status.value}, Latency: {metrics.latency_ms:.0f}ms")

                # Log stats periodically
                if time.time() - last_log_time > 5.0:
                    self._log_stats()
                    last_log_time = time.time()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    def _log_stats(self):
        """Log current statistics"""
        compression_ratio = 0
        if self.stats['total_original_bytes'] > 0:
            compression_ratio = 1 - (self.stats['total_compressed_bytes'] / self.stats['total_original_bytes'])

        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0

        logger.info(
            f"Stats: {self.stats['frames_processed']} frames, "
            f"{self.stats['keyframes_extracted']} keyframes, "
            f"{self.stats['anomalies_detected']} anomalies, "
            f"Compression: {compression_ratio:.1%}, "
            f"FPS: {fps:.1f}"
        )

    def _cleanup(self):
        """Cleanup resources"""
        self._running = False

        if self._cap:
            self._cap.release()

        if self.show_preview:
            cv2.destroyAllWindows()

        self.network_monitor.stop_monitoring()

        # Print final summary
        self._print_summary()

    def _print_summary(self):
        """Print final simulation summary"""
        print("\n" + "=" * 60)
        print("  SIMULATION SUMMARY")
        print("=" * 60)

        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()

        print(f"\nDuration: {elapsed:.1f} seconds")
        print(f"Frames read: {self.stats['frames_read']}")
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Keyframes extracted: {self.stats['keyframes_extracted']}")
        print(f"Anomalies detected: {self.stats['anomalies_detected']}")

        if self.stats['total_original_bytes'] > 0:
            compression = 1 - (self.stats['total_compressed_bytes'] / self.stats['total_original_bytes'])
            print(f"\nCompression ratio: {compression:.1%}")
            print(f"Original data: {self.stats['total_original_bytes'] / 1024 / 1024:.2f} MB")
            print(f"Compressed data: {self.stats['total_compressed_bytes'] / 1024 / 1024:.2f} MB")
            print(f"Bandwidth saved: {(self.stats['total_original_bytes'] - self.stats['total_compressed_bytes']) / 1024 / 1024:.2f} MB")

        print(f"\nDetections by class:")
        for cls, count in self.stats['detections'].items():
            print(f"  {cls}: {count}")

        # Storage stats
        storage_stats = self.offline_storage.get_statistics()
        print(f"\nOffline storage:")
        print(f"  Frames stored: {storage_stats.get('pending_frames', 0)}")
        print(f"  Database size: {storage_stats.get('db_size_mb', 0):.2f} MB")

        print(f"\nOutput saved to: {self.output_dir.absolute()}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Edge Device Simulator - Test on Mac before deploying to Raspberry Pi'
    )

    parser.add_argument(
        '--source', '-s',
        default='0',
        help='Video source: webcam ID (0, 1, ...) or path to video file'
    )
    parser.add_argument(
        '--output', '-o',
        default='simulation_output',
        help='Output directory for saved frames'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save keyframes to disk'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Do not show preview window'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TRAFFIC ANOMALY DETECTION - EDGE SIMULATOR")
    print("=" * 60)
    print(f"\nSource: {args.source}")
    print(f"Output: {args.output}")
    print(f"Save frames: {not args.no_save}")
    print(f"Show preview: {not args.no_preview}")
    print("")

    simulator = EdgeSimulator(
        source=args.source,
        output_dir=args.output,
        save_frames=not args.no_save,
        show_preview=not args.no_preview
    )

    simulator.run()


if __name__ == '__main__':
    main()
