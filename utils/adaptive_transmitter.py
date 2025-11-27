"""
Adaptive Transmission Module
Automatically adjusts image quality and transmission strategy based on network conditions.
Designed for Raspberry Pi edge deployment with RTA Dubai integration.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from queue import Queue, Empty, Full
from enum import Enum

from .network_monitor import NetworkMonitor, NetworkStatus, NetworkConfig
from .edge_processor import (
    EdgeProcessor,
    ProcessedFrame,
    CompressionLevel,
)

logger = logging.getLogger(__name__)


class TransmissionMode(Enum):
    """Transmission mode based on network conditions"""
    REALTIME = "realtime"           # Send immediately, full quality
    OPTIMIZED = "optimized"         # Send with moderate compression
    DEGRADED = "degraded"           # Send with heavy compression
    BATCH = "batch"                 # Queue and batch send
    OFFLINE = "offline"             # Store locally only


@dataclass
class TransmissionPolicy:
    """Policy for each transmission mode"""
    mode: TransmissionMode
    compression_level: CompressionLevel
    max_queue_size: int
    batch_size: int
    batch_interval_seconds: float
    priority_only: bool  # Only send high-priority (anomaly) frames
    retry_attempts: int
    retry_delay_seconds: float


class AdaptiveTransmitter:
    """
    Manages adaptive transmission of processed frames based on network conditions.
    Automatically switches between transmission modes and handles queuing/batching.
    """

    # Default policies for each network status
    DEFAULT_POLICIES = {
        NetworkStatus.EXCELLENT: TransmissionPolicy(
            mode=TransmissionMode.REALTIME,
            compression_level=CompressionLevel.HIGH,
            max_queue_size=100,
            batch_size=1,
            batch_interval_seconds=0,
            priority_only=False,
            retry_attempts=3,
            retry_delay_seconds=1.0
        ),
        NetworkStatus.GOOD: TransmissionPolicy(
            mode=TransmissionMode.REALTIME,
            compression_level=CompressionLevel.HIGH,
            max_queue_size=200,
            batch_size=1,
            batch_interval_seconds=0,
            priority_only=False,
            retry_attempts=3,
            retry_delay_seconds=2.0
        ),
        NetworkStatus.FAIR: TransmissionPolicy(
            mode=TransmissionMode.OPTIMIZED,
            compression_level=CompressionLevel.MEDIUM,
            max_queue_size=500,
            batch_size=5,
            batch_interval_seconds=5.0,
            priority_only=False,
            retry_attempts=2,
            retry_delay_seconds=5.0
        ),
        NetworkStatus.POOR: TransmissionPolicy(
            mode=TransmissionMode.DEGRADED,
            compression_level=CompressionLevel.LOW,
            max_queue_size=1000,
            batch_size=10,
            batch_interval_seconds=15.0,
            priority_only=True,
            retry_attempts=2,
            retry_delay_seconds=10.0
        ),
        NetworkStatus.CRITICAL: TransmissionPolicy(
            mode=TransmissionMode.BATCH,
            compression_level=CompressionLevel.MINIMAL,
            max_queue_size=2000,
            batch_size=20,
            batch_interval_seconds=30.0,
            priority_only=True,
            retry_attempts=1,
            retry_delay_seconds=30.0
        ),
        NetworkStatus.OFFLINE: TransmissionPolicy(
            mode=TransmissionMode.OFFLINE,
            compression_level=CompressionLevel.LOW,
            max_queue_size=5000,
            batch_size=50,
            batch_interval_seconds=60.0,
            priority_only=True,
            retry_attempts=0,
            retry_delay_seconds=0
        )
    }

    def __init__(
        self,
        network_monitor: NetworkMonitor,
        send_func: Callable[[List[Dict[str, Any]]], bool],
        policies: Optional[Dict[NetworkStatus, TransmissionPolicy]] = None,
        on_mode_change: Optional[Callable[[TransmissionMode, TransmissionMode], None]] = None,
        on_send_success: Optional[Callable[[int], None]] = None,
        on_send_failure: Optional[Callable[[int, str], None]] = None
    ):
        """
        Initialize adaptive transmitter.

        Args:
            network_monitor: Network monitor instance
            send_func: Function to send frames to server (takes list of frame dicts, returns success)
            policies: Custom policies for network statuses
            on_mode_change: Callback when transmission mode changes
            on_send_success: Callback on successful send (frame count)
            on_send_failure: Callback on failed send (frame count, error)
        """
        self.network_monitor = network_monitor
        self.send_func = send_func
        self.policies = policies or self.DEFAULT_POLICIES.copy()
        self.on_mode_change = on_mode_change
        self.on_send_success = on_send_success
        self.on_send_failure = on_send_failure

        # Current state
        self._current_mode = TransmissionMode.OFFLINE
        self._current_policy = self.policies[NetworkStatus.OFFLINE]

        # Queues
        self._priority_queue: deque = deque(maxlen=1000)  # High priority (anomalies)
        self._normal_queue: deque = deque(maxlen=5000)    # Normal frames

        # Statistics
        self._stats = {
            'frames_queued': 0,
            'frames_sent': 0,
            'frames_dropped': 0,
            'bytes_sent': 0,
            'send_failures': 0,
            'mode_changes': 0
        }

        # Threading
        self._running = False
        self._transmit_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Register network status callback
        self.network_monitor.on_status_change = self._on_network_status_change

    @property
    def current_mode(self) -> TransmissionMode:
        """Get current transmission mode"""
        with self._lock:
            return self._current_mode

    @property
    def current_policy(self) -> TransmissionPolicy:
        """Get current transmission policy"""
        with self._lock:
            return self._current_policy

    @property
    def queue_size(self) -> int:
        """Get total queue size"""
        with self._lock:
            return len(self._priority_queue) + len(self._normal_queue)

    def _on_network_status_change(self, old_status: NetworkStatus, new_status: NetworkStatus):
        """Handle network status change"""
        new_policy = self.policies.get(new_status, self.policies[NetworkStatus.OFFLINE])

        with self._lock:
            old_mode = self._current_mode
            self._current_mode = new_policy.mode
            self._current_policy = new_policy

        if old_mode != new_policy.mode:
            self._stats['mode_changes'] += 1
            logger.info(f"Transmission mode changed: {old_mode.value} -> {new_policy.mode.value}")

            if self.on_mode_change:
                try:
                    self.on_mode_change(old_mode, new_policy.mode)
                except Exception as e:
                    logger.error(f"Mode change callback error: {e}")

    def get_compression_level(self) -> CompressionLevel:
        """Get current compression level based on network status"""
        return self.current_policy.compression_level

    def queue_frame(self, frame: ProcessedFrame, is_priority: bool = False) -> bool:
        """
        Add frame to transmission queue.

        Args:
            frame: Processed frame to queue
            is_priority: Whether this is a high-priority frame (anomaly)

        Returns:
            True if queued successfully, False if dropped
        """
        policy = self.current_policy

        # In offline mode with priority_only, drop non-priority frames
        if policy.priority_only and not is_priority:
            self._stats['frames_dropped'] += 1
            return False

        frame_dict = frame.to_dict()

        with self._lock:
            if is_priority:
                if len(self._priority_queue) >= policy.max_queue_size // 2:
                    self._priority_queue.popleft()
                    self._stats['frames_dropped'] += 1
                self._priority_queue.append(frame_dict)
            else:
                if len(self._normal_queue) >= policy.max_queue_size:
                    self._normal_queue.popleft()
                    self._stats['frames_dropped'] += 1
                self._normal_queue.append(frame_dict)

            self._stats['frames_queued'] += 1

        return True

    def _get_batch(self) -> List[Dict[str, Any]]:
        """Get next batch of frames to send"""
        policy = self.current_policy
        batch = []

        with self._lock:
            # Priority frames first
            while self._priority_queue and len(batch) < policy.batch_size:
                batch.append(self._priority_queue.popleft())

            # Then normal frames if space and not priority_only mode
            if not policy.priority_only:
                while self._normal_queue and len(batch) < policy.batch_size:
                    batch.append(self._normal_queue.popleft())

        return batch

    def _send_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Send a batch of frames with retry logic"""
        if not batch:
            return True

        policy = self.current_policy

        for attempt in range(max(1, policy.retry_attempts)):
            try:
                success = self.send_func(batch)

                if success:
                    self._stats['frames_sent'] += len(batch)
                    self._stats['bytes_sent'] += sum(
                        len(f.get('data_base64', '')) for f in batch
                    )

                    if self.on_send_success:
                        try:
                            self.on_send_success(len(batch))
                        except Exception:
                            pass

                    return True

            except Exception as e:
                logger.warning(f"Send attempt {attempt + 1} failed: {e}")

            if attempt < policy.retry_attempts - 1:
                time.sleep(policy.retry_delay_seconds)

        # All retries failed
        self._stats['send_failures'] += 1

        if self.on_send_failure:
            try:
                self.on_send_failure(len(batch), "Max retries exceeded")
            except Exception:
                pass

        # Re-queue failed frames (priority only in degraded conditions)
        if policy.mode not in [TransmissionMode.OFFLINE, TransmissionMode.BATCH]:
            with self._lock:
                for frame in batch:
                    if frame.get('metadata', {}).get('anomaly_detected'):
                        self._priority_queue.appendleft(frame)

        return False

    def _transmit_loop(self):
        """Background transmission loop"""
        last_batch_time = datetime.now()

        while self._running:
            try:
                policy = self.current_policy

                # Check if it's time to send a batch
                should_send = False
                elapsed = (datetime.now() - last_batch_time).total_seconds()

                if policy.mode == TransmissionMode.OFFLINE:
                    # Don't send in offline mode
                    time.sleep(1)
                    continue
                elif policy.mode == TransmissionMode.REALTIME:
                    # Send immediately if there are frames
                    should_send = self.queue_size > 0
                else:
                    # Batch mode - wait for interval or batch size
                    should_send = (
                        elapsed >= policy.batch_interval_seconds or
                        self.queue_size >= policy.batch_size
                    )

                if should_send:
                    batch = self._get_batch()
                    if batch:
                        self._send_batch(batch)
                        last_batch_time = datetime.now()

                # Small sleep to prevent busy loop
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Transmit loop error: {e}")
                time.sleep(1)

    def start(self):
        """Start the adaptive transmitter"""
        if self._running:
            return

        self._running = True
        self._transmit_thread = threading.Thread(target=self._transmit_loop, daemon=True)
        self._transmit_thread.start()

        # Start network monitoring if not already running
        self.network_monitor.start_monitoring()

        logger.info("Adaptive transmitter started")

    def stop(self):
        """Stop the adaptive transmitter"""
        self._running = False

        if self._transmit_thread:
            self._transmit_thread.join(timeout=10)
            self._transmit_thread = None

        logger.info("Adaptive transmitter stopped")

    def flush(self) -> int:
        """
        Force send all queued frames immediately.

        Returns:
            Number of frames sent
        """
        sent = 0
        while self.queue_size > 0:
            batch = self._get_batch()
            if not batch:
                break
            if self._send_batch(batch):
                sent += len(batch)
            else:
                break
        return sent

    def get_statistics(self) -> Dict[str, Any]:
        """Get transmission statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['priority_queue_size'] = len(self._priority_queue)
            stats['normal_queue_size'] = len(self._normal_queue)
            stats['current_mode'] = self._current_mode.value
            stats['compression_level'] = self._current_policy.compression_level.value

        return stats

    def clear_queues(self):
        """Clear all queued frames"""
        with self._lock:
            dropped = len(self._priority_queue) + len(self._normal_queue)
            self._priority_queue.clear()
            self._normal_queue.clear()
            self._stats['frames_dropped'] += dropped

        logger.info(f"Cleared {dropped} queued frames")


class TransmissionManager:
    """
    High-level manager that coordinates edge processing and adaptive transmission.
    Provides a simple interface for the main application.
    """

    def __init__(
        self,
        camera_id: str,
        location_id: str,
        server_url: str,
        api_key: Optional[str] = None,
        network_config: Optional[NetworkConfig] = None
    ):
        """
        Initialize transmission manager.

        Args:
            camera_id: Unique camera identifier
            location_id: Location/intersection ID for RTA
            server_url: RTA server URL
            api_key: API key for authentication
            network_config: Network monitoring configuration
        """
        self.camera_id = camera_id
        self.location_id = location_id
        self.server_url = server_url
        self.api_key = api_key

        # Configure network monitoring
        if network_config:
            network_config.primary_server = server_url
        else:
            network_config = NetworkConfig(primary_server=server_url)

        # Initialize components
        self.network_monitor = NetworkMonitor(
            config=network_config,
            on_offline=self._on_offline,
            on_reconnect=self._on_reconnect
        )

        self.edge_processor = EdgeProcessor(
            camera_id=camera_id,
            location_id=location_id
        )

        self.transmitter = AdaptiveTransmitter(
            network_monitor=self.network_monitor,
            send_func=self._send_to_server,
            on_mode_change=self._on_mode_change
        )

        # Callbacks
        self.on_offline: Optional[Callable] = None
        self.on_reconnect: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None

    def _send_to_server(self, frames: List[Dict[str, Any]]) -> bool:
        """Send frames to RTA server"""
        # This will be implemented in rta_client.py
        # Placeholder for now
        try:
            import urllib.request
            import json

            url = f"{self.server_url}/api/v1/frames"
            data = json.dumps({
                'camera_id': self.camera_id,
                'location_id': self.location_id,
                'frames': frames,
                'timestamp': datetime.now().isoformat()
            }).encode('utf-8')

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }

            request = urllib.request.Request(url, data=data, headers=headers, method='POST')

            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"Failed to send to server: {e}")
            return False

    def _on_offline(self):
        """Handle offline event"""
        logger.warning("Network went offline - switching to local storage mode")
        if self.on_offline:
            self.on_offline()

    def _on_reconnect(self):
        """Handle reconnect event"""
        logger.info("Network reconnected - resuming transmission")
        if self.on_reconnect:
            self.on_reconnect()

    def _on_mode_change(self, old_mode: TransmissionMode, new_mode: TransmissionMode):
        """Handle mode change"""
        if self.on_mode_change:
            self.on_mode_change(old_mode, new_mode)

    def start(self):
        """Start the transmission manager"""
        self.transmitter.start()
        logger.info(f"Transmission manager started for camera {self.camera_id}")

    def stop(self):
        """Stop the transmission manager"""
        self.transmitter.stop()
        self.network_monitor.stop_monitoring()
        logger.info(f"Transmission manager stopped for camera {self.camera_id}")

    def process_and_queue(
        self,
        frame,
        anomaly_detected: bool = False,
        anomaly_class: Optional[str] = None,
        anomaly_confidence: Optional[float] = None
    ) -> bool:
        """
        Process a frame and queue for transmission.

        Args:
            frame: Input frame (BGR numpy array)
            anomaly_detected: Whether anomaly was detected
            anomaly_class: Detected anomaly class
            anomaly_confidence: Detection confidence

        Returns:
            True if frame was queued, False if dropped
        """
        # Get current compression level
        compression_level = self.transmitter.get_compression_level()

        # Process frame
        processed = self.edge_processor.process_frame(
            frame,
            compression_level=compression_level,
            anomaly_detected=anomaly_detected,
            anomaly_class=anomaly_class,
            anomaly_confidence=anomaly_confidence
        )

        if processed is None:
            return False

        # Queue for transmission
        is_priority = anomaly_detected and anomaly_class in ['accident', 'fire']
        return self.transmitter.queue_frame(processed, is_priority=is_priority)

    def get_status(self) -> Dict[str, Any]:
        """Get complete status of the transmission system"""
        return {
            'camera_id': self.camera_id,
            'location_id': self.location_id,
            'network': self.network_monitor.get_statistics(),
            'transmission': self.transmitter.get_statistics(),
            'processing': self.edge_processor.get_statistics()
        }
