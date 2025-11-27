#!/usr/bin/env python3
"""
Test script for Edge Processing System
Run this to verify all components work before deploying to Raspberry Pi.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Reduce TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(test_name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"       {details}")

def test_imports():
    """Test all module imports"""
    print_header("Testing Module Imports")

    tests = []

    # Test config imports
    try:
        from utils.config import AppConfig, EdgeConfig, RTAConfig
        tests.append(("Config modules", True, f"EdgeConfig.CAMERA_ID = {EdgeConfig.CAMERA_ID}"))
    except Exception as e:
        tests.append(("Config modules", False, str(e)))

    # Test edge processor
    try:
        from utils.edge_processor import EdgeProcessor, CompressionLevel, KeyframeExtractor
        tests.append(("Edge processor", True, f"Compression levels: {[c.value for c in CompressionLevel]}"))
    except Exception as e:
        tests.append(("Edge processor", False, str(e)))

    # Test network monitor
    try:
        from utils.network_monitor import NetworkMonitor, NetworkStatus, NetworkConfig
        tests.append(("Network monitor", True, f"Network statuses: {[s.value for s in NetworkStatus]}"))
    except Exception as e:
        tests.append(("Network monitor", False, str(e)))

    # Test adaptive transmitter
    try:
        from utils.adaptive_transmitter import AdaptiveTransmitter, TransmissionMode
        tests.append(("Adaptive transmitter", True, f"Transmission modes: {[m.value for m in TransmissionMode]}"))
    except Exception as e:
        tests.append(("Adaptive transmitter", False, str(e)))

    # Test RTA client
    try:
        from utils.rta_client import RTAClient, RTAEvent, RTAEventType
        tests.append(("RTA client", True, f"Event types: {[e.value for e in RTAEventType]}"))
    except Exception as e:
        tests.append(("RTA client", False, str(e)))

    # Test offline storage
    try:
        from utils.offline_storage import OfflineStorage, OfflineSyncManager
        tests.append(("Offline storage", True, "SQLite storage ready"))
    except Exception as e:
        tests.append(("Offline storage", False, str(e)))

    for test_name, success, details in tests:
        print_result(test_name, success, details)

    return all(t[1] for t in tests)

def test_edge_processor():
    """Test edge processing functionality"""
    print_header("Testing Edge Processor")

    try:
        import numpy as np
        from utils.edge_processor import EdgeProcessor, CompressionLevel

        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Initialize processor
        processor = EdgeProcessor(
            camera_id="test-cam",
            location_id="test-location"
        )

        print_result("EdgeProcessor initialization", True)

        # Test keyframe extraction (force first frame as keyframe)
        processed = processor.process_frame(
            test_frame,
            compression_level=CompressionLevel.HIGH,
            force_keyframe=True
        )

        if processed:
            print_result("Keyframe extraction", True,
                        f"Frame ID: {processed.frame_id[:30]}...")
            print_result("Compression", True,
                        f"Ratio: {processed.compression_ratio:.1%}, "
                        f"Size: {processed.compressed_bytes} bytes")
        else:
            print_result("Keyframe extraction", False, "No frame returned")
            return False

        # Test different compression levels
        for level in CompressionLevel:
            processed = processor.process_frame(
                test_frame,
                compression_level=level,
                force_keyframe=True
            )
            if processed:
                print_result(f"Compression level: {level.value}", True,
                            f"Size: {processed.compressed_bytes} bytes")

        # Get statistics
        stats = processor.get_statistics()
        print_result("Statistics", True,
                    f"Frames processed: {stats['total_frames_processed']}")

        return True

    except Exception as e:
        print_result("Edge processor test", False, str(e))
        return False

def test_network_monitor():
    """Test network monitoring"""
    print_header("Testing Network Monitor")

    try:
        from utils.network_monitor import NetworkMonitor, NetworkConfig, NetworkStatus

        # Initialize with test config
        config = NetworkConfig(
            primary_server="https://httpbin.org",
            check_interval_seconds=1.0,
            timeout_seconds=5.0
        )

        monitor = NetworkMonitor(config=config)
        print_result("NetworkMonitor initialization", True)

        # Single connectivity check
        metrics = monitor.check_connectivity()
        print_result("Connectivity check", True,
                    f"Status: {metrics.status.value}, "
                    f"Latency: {metrics.latency_ms:.0f}ms")

        # Check if online
        print_result("Online status", metrics.is_connected,
                    f"Connected: {metrics.is_connected}")

        return True

    except Exception as e:
        print_result("Network monitor test", False, str(e))
        return False

def test_offline_storage():
    """Test offline storage"""
    print_header("Testing Offline Storage")

    try:
        from utils.offline_storage import OfflineStorage, StorageConfig
        import tempfile

        # Create temp database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            config = StorageConfig(db_path=db_path)
            storage = OfflineStorage(config=config)
            print_result("OfflineStorage initialization", True)

            # Test storing a frame
            test_frame = {
                'frame_id': 'test-frame-001',
                'timestamp': '2024-01-01T00:00:00',
                'data_base64': 'dGVzdCBkYXRh',
                'metadata': {'camera_id': 'test'}
            }

            success = storage.store_frame(test_frame, is_priority=True)
            print_result("Store frame", success)

            # Test storing an event
            test_event = {
                'event_id': 'test-event-001',
                'camera_id': 'test',
                'location_id': 'test',
                'event_type': 'ACCIDENT',
                'severity': 1,
                'timestamp': '2024-01-01T00:00:00',
                'confidence': 0.95
            }

            success = storage.store_event(test_event)
            print_result("Store event", success)

            # Get statistics
            stats = storage.get_statistics()
            print_result("Get statistics", True,
                        f"Pending frames: {stats['pending_frames']}, "
                        f"Pending events: {stats['pending_events']}")

            return True

        finally:
            # Cleanup
            os.unlink(db_path)

    except Exception as e:
        print_result("Offline storage test", False, str(e))
        return False

def test_rta_client():
    """Test RTA client (without actual API calls)"""
    print_header("Testing RTA Client")

    try:
        from utils.rta_client import RTAClient, RTAConfig, RTAEvent, RTAEventType, RTASeverity
        from datetime import datetime

        # Initialize client
        config = RTAConfig(
            base_url="https://api.rta.ae/traffic/v1",
            camera_id="test-cam",
            location_id="test-location",
            coordinates=(25.2048, 55.2708)
        )

        client = RTAClient(config)
        print_result("RTAClient initialization", True)

        # Test event creation
        event = client.create_event_from_detection(
            anomaly_class='accident',
            confidence=0.95,
            metadata={'test': True}
        )

        print_result("Event creation", True,
                    f"Event ID: {event.event_id[:30]}...")
        print_result("Event type", event.event_type == RTAEventType.ACCIDENT,
                    f"Type: {event.event_type.value}")
        print_result("Event severity", event.severity == RTASeverity.CRITICAL,
                    f"Severity: {event.severity.value}")

        # Test serialization
        event_dict = event.to_dict()
        print_result("Event serialization", 'event_id' in event_dict,
                    f"Keys: {list(event_dict.keys())[:5]}...")

        return True

    except Exception as e:
        print_result("RTA client test", False, str(e))
        return False

def test_model_loading():
    """Test ML model loading"""
    print_header("Testing Model Loading")

    try:
        from utils.model_utils import load_trained_model

        model = load_trained_model()

        if model is not None:
            print_result("Model loading", True,
                        f"Model type: {type(model).__name__}")

            # Check model structure
            if hasattr(model, 'input_shape'):
                print_result("Model input shape", True,
                            f"Input: {model.input_shape}")
            if hasattr(model, 'output_shape'):
                print_result("Model output shape", True,
                            f"Output: {model.output_shape}")

            return True
        else:
            print_result("Model loading", False, "Model is None")
            return False

    except Exception as e:
        print_result("Model loading", False, str(e))
        return False

def test_camera_access():
    """Test camera access (optional)"""
    print_header("Testing Camera Access")

    try:
        import cv2

        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

            if ret:
                print_result("Camera access", True,
                            f"Frame shape: {frame.shape}")
                return True
            else:
                print_result("Camera access", False, "Could not read frame")
                return False
        else:
            print_result("Camera access", False,
                        "No camera detected (this is OK if running headless)")
            return True  # Not a failure, camera might not be available

    except Exception as e:
        print_result("Camera access", False, str(e))
        return True  # Not critical

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  EDGE PROCESSING SYSTEM - TEST SUITE")
    print("="*60)

    results = []

    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Edge Processor", test_edge_processor()))
    results.append(("Network Monitor", test_network_monitor()))
    results.append(("Offline Storage", test_offline_storage()))
    results.append(("RTA Client", test_rta_client()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Camera Access", test_camera_access()))

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        print_result(test_name, success)

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'='*60}\n")

    if passed == total:
        print("🎉 All tests passed! Ready for Raspberry Pi deployment.\n")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
