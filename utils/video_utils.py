"""
Video processing utilities with temporal analysis
"""

import cv2
import numpy as np
from collections import deque
import time
from pathlib import Path
from utils.config import AppConfig
from utils.model_utils import predict_single_image

config = AppConfig()

class VideoAnomalyDetector:
    """Video anomaly detector with temporal analysis"""
    
    def __init__(self, model):
        """
        Initialize video detector
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.frame_buffer = deque(maxlen=config.VIDEO_BUFFER_SIZE)
        self.prediction_history = deque(maxlen=config.VIDEO_BUFFER_SIZE)
        self.last_alert_time = {}
        
    def reset(self):
        """Reset detector state"""
        self.frame_buffer.clear()
        self.prediction_history.clear()
        self.last_alert_time.clear()
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Prediction dictionary
        """
        result = predict_single_image(self.model, frame)
        self.prediction_history.append(result['raw_prediction'])
        return result
    
    def temporal_smoothing(self, current_pred):
        """
        Apply exponential moving average for temporal smoothing
        
        Args:
            current_pred: Current frame prediction array
        
        Returns:
            Smoothed prediction array
        """
        if len(self.prediction_history) == 0:
            return current_pred
        
        alpha = config.SMOOTHING_ALPHA
        prev_pred = self.prediction_history[-1]
        smoothed = alpha * current_pred + (1 - alpha) * prev_pred
        
        return smoothed
    
    def detect_anomaly_event(self, window_size=None, threshold=None):
        """
        Detect persistent anomalies across multiple frames
        
        Args:
            window_size: Number of frames to analyze
            threshold: Confidence threshold
        
        Returns:
            Event dictionary or None
        """
        if window_size is None:
            window_size = config.TEMPORAL_WINDOW
        if threshold is None:
            threshold = config.CONFIDENCE_THRESHOLD
        
        if len(self.prediction_history) < window_size:
            return None
        
        # Analyze recent predictions
        recent_preds = list(self.prediction_history)[-window_size:]
        avg_predictions = np.mean(recent_preds, axis=0)
        
        max_class_idx = np.argmax(avg_predictions)
        confidence = avg_predictions[max_class_idx]
        
        # Check if anomaly is persistent and above threshold
        if confidence > threshold:
            class_name = config.CLASS_NAMES[max_class_idx]
            
            # Check alert cooldown
            current_time = time.time()
            if class_name in self.last_alert_time:
                time_since_last = current_time - self.last_alert_time[class_name]
                if time_since_last < config.ALERT_COOLDOWN:
                    return None
            
            self.last_alert_time[class_name] = current_time
            
            return {
                'class': class_name,
                'confidence': float(confidence),
                'persistent': True,
                'severity': config.get_severity(class_name),
                'emoji': config.get_emoji(class_name)
            }
        
        return None
    
    def annotate_frame(self, frame, prediction, event=None, frame_number=None, fps=None):
        """
        Annotate frame with detection results
        
        Args:
            frame: Input frame
            prediction: Prediction dictionary
            event: Persistent event dictionary (optional)
            frame_number: Current frame number (optional)
            fps: Video FPS (optional)
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Get prediction info
        top_class = prediction['class']
        confidence = prediction['confidence']
        color = config.get_color_bgr(top_class)
        
        # Draw alert box if persistent event
        if event and event['persistent']:
            alert_height = 120
            cv2.rectangle(annotated, (0, 0), (width, alert_height), color, -1)
            
            # Alert text
            alert_text = f"{event['emoji']} ALERT: {event['class'].upper().replace('_', ' ')}"
            cv2.putText(annotated, alert_text,
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            conf_text = f"Confidence: {event['confidence']*100:.1f}%"
            cv2.putText(annotated, conf_text,
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Status bar at bottom
        status_height = 80
        cv2.rectangle(annotated, (0, height-status_height), (width, height), (0, 0, 0), -1)
        
        # Main status text
        status_text = f"{config.get_emoji(top_class)} {top_class.replace('_', ' ').title()}: {confidence:.1f}%"
        cv2.putText(annotated, status_text,
                   (15, height-45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Frame info
        if frame_number is not None and fps is not None:
            timestamp = frame_number / fps
            time_text = f"Frame: {frame_number} | Time: {timestamp:.2f}s"
            cv2.putText(annotated, time_text,
                       (15, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Probability bars
        bar_width = width // 4
        bar_height_max = 50
        for idx, class_name in enumerate(config.CLASS_NAMES):
            prob = prediction['probabilities'][class_name] / 100
            bar_height = int(prob * bar_height_max)
            x_pos = idx * bar_width + 5
            
            # Draw bar
            bar_color = config.get_color_bgr(class_name)
            cv2.rectangle(annotated,
                         (x_pos, height-status_height-bar_height),
                         (x_pos + bar_width - 10, height-status_height),
                         bar_color, -1)
            
            # Draw label
            label = class_name.replace('_', ' ')[:8]
            cv2.putText(annotated, label,
                       (x_pos, height-status_height-bar_height-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
        
        return annotated
    
    def process_video_file(self, video_path, output_path=None, progress_callback=None):
        """
        Process entire video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            progress_callback: Callback function(current, total)
        
        Returns:
            Dictionary with processing results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing
        self.reset()
        frame_count = 0
        processed_frames = 0
        anomaly_events = []
        frame_predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (with frame skipping)
            if frame_count % config.SKIP_FRAMES == 0:
                prediction = self.process_frame(frame)
                smoothed_pred = self.temporal_smoothing(prediction['raw_prediction'])
                
                # Update prediction with smoothed values
                smoothed_class_idx = np.argmax(smoothed_pred)
                prediction['class'] = config.CLASS_NAMES[smoothed_class_idx]
                prediction['confidence'] = smoothed_pred[smoothed_class_idx] * 100
                
                # Detect persistent events
                event = self.detect_anomaly_event()
                
                # Annotate
                annotated_frame = self.annotate_frame(frame, prediction, event, frame_count, fps)
                
                # Record event
                if event and event['persistent']:
                    event_data = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        **event
                    }
                    anomaly_events.append(event_data)
                
                # Store frame prediction
                frame_predictions.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'class': prediction['class'],
                    'confidence': prediction['confidence']
                })
                
                processed_frames += 1
            else:
                annotated_frame = frame
            
            # Write frame
            if out:
                out.write(annotated_frame)
            
            # Progress callback
            if progress_callback:
                progress_callback(frame_count + 1, total_frames)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        # Generate summary statistics
        class_counts = {}
        for pred in frame_predictions:
            class_name = pred['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        results = {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'duration': duration,
            'anomaly_events': anomaly_events,
            'frame_predictions': frame_predictions,
            'class_distribution': class_counts,
            'output_path': output_path
        }
        
        return results
    
    def process_stream(self, source=0, max_frames=None):
        """
        Process live video stream (for real-time detection)
        
        Args:
            source: Video source (0 for webcam, RTSP URL for IP camera)
            max_frames: Maximum frames to process (None for infinite)
        
        Yields:
            Tuple (annotated_frame, prediction, event)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default for streams
        
        self.reset()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            if frame_count % config.SKIP_FRAMES == 0:
                prediction = self.process_frame(frame)
                event = self.detect_anomaly_event()
                annotated_frame = self.annotate_frame(frame, prediction, event, frame_count, fps)
                
                yield annotated_frame, prediction, event
            
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()