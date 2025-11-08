"""
Real-Time Stream Detection Page
"""

import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
from utils import (
    load_trained_model,
    VideoAnomalyDetector,
    AppConfig
)

config = AppConfig()

def show():
    """Display real-time stream detection page"""
    
    st.header("📡 Real-Time Stream Detection")
    st.markdown("Connect to a live video stream or webcam for continuous monitoring")
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.error("❌ Model not loaded. Please check the setup instructions.")
        return
    
    # Stream source selection
    st.markdown("### 🎥 Video Source")
    
    source_type = st.radio(
        "Select Source",
        ["Webcam", "RTSP Stream", "Video File (Simulated Stream)"],
        horizontal=True
    )
    
    video_source = None
    
    if source_type == "Webcam":
        camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0)
        video_source = camera_id
        st.info("📹 Using local webcam")
    
    elif source_type == "RTSP Stream":
        rtsp_url = st.text_input(
            "RTSP URL",
            placeholder="rtsp://username:password@ip_address:port/stream",
            help="Example: rtsp://admin:password@192.168.1.100:554/stream1"
        )
        if rtsp_url:
            video_source = rtsp_url
            st.info(f"📡 Connecting to: {rtsp_url}")
    
    else:  # Video File
        uploaded_video = st.file_uploader(
            "Upload Video File",
            type=config.SUPPORTED_VIDEO_FORMATS
        )
        if uploaded_video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_source = tmp_file.name
            st.info("🎬 Using uploaded video file")
    
    if not video_source:
        st.warning("⚠️ Please configure a video source to begin")
        return
    
    # Detection settings
    st.markdown("### ⚙️ Detection Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        frame_skip = st.slider("Frame Skip", 1, 10, config.SKIP_FRAMES)
    
    with col2:
        confidence_threshold = st.slider(
            "Alert Threshold (%)",
            50, 95,
            int(config.CONFIDENCE_THRESHOLD * 100)
        ) / 100
    
    with col3:
        alert_cooldown = st.slider("Alert Cooldown (s)", 1, 30, config.ALERT_COOLDOWN)
    
    # Alert preferences
    show_alerts = st.checkbox("🔔 Show Alerts", value=True)
    show_stats = st.checkbox("📊 Show Statistics", value=True)
    
    # Start/Stop controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        start_button = st.button("▶️ Start Detection", type="primary", use_container_width=True)
    
    with col2:
        stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
    
    # Session state for stream control
    if 'stream_active' not in st.session_state:
        st.session_state['stream_active'] = False
    
    if start_button:
        st.session_state['stream_active'] = True
    
    if stop_button:
        st.session_state['stream_active'] = False
    
    # Stream display
    if st.session_state['stream_active']:
        run_stream_detection(
            video_source,
            model,
            frame_skip,
            confidence_threshold,
            alert_cooldown,
            show_alerts,
            show_stats
        )
    else:
        st.info("👆 Click 'Start Detection' to begin monitoring")
        
        # Show demo info
        with st.expander("ℹ️ Stream Detection Features"):
            st.markdown("""
            ### Features
            - **Real-time Processing**: Analyze video frames as they arrive
            - **Temporal Smoothing**: Reduce false positives using frame history
            - **Persistent Alerts**: Only alert on sustained anomalies
            - **Live Statistics**: Monitor detection performance
            - **Alert Management**: Configurable cooldown periods
            
            ### Performance Tips
            - Increase frame skip for better performance
            - Adjust threshold based on environment
            - Use hardware acceleration if available
            - Monitor system resources
            """)

def run_stream_detection(video_source, model, frame_skip, confidence_threshold, 
                        alert_cooldown, show_alerts, show_stats):
    """Run real-time stream detection"""
    
    st.markdown("### 🔴 Live Detection Active")
    
    # Update config
    original_skip = config.SKIP_FRAMES
    original_threshold = config.CONFIDENCE_THRESHOLD
    original_cooldown = config.ALERT_COOLDOWN
    
    config.SKIP_FRAMES = frame_skip
    config.CONFIDENCE_THRESHOLD = confidence_threshold
    config.ALERT_COOLDOWN = alert_cooldown
    
    # Create placeholders
    video_placeholder = st.empty()
    
    if show_alerts:
        alert_placeholder = st.empty()
    
    if show_stats:
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        fps_metric = stats_col1.empty()
        frames_metric = stats_col2.empty()
        alerts_metric = stats_col3.empty()
        current_class_metric = stats_col4.empty()
    
    # Initialize detector
    detector = VideoAnomalyDetector(model)
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error(f"❌ Cannot open video source: {video_source}")
        config.SKIP_FRAMES = original_skip
        config.CONFIDENCE_THRESHOLD = original_threshold
        config.ALERT_COOLDOWN = original_cooldown
        return
    
    # Stream processing
    frame_count = 0
    alert_count = 0
    fps_history = deque(maxlen=30)
    last_time = time.time()
    
    try:
        while st.session_state.get('stream_active', False):
            ret, frame = cap.read()
            
            if not ret:
                # Loop video if simulated stream
                if isinstance(video_source, str) and not video_source.startswith('rtsp'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Process frame
            if frame_count % frame_skip == 0:
                # Predict
                prediction = detector.process_frame(frame)
                event = detector.detect_anomaly_event()
                
                # Annotate frame
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                annotated_frame = detector.annotate_frame(
                    frame,
                    prediction,
                    event,
                    frame_count,
                    fps
                )
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Show alert
                if show_alerts and event and event['persistent']:
                    alert_count += 1
                    emoji = event['emoji']
                    class_name = event['class'].replace('_', ' ').title()
                    confidence = event['confidence'] * 100
                    
                    if event['severity'] == 'critical':
                        alert_placeholder.error(
                            f"{emoji} **CRITICAL ALERT: {class_name}** "
                            f"(Confidence: {confidence:.1f}%)"
                        )
                    elif event['severity'] == 'warning':
                        alert_placeholder.warning(
                            f"{emoji} **WARNING: {class_name}** "
                            f"(Confidence: {confidence:.1f}%)"
                        )
                
                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - last_time
                if frame_time > 0:
                    current_fps = 1.0 / frame_time
                    fps_history.append(current_fps)
                last_time = current_time
                
                # Update statistics
                if show_stats:
                    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                    fps_metric.metric("FPS", f"{avg_fps:.1f}")
                    frames_metric.metric("Frames", f"{frame_count:,}")
                    alerts_metric.metric("Alerts", alert_count)
                    
                    emoji = prediction['emoji']
                    class_name = prediction['class'].replace('_', ' ').title()
                    current_class_metric.metric(
                        "Current",
                        f"{emoji} {class_name}",
                        delta=f"{prediction['confidence']:.1f}%"
                    )
            
            frame_count += 1
            
            # Small delay to prevent overwhelming the browser
            time.sleep(0.01)
    
    except Exception as e:
        st.error(f"❌ Error during stream processing: {str(e)}")
    
    finally:
        # Cleanup
        cap.release()
        st.session_state['stream_active'] = False
        
        # Restore config
        config.SKIP_FRAMES = original_skip
        config.CONFIDENCE_THRESHOLD = original_threshold
        config.ALERT_COOLDOWN = original_cooldown
        
        st.info("⏹️ Stream detection stopped")