"""
Video Detection Page
"""

import streamlit as st
import tempfile
from pathlib import Path
import cv2
from utils import (
    load_trained_model,
    VideoAnomalyDetector,
    create_timeline_chart,
    create_confidence_trend,
    create_pie_chart,
    AppConfig
)

config = AppConfig()

def show():
    """Display video detection page"""
    
    st.header("🎥 Video Anomaly Detection")
    st.markdown("Upload a video file to detect traffic anomalies with temporal analysis")
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.error("❌ Model not loaded. Please check the setup instructions.")
        return
    
    # File uploader
    uploaded_video = st.file_uploader(
        "Upload Traffic Video",
        type=config.SUPPORTED_VIDEO_FORMATS,
        help=f"Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS)}. Max size: {config.MAX_VIDEO_SIZE_MB}MB"
    )
    
    if uploaded_video:
        # Check file size
        file_size_mb = uploaded_video.size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        with col2:
            st.metric("Format", uploaded_video.name.split('.')[-1].upper())
        with col3:
            st.info("Ready to process")
        
        if file_size_mb > config.MAX_VIDEO_SIZE_MB:
            st.error(f"❌ File too large. Maximum size: {config.MAX_VIDEO_SIZE_MB}MB")
            return
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Display original video
        st.markdown("### 📹 Original Video")
        st.video(uploaded_video)
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            skip_frames = st.slider(
                "Frame Skip",
                min_value=1,
                max_value=10,
                value=config.SKIP_FRAMES,
                help="Process every Nth frame (higher = faster but less accurate)"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold (%)",
                min_value=50,
                max_value=95,
                value=int(config.CONFIDENCE_THRESHOLD * 100),
                help="Minimum confidence to trigger alert"
            ) / 100
        
        save_output = st.checkbox(
            "💾 Save Annotated Video",
            value=True,
            help="Save video with detection overlays"
        )
        
        # Process button
        process_button = st.button("🚀 Process Video", type="primary", use_container_width=True)
        
        if process_button:
            # Create output path
            output_path = None
            if save_output:
                output_path = config.OUTPUT_DIR / f"annotated_{uploaded_video.name}"
            
            # Progress tracking
            st.markdown("### 🔄 Processing Video...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {current}/{total} ({progress*100:.1f}%)")
            
            # Create detector
            detector = VideoAnomalyDetector(model)
            
            # Update config temporarily
            original_skip = config.SKIP_FRAMES
            original_threshold = config.CONFIDENCE_THRESHOLD
            config.SKIP_FRAMES = skip_frames
            config.CONFIDENCE_THRESHOLD = confidence_threshold
            
            try:
                # Process video
                with st.spinner("Processing video..."):
                    results = detector.process_video_file(
                        video_path,
                        output_path=str(output_path) if output_path else None,
                        progress_callback=progress_callback
                    )
                
                # Restore config
                config.SKIP_FRAMES = original_skip
                config.CONFIDENCE_THRESHOLD = original_threshold
                
                st.success("✅ Video processing complete!")
                
                # Display results
                display_video_results(results, output_path)
                
                # Store in session state
                st.session_state['video_results'] = results
                st.session_state['video_output'] = output_path
            
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                # Restore config
                config.SKIP_FRAMES = original_skip
                config.CONFIDENCE_THRESHOLD = original_threshold
        
        # Display previous results if available
        elif 'video_results' in st.session_state:
            st.info("Showing previous results. Click 'Process Video' to reprocess.")
            display_video_results(
                st.session_state['video_results'],
                st.session_state.get('video_output')
            )
    
    else:
        # Instructions
        st.info("""
        ### 📝 Instructions
        1. Upload a traffic video file
        2. Adjust processing options if needed
        3. Click 'Process Video' to analyze
        4. View results and download annotated video
        
        ### ✨ Features
        - **Frame-by-frame detection** with temporal smoothing
        - **Persistent anomaly detection** across multiple frames
        - **Timeline visualization** of detected events
        - **Confidence trends** analysis
        - **Annotated video output** with real-time overlays
        """)

def display_video_results(results, output_path):
    """Display video processing results"""
    
    # Summary statistics
    st.markdown("### 📊 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Frames", f"{results['total_frames']:,}")
    with col2:
        st.metric("Processed", f"{results['processed_frames']:,}")
    with col3:
        st.metric("Duration", f"{results['duration']:.1f}s")
    with col4:
        st.metric("Anomaly Events", len(results['anomaly_events']))
    
    # Annotated video
    if output_path and Path(output_path).exists():
        st.markdown("### 🎬 Annotated Video")
        with open(output_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
        
        # Download button
        st.download_button(
            label="📥 Download Annotated Video",
            data=video_bytes,
            file_name=Path(output_path).name,
            mime="video/mp4",
            use_container_width=True
        )
    
    # Anomaly events timeline
    if results['anomaly_events']:
        st.markdown("### ⏱️ Detected Anomaly Events")
        
        # Timeline chart
        timeline_fig = create_timeline_chart(
            results['anomaly_events'],
            results['duration']
        )
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Events table
        with st.expander("📋 Event Details"):
            for idx, event in enumerate(results['anomaly_events'], 1):
                emoji = config.get_emoji(event['class'])
                st.write(
                    f"**{idx}. {emoji} {event['class'].replace('_', ' ').title()}** - "
                    f"Frame {event['frame']} ({event['timestamp']:.2f}s) - "
                    f"Confidence: {event['confidence']*100:.1f}%"
                )
    else:
        st.info("✅ No critical anomaly events detected in this video.")
    
    # Class distribution
    if results['class_distribution']:
        st.markdown("### 📈 Traffic Conditions Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            pie_fig = create_pie_chart(
                results['class_distribution'],
                title="Frame Classification"
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            # Statistics
            st.markdown("#### Frame Statistics")
            total = sum(results['class_distribution'].values())
            for class_name, count in sorted(
                results['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                emoji = config.get_emoji(class_name)
                percentage = (count / total) * 100
                st.write(
                    f"{emoji} **{class_name.replace('_', ' ').title()}:** "
                    f"{count:,} frames ({percentage:.1f}%)"
                )
    
    # Confidence trends
    if results['frame_predictions']:
        st.markdown("### 📉 Confidence Trends")
        confidence_fig = create_confidence_trend(results['frame_predictions'])
        st.plotly_chart(confidence_fig, use_container_width=True)
    
    # Export report
    st.markdown("### 📥 Export Analysis Report")
    
    report_text = generate_report_text(results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Text Report",
            data=report_text,
            file_name="video_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # CSV export of events
        if results['anomaly_events']:
            import pandas as pd
            df = pd.DataFrame(results['anomaly_events'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Events CSV",
                data=csv,
                file_name="anomaly_events.csv",
                mime="text/csv",
                use_container_width=True
            )

def generate_report_text(results):
    """Generate text report from results"""
    
    report = f"""
Traffic Anomaly Detection - Video Analysis Report
{'='*60}

VIDEO INFORMATION:
- Total Frames: {results['total_frames']:,}
- Processed Frames: {results['processed_frames']:,}
- Frame Rate: {results['fps']} FPS
- Duration: {results['duration']:.2f} seconds

ANOMALY EVENTS DETECTED: {len(results['anomaly_events'])}
"""
    
    if results['anomaly_events']:
        report += "\nDETAILED EVENTS:\n"
        for idx, event in enumerate(results['anomaly_events'], 1):
            report += f"\n{idx}. {event['class'].upper().replace('_', ' ')}\n"
            report += f"   Frame: {event['frame']}\n"
            report += f"   Timestamp: {event['timestamp']:.2f}s\n"
            report += f"   Confidence: {event['confidence']*100:.1f}%\n"
    
    if results['class_distribution']:
        report += "\n\nFRAME CLASSIFICATION DISTRIBUTION:\n"
        total = sum(results['class_distribution'].values())
        for class_name, count in sorted(
            results['class_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / total) * 100
            report += f"- {class_name.replace('_', ' ').title()}: {count:,} frames ({percentage:.1f}%)\n"
    
    report += f"\n\nReport generated by Traffic Anomaly Detection System v1.0\n"
    
    return report