"""
Analytics Dashboard Page
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from utils import (
    load_trained_model,
    get_model_summary,
    AppConfig
)

config = AppConfig()

def show():
    """Display analytics dashboard page"""
    
    st.header("📊 Analytics Dashboard")
    st.markdown("System performance metrics, model information, and usage statistics")
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🤖 Model Info",
        "📉 Performance",
        "⚙️ System"
    ])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_model_info()
    
    with tab3:
        show_performance_metrics()
    
    with tab4:
        show_system_info()

def show_overview():
    """Show overview statistics"""
    
    st.subheader("System Overview")
    
    # Check if we have session data
    has_image_results = 'last_result' in st.session_state
    has_batch_results = 'batch_results' in st.session_state
    has_video_results = 'video_results' in st.session_state
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_images = 0
        if has_image_results:
            total_images += 1
        if has_batch_results:
            total_images += len(st.session_state['batch_results'])
        st.metric("Images Processed", f"{total_images:,}")
    
    with col2:
        total_videos = 1 if has_video_results else 0
        st.metric("Videos Processed", total_videos)
    
    with col3:
        model = load_trained_model()
        status = "✅ Loaded" if model else "❌ Not Loaded"
        st.metric("Model Status", status)
    
    with col4:
        st.metric("Model Version", "1.0")
    
    # Recent activity
    st.markdown("### 🕐 Recent Activity")
    
    activities = []
    
    if has_image_results:
        result = st.session_state['last_result']
        activities.append({
            'type': 'Image Detection',
            'result': result['class'].replace('_', ' ').title(),
            'confidence': f"{result['confidence']:.1f}%",
            'time': 'Recent'
        })
    
    if has_batch_results:
        activities.append({
            'type': 'Batch Analysis',
            'result': f"{len(st.session_state['batch_results'])} images",
            'confidence': 'N/A',
            'time': 'Recent'
        })
    
    if has_video_results:
        results = st.session_state['video_results']
        activities.append({
            'type': 'Video Analysis',
            'result': f"{results['total_frames']} frames",
            'confidence': 'N/A',
            'time': 'Recent'
        })
    
    if activities:
        df = pd.DataFrame(activities)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent activity. Start using the system to see statistics here.")
    
    # Class distribution across all analyses
    if has_batch_results or has_video_results or has_image_results:
        st.markdown("### 📊 Overall Class Distribution")
        
        class_counts = {}
        
        # Aggregate from all sources
        if has_image_results:
            result = st.session_state['last_result']
            class_counts[result['class']] = class_counts.get(result['class'], 0) + 1
        
        if has_batch_results:
            for result in st.session_state['batch_results']:
                class_name = result['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if has_video_results:
            for class_name, count in st.session_state['video_results']['class_distribution'].items():
                class_counts[class_name] = class_counts.get(class_name, 0) + count
        
        # Create pie chart
        from utils.visualization import create_pie_chart
        fig = create_pie_chart(class_counts, title="Aggregated Detection Results")
        st.plotly_chart(fig, use_container_width=True)

def show_model_info():
    """Show model information"""
    
    st.subheader("🤖 Model Information")
    
    model = load_trained_model()
    
    if model is None:
        st.error("❌ Model not loaded")
        return
    
    # Model details
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Configuration")
        st.write(f"**Model Name:** {config.MODEL_NAME}")
        st.write(f"**Input Size:** {config.IMAGE_SIZE[0]} x {config.IMAGE_SIZE[1]}")
        st.write(f"**Classes:** {len(config.CLASS_NAMES)}")
        st.write(f"**Architecture:** CNN with Batch Normalization")
    
    with col2:
        st.markdown("#### Training Parameters")
        st.write(f"**Batch Size:** {config.BATCH_SIZE}")
        st.write(f"**Epochs:** {config.EPOCHS}")
        st.write(f"**Learning Rate:** {config.LEARNING_RATE}")
        st.write(f"**Validation Split:** {config.VALIDATION_SPLIT}")
    
    # Class information
    st.markdown("#### Detected Classes")
    
    cols = st.columns(4)
    for idx, class_name in enumerate(config.CLASS_NAMES):
        with cols[idx]:
            emoji = config.get_emoji(class_name)
            severity = config.get_severity(class_name)
            
            st.markdown(f"### {emoji}")
            st.markdown(f"**{class_name.replace('_', ' ').title()}**")
            
            if severity == 'critical':
                st.error(f"Severity: {severity.upper()}")
            elif severity == 'warning':
                st.warning(f"Severity: {severity.upper()}")
            else:
                st.success(f"Severity: {severity.upper()}")
    
    # Model architecture
    st.markdown("#### Model Architecture")
    
    with st.expander("🏗️ View Detailed Architecture"):
        summary = get_model_summary()
        st.text(summary)
    
    # Model file info
    st.markdown("#### Model File")
    
    if config.MODEL_PATH.exists():
        file_size = config.MODEL_PATH.stat().st_size / (1024 * 1024)
        st.success(f"✅ Model file found: {config.MODEL_PATH}")
        st.info(f"📦 File size: {file_size:.2f} MB")
    else:
        st.error(f"❌ Model file not found at: {config.MODEL_PATH}")

def show_performance_metrics():
    """Show performance metrics"""
    
    st.subheader("📉 Performance Metrics")
    
    # Check if we have performance data
    has_video_results = 'video_results' in st.session_state
    has_batch_results = 'batch_results' in st.session_state
    
    if not (has_video_results or has_batch_results):
        st.info("No performance data available yet. Process some images or videos to see metrics.")
        return
    
    # Processing statistics
    if has_batch_results:
        st.markdown("### 📁 Batch Processing Performance")
        
        results = st.session_state['batch_results']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Images", len(results))
        
        with col2:
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            st.metric("Avg Confidence", f"{avg_conf:.2f}%")
        
        with col3:
            # Count high confidence predictions
            high_conf = sum(1 for r in results if r['confidence'] >= 80)
            st.metric("High Confidence", f"{high_conf}/{len(results)}")
        
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        
        fig = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7
        )])
        
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence (%)",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if has_video_results:
        st.markdown("### 🎥 Video Processing Performance")
        
        results = st.session_state['video_results']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Frames", f"{results['total_frames']:,}")
        
        with col2:
            st.metric("Processed", f"{results['processed_frames']:,}")
        
        with col3:
            processing_rate = (results['processed_frames'] / results['total_frames']) * 100
            st.metric("Processing Rate", f"{processing_rate:.1f}%")
        
        with col4:
            st.metric("Anomaly Events", len(results['anomaly_events']))
        
        # Event confidence distribution
        if results['anomaly_events']:
            event_confidences = [e['confidence'] * 100 for e in results['anomaly_events']]
            
            fig = go.Figure(data=[go.Box(
                y=event_confidences,
                name='Event Confidence',
                marker_color='coral'
            )])
            
            fig.update_layout(
                title="Anomaly Event Confidence Distribution",
                yaxis_title="Confidence (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_system_info():
    """Show system information"""
    
    st.subheader("⚙️ System Configuration")
    
    # Application settings
    st.markdown("### 🔧 Application Settings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Processing Settings")
        st.write(f"**Frame Skip:** {config.SKIP_FRAMES}")
        st.write(f"**Confidence Threshold:** {config.CONFIDENCE_THRESHOLD * 100}%")
        st.write(f"**Temporal Window:** {config.TEMPORAL_WINDOW} frames")
        st.write(f"**Smoothing Alpha:** {config.SMOOTHING_ALPHA}")
    
    with col2:
        st.markdown("#### Resource Limits")
        st.write(f"**Max Batch Size:** {config.MAX_BATCH_SIZE} images")
        st.write(f"**Max Video Size:** {config.MAX_VIDEO_SIZE_MB} MB")
        st.write(f"**Buffer Size:** {config.VIDEO_BUFFER_SIZE} frames")
        st.write(f"**Alert Cooldown:** {config.ALERT_COOLDOWN}s")
    
    # File format support
    st.markdown("### 📄 Supported File Formats")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Image Formats")
        st.write(", ".join(config.SUPPORTED_IMAGE_FORMATS))
    
    with col2:
        st.markdown("#### Video Formats")
        st.write(", ".join(config.SUPPORTED_VIDEO_FORMATS))
    
    # Directory information
    st.markdown("### 📂 Directories")
    
    dirs = {
        "Models": config.MODEL_DIR,
        "Temporary Files": config.TEMP_DIR,
        "Output Files": config.OUTPUT_DIR
    }
    
    for name, path in dirs.items():
        exists = "✅" if path.exists() else "❌"
        st.write(f"**{name}:** {exists} {path}")
    
    # System health check
    st.markdown("### 🏥 System Health")
    
    health_checks = []
    
    # Check model
    model = load_trained_model()
    health_checks.append({
        'Component': 'Model',
        'Status': '✅ OK' if model else '❌ Failed',
        'Details': 'Model loaded successfully' if model else 'Model not found'
    })
    
    # Check directories
    all_dirs_exist = all(path.exists() for path in [config.MODEL_DIR, config.TEMP_DIR, config.OUTPUT_DIR])
    health_checks.append({
        'Component': 'Directories',
        'Status': '✅ OK' if all_dirs_exist else '⚠️ Warning',
        'Details': 'All directories exist' if all_dirs_exist else 'Some directories missing'
    })
    
    # Check model file
    model_exists = config.MODEL_PATH.exists()
    health_checks.append({
        'Component': 'Model File',
        'Status': '✅ OK' if model_exists else '❌ Failed',
        'Details': f'Found at {config.MODEL_PATH}' if model_exists else 'File not found'
    })
    
    df = pd.DataFrame(health_checks)
    st.dataframe(df, use_container_width=True)
    
    # Environment info
    with st.expander("🌐 Environment Information"):
        import sys
        import platform
        
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**Processor:** {platform.processor()}")
        
        # Package versions
        try:
            import tensorflow as tf
            st.write(f"**TensorFlow:** {tf.__version__}")
        except:
            st.write("**TensorFlow:** Not installed")
        
        try:
            import cv2
            st.write(f"**OpenCV:** {cv2.__version__}")
        except:
            st.write("**OpenCV:** Not installed")
        
        try:
            import streamlit
            st.write(f"**Streamlit:** {streamlit.__version__}")
        except:
            st.write("**Streamlit:** Not installed")