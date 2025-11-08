"""
Traffic Anomaly Detection System - Main Application
Production-ready Streamlit application for traffic anomaly detection
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config import AppConfig
from pages import image_detection, video_detection, batch_analysis, analytics_dashboard, realtime_stream

# Page configuration
st.set_page_config(
    page_title="Traffic Anomaly Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffa500;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-success {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("🚦 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Mode",
        ["📸 Image Detection", "🎥 Video Detection", "📁 Batch Analysis", 
         "📡 Real-Time Stream", "📊 Analytics Dashboard"],
        index=0
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This system uses deep learning to detect traffic anomalies including:\n\n"
        "🚨 Accidents\n\n"
        "🔥 Fire incidents\n\n"
        "🚗 Dense traffic\n\n"
        "✅ Sparse traffic"
    )
    
    # Model info
    config = AppConfig()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Configuration")
    st.sidebar.text(f"Model: {config.MODEL_NAME}")
    st.sidebar.text(f"Image Size: {config.IMAGE_SIZE}")
    st.sidebar.text(f"Classes: {len(config.CLASS_NAMES)}")
    
    # Main content header
    st.markdown('<h1 class="main-header">🚦 Traffic Anomaly Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Route to appropriate page
    if page == "📸 Image Detection":
        image_detection.show()
    elif page == "🎥 Video Detection":
        video_detection.show()
    elif page == "📁 Batch Analysis":
        batch_analysis.show()
    elif page == "📡 Real-Time Stream":
        realtime_stream.show()
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard.show()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by TensorFlow & Streamlit | Traffic Anomaly Detection v1.0"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()