"""
Edge Device Monitor Page
Streamlit page for monitoring edge devices and RTA integration status.
"""

import streamlit as st
import time
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Edge Device Monitor",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Edge Device Monitor")
st.markdown("Monitor edge devices, network status, and RTA Dubai integration")

# Sidebar configuration
st.sidebar.header("Edge Device Settings")

# RTA Configuration
st.sidebar.subheader("RTA Dubai API")
rta_api_url = st.sidebar.text_input(
    "API URL",
    value="https://api.rta.ae/traffic/v1",
    help="RTA Traffic API base URL"
)
rta_api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your RTA API key"
)

# Device Configuration
st.sidebar.subheader("Device Settings")
camera_id = st.sidebar.text_input("Camera ID", value="cam-001")
location_id = st.sidebar.text_input("Location ID", value="intersection-001")
intersection_name = st.sidebar.text_input("Intersection Name", value="Traffic Intersection")

# Coordinates
col1, col2 = st.sidebar.columns(2)
with col1:
    latitude = st.number_input("Latitude", value=25.2048, format="%.4f")
with col2:
    longitude = st.number_input("Longitude", value=55.2708, format="%.4f")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Status Overview",
    "🌐 Network Monitor",
    "📤 Transmission Queue",
    "⚙️ Configuration"
])

with tab1:
    st.header("System Status Overview")

    # Status cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Network Status",
            value="Online",
            delta="Excellent",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Queue Size",
            value="0",
            delta="Empty",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Frames Sent",
            value="1,234",
            delta="+56 last min"
        )

    with col4:
        st.metric(
            label="Events Reported",
            value="12",
            delta="+2 today"
        )

    st.divider()

    # Connection status
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔌 Connection Status")

        connection_data = {
            "RTA API": {"status": "Connected", "latency": "45ms"},
            "Camera": {"status": "Active", "fps": "30"},
            "Storage": {"status": "Healthy", "used": "12%"}
        }

        for name, info in connection_data.items():
            with st.container():
                cols = st.columns([2, 1, 1])
                cols[0].write(f"**{name}**")
                cols[1].write(info["status"])
                cols[2].write(list(info.values())[1])

    with col2:
        st.subheader("📈 Recent Activity")

        activity_data = [
            {"time": "10:45:32", "event": "Frame batch sent", "count": 10},
            {"time": "10:45:30", "event": "Keyframe extracted", "count": 1},
            {"time": "10:45:25", "event": "Anomaly detected", "count": 1},
            {"time": "10:45:20", "event": "Frame batch sent", "count": 10},
        ]

        for activity in activity_data:
            st.text(f"{activity['time']} - {activity['event']} ({activity['count']})")

with tab2:
    st.header("Network Monitor")

    # Network quality indicator
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Current Status")
        st.info("**EXCELLENT** - Full quality transmission enabled")

        st.write("**Metrics:**")
        st.write("- Latency: 45ms")
        st.write("- Packet Loss: 0%")
        st.write("- Bandwidth: ~10 Mbps")

    with col2:
        st.subheader("Transmission Mode")
        st.success("**REALTIME**")

        st.write("**Settings:**")
        st.write("- Compression: HIGH (80%)")
        st.write("- Batch Size: 1")
        st.write("- Priority Only: No")

    with col3:
        st.subheader("Server Health")
        st.success("**Primary Server Active**")

        st.write("**Endpoints:**")
        st.code(rta_api_url)
        st.write("Last heartbeat: 30s ago")

    st.divider()

    # Network history chart placeholder
    st.subheader("Network Quality History (Last Hour)")

    # Simulated chart data
    import random
    chart_data = {
        "Time": [f"{i}:00" for i in range(60)],
        "Latency (ms)": [random.randint(30, 80) for _ in range(60)],
        "Quality Score": [random.randint(80, 100) for _ in range(60)]
    }

    st.line_chart(chart_data, x="Time", y=["Latency (ms)", "Quality Score"])

with tab3:
    st.header("Transmission Queue")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Priority Queue")
        st.metric("Items", "0", "Empty")

        st.write("Priority queue contains anomaly frames (accidents, fires)")

        if st.button("Force Flush Priority Queue"):
            st.success("Priority queue flushed!")

    with col2:
        st.subheader("Normal Queue")
        st.metric("Items", "0", "Empty")

        st.write("Normal queue contains regular keyframes")

        if st.button("Force Flush Normal Queue"):
            st.success("Normal queue flushed!")

    st.divider()

    # Offline storage status
    st.subheader("Offline Storage")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pending Frames", "0")

    with col2:
        st.metric("Pending Events", "0")

    with col3:
        st.metric("Storage Used", "12 MB / 500 MB")

    # Sync status
    st.subheader("Sync Status")
    st.progress(100)
    st.write("✅ All data synchronized")

with tab4:
    st.header("Edge Processing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Keyframe Extraction")

        scene_threshold = st.slider(
            "Scene Change Threshold",
            min_value=10.0,
            max_value=100.0,
            value=30.0,
            help="Threshold for detecting scene changes"
        )

        motion_threshold = st.slider(
            "Motion Threshold",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            help="Threshold for motion detection"
        )

        min_interval = st.number_input(
            "Min Keyframe Interval (frames)",
            min_value=10,
            max_value=100,
            value=30
        )

        max_interval = st.number_input(
            "Max Keyframe Interval (frames)",
            min_value=50,
            max_value=500,
            value=150
        )

    with col2:
        st.subheader("Compression Settings")

        compression_level = st.selectbox(
            "Default Compression",
            options=["full", "high", "medium", "low", "minimal"],
            index=1
        )

        image_format = st.selectbox(
            "Image Format",
            options=["jpeg", "webp"],
            index=0
        )

        st.write("")
        st.write("**Compression Levels:**")
        st.write("- **Full**: 95% quality, original resolution")
        st.write("- **High**: 80% quality, original resolution")
        st.write("- **Medium**: 60% quality, 75% resolution")
        st.write("- **Low**: 40% quality, 50% resolution")
        st.write("- **Minimal**: 20% quality, 25% resolution, grayscale")

    st.divider()

    # Generate configuration
    st.subheader("Generated Configuration")

    config_dict = {
        "device": {
            "camera_id": camera_id,
            "location_id": location_id,
            "intersection_name": intersection_name,
            "coordinates": [latitude, longitude]
        },
        "rta": {
            "api_url": rta_api_url,
            "api_key": "***" if rta_api_key else ""
        },
        "keyframe": {
            "scene_threshold": scene_threshold,
            "motion_threshold": motion_threshold,
            "min_interval": min_interval,
            "max_interval": max_interval
        },
        "compression": {
            "level": compression_level,
            "format": image_format
        }
    }

    st.json(config_dict)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved!")

    with col2:
        st.download_button(
            "📥 Download as JSON",
            data=json.dumps(config_dict, indent=2),
            file_name="edge_config.json",
            mime="application/json"
        )

# Footer
st.divider()
st.caption("Traffic Anomaly Detection System - Edge Device Monitor | RTA Dubai Integration")
