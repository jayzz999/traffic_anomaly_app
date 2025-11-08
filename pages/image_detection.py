"""
Single Image Detection Page
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from utils import (
    load_trained_model,
    predict_single_image,
    validate_image,
    create_probability_bar_chart,
    format_alert_message,
    AppConfig
)

config = AppConfig()

def show():
    """Display single image detection page"""
    
    st.header("📸 Single Image Detection")
    st.markdown("Upload a traffic image to detect anomalies in real-time")
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.error("❌ Model not loaded. Please check the setup instructions.")
        return
    
    # Create layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a traffic image...",
            type=config.SUPPORTED_IMAGE_FORMATS,
            help="Supported formats: " + ", ".join(config.SUPPORTED_IMAGE_FORMATS)
        )
        
        # Camera input option
        st.markdown("**OR**")
        use_camera = st.checkbox("📷 Use Camera")
        
        camera_image = None
        if use_camera:
            camera_image = st.camera_input("Take a picture")
        
        # Determine which image to use
        image_source = camera_image if camera_image else uploaded_file
        
        if image_source:
            # Load image
            image = Image.open(image_source)
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            
            # Display original image
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image info
            with st.expander("ℹ️ Image Information"):
                st.write(f"**Dimensions:** {image_array.shape[1]} x {image_array.shape[0]} pixels")
                st.write(f"**Channels:** {image_array.shape[2] if len(image_array.shape) == 3 else 1}")
                st.write(f"**Size:** {image_source.size if hasattr(image_source, 'size') else 'N/A'} bytes")
            
            # Validate image
            is_valid, error_msg = validate_image(image_bgr)
            
            if not is_valid:
                st.error(f"❌ Invalid image: {error_msg}")
                return
            
            # Analyze button
            analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
            
            if analyze_button:
                with st.spinner("🔄 Analyzing traffic scene..."):
                    # Predict
                    result = predict_single_image(model, image_bgr)
                
                # Store result in session state
                st.session_state['last_result'] = result
                st.session_state['last_image'] = image_array
    
    with col2:
        st.subheader("Analysis Results")
        
        # Display results if available
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Alert display
            severity = result['severity']
            class_name = result['class'].replace('_', ' ').title()
            emoji = result['emoji']
            confidence = result['confidence']
            
            if severity == 'critical':
                st.error(f"{emoji} **CRITICAL ALERT: {class_name}**")
            elif severity == 'warning':
                st.warning(f"{emoji} **WARNING: {class_name}**")
            else:
                st.success(f"{emoji} **{class_name}**")
            
            # Confidence metric
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%",
                delta=None
            )
            
            # Probability chart
            st.markdown("### Class Probabilities")
            fig = create_probability_bar_chart(
                result['probabilities'],
                highlight_class=result['class']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            with st.expander("📊 Detailed Probabilities"):
                for cls, prob in sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    st.write(f"**{cls.replace('_', ' ').title()}:** {prob:.2f}%")
            
            # Recommendations
            with st.expander("💡 Recommendations"):
                if result['class'] == 'accident':
                    st.warning("""
                    **Action Required:**
                    - Alert emergency services immediately
                    - Deploy traffic management team
                    - Activate warning signs for approaching vehicles
                    - Clear adjacent lanes if possible
                    """)
                elif result['class'] == 'fire':
                    st.error("""
                    **Critical Action Required:**
                    - Alert fire department immediately
                    - Evacuate nearby areas
                    - Block all traffic access
                    - Prepare emergency response team
                    """)
                elif result['class'] == 'dense_traffic':
                    st.info("""
                    **Traffic Management:**
                    - Activate traffic signal optimization
                    - Consider opening additional lanes
                    - Alert commuters via traffic apps
                    - Monitor for potential escalation
                    """)
                else:
                    st.success("""
                    **Normal Operations:**
                    - Continue routine monitoring
                    - No immediate action required
                    - Maintain standard protocols
                    """)
            
            # Download results
            st.markdown("### 📥 Export Results")
            results_text = f"""
Traffic Anomaly Detection Report
================================
Detection Class: {class_name}
Confidence: {confidence:.2f}%
Severity: {severity.upper()}

Detailed Probabilities:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v:.2f}%" for k, v in result['probabilities'].items()])}
            """
            
            st.download_button(
                label="Download Report (TXT)",
                data=results_text,
                file_name="anomaly_detection_report.txt",
                mime="text/plain"
            )
        
        else:
            # Placeholder
            st.info("👆 Upload an image or take a picture to begin analysis")
            
            # Example predictions
            st.markdown("### Expected Classes")
            for class_name in config.CLASS_NAMES:
                emoji = config.get_emoji(class_name)
                st.write(f"{emoji} **{class_name.replace('_', ' ').title()}**")