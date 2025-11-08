"""
Batch Analysis Page
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import zipfile
import tempfile
from pathlib import Path
from utils import (
    load_trained_model,
    predict_batch_images,
    create_batch_summary_chart,
    create_pie_chart,
    AppConfig
)

config = AppConfig()

def show():
    """Display batch analysis page"""
    
    st.header("📁 Batch Image Analysis")
    st.markdown("Upload multiple images or a ZIP file for batch processing")
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.error("❌ Model not loaded. Please check the setup instructions.")
        return
    
    # Upload options
    upload_mode = st.radio(
        "Upload Mode",
        ["Multiple Images", "ZIP File"],
        horizontal=True
    )
    
    if upload_mode == "Multiple Images":
        uploaded_files = st.file_uploader(
            "Upload Traffic Images",
            type=config.SUPPORTED_IMAGE_FORMATS,
            accept_multiple_files=True,
            help=f"Maximum {config.MAX_BATCH_SIZE} images"
        )
        
        if uploaded_files:
            if len(uploaded_files) > config.MAX_BATCH_SIZE:
                st.error(f"❌ Too many files. Maximum: {config.MAX_BATCH_SIZE} images")
                return
            
            process_uploaded_files(uploaded_files, model)
    
    else:  # ZIP File
        uploaded_zip = st.file_uploader(
            "Upload ZIP File",
            type=['zip'],
            help="ZIP file containing traffic images"
        )
        
        if uploaded_zip:
            process_zip_file(uploaded_zip, model)

def process_uploaded_files(uploaded_files, model):
    """Process multiple uploaded files"""
    
    st.info(f"📊 {len(uploaded_files)} images uploaded")
    
    # Process button
    if st.button("🚀 Analyze All Images", type="primary", use_container_width=True):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        images_data = []
        
        # Process each image
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({idx+1}/{len(uploaded_files)})")
            
            # Load image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert RGB to BGR
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            
            # Predict
            from utils.model_utils import predict_single_image
            result = predict_single_image(model, image_bgr)
            
            # Store results
            result['filename'] = uploaded_file.name
            results.append(result)
            images_data.append({
                'filename': uploaded_file.name,
                'image': image_array,
                'result': result
            })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Processing complete!")
        
        # Display results
        display_batch_results(results, images_data)
        
        # Store in session state
        st.session_state['batch_results'] = results
        st.session_state['batch_images'] = images_data
    
    # Display previous results if available
    elif 'batch_results' in st.session_state:
        st.info("Showing previous results. Click 'Analyze All Images' to reprocess.")
        display_batch_results(
            st.session_state['batch_results'],
            st.session_state['batch_images']
        )

def process_zip_file(uploaded_zip, model):
    """Process ZIP file containing images"""
    
    # Extract ZIP
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        
        # Find all image files
        image_files = []
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(temp_path.rglob(f"*.{ext}"))
            image_files.extend(temp_path.rglob(f"*.{ext.upper()}"))
        
        if not image_files:
            st.error("❌ No valid images found in ZIP file")
            return
        
        if len(image_files) > config.MAX_BATCH_SIZE:
            st.warning(f"⚠️ Found {len(image_files)} images. Processing first {config.MAX_BATCH_SIZE}...")
            image_files = image_files[:config.MAX_BATCH_SIZE]
        
        st.info(f"📊 Found {len(image_files)} images in ZIP file")
        
        # Process button
        if st.button("🚀 Analyze All Images", type="primary", use_container_width=True):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            images_data = []
            
            # Process each image
            for idx, image_path in enumerate(image_files):
                status_text.text(f"Processing {image_path.name}... ({idx+1}/{len(image_files)})")
                
                # Load image
                image_bgr = cv2.imread(str(image_path))
                
                if image_bgr is None:
                    continue
                
                # Predict
                from utils.model_utils import predict_single_image
                result = predict_single_image(model, image_bgr)
                
                # Store results
                result['filename'] = image_path.name
                results.append(result)
                
                # Convert BGR to RGB for display
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                images_data.append({
                    'filename': image_path.name,
                    'image': image_rgb,
                    'result': result
                })
                
                # Update progress
                progress_bar.progress((idx + 1) / len(image_files))
            
            status_text.text("✅ Processing complete!")
            
            # Display results
            display_batch_results(results, images_data)
            
            # Store in session state
            st.session_state['batch_results'] = results
            st.session_state['batch_images'] = images_data

def display_batch_results(results, images_data):
    """Display batch processing results"""
    
    # Summary statistics
    st.markdown("### 📊 Batch Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Count by class
    class_counts = {}
    for result in results:
        class_name = result['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Display metrics
    metrics = [col1, col2, col3, col4]
    for idx, (class_name, count) in enumerate(class_counts.items()):
        if idx < 4:
            emoji = config.get_emoji(class_name)
            metrics[idx].metric(
                class_name.replace('_', ' ').title(),
                count,
                delta=emoji
            )
    
    # Charts
    st.markdown("### 📈 Analysis Charts")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Class distribution pie chart
        pie_fig = create_pie_chart(class_counts, title="Class Distribution")
        st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        # Summary chart
        summary_fig = create_batch_summary_chart(results)
        st.plotly_chart(summary_fig, use_container_width=True)
    
    # Image grid with results
    st.markdown("### 🖼️ Processed Images")
    
    # Filter options
    filter_class = st.selectbox(
        "Filter by class",
        ["All"] + config.CLASS_NAMES,
        key="batch_filter"
    )
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["Filename", "Confidence (High to Low)", "Confidence (Low to High)", "Class"],
        key="batch_sort"
    )
    
    # Apply filters and sorting
    filtered_data = images_data.copy()
    
    if filter_class != "All":
        filtered_data = [d for d in filtered_data if d['result']['class'] == filter_class]
    
    if sort_by == "Confidence (High to Low)":
        filtered_data.sort(key=lambda x: x['result']['confidence'], reverse=True)
    elif sort_by == "Confidence (Low to High)":
        filtered_data.sort(key=lambda x: x['result']['confidence'])
    elif sort_by == "Class":
        filtered_data.sort(key=lambda x: x['result']['class'])
    else:  # Filename
        filtered_data.sort(key=lambda x: x['filename'])
    
    # Display grid
    cols_per_row = 3
    for i in range(0, len(filtered_data), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(filtered_data):
                data = filtered_data[idx]
                result = data['result']
                
                with cols[j]:
                    # Display image
                    st.image(data['image'], use_column_width=True)
                    
                    # Display info
                    emoji = result['emoji']
                    class_name = result['class'].replace('_', ' ').title()
                    confidence = result['confidence']
                    
                    # Color-coded box
                    severity = result['severity']
                    if severity == 'critical':
                        st.error(f"{emoji} {class_name}\n{confidence:.1f}%")
                    elif severity == 'warning':
                        st.warning(f"{emoji} {class_name}\n{confidence:.1f}%")
                    else:
                        st.success(f"{emoji} {class_name}\n{confidence:.1f}%")
                    
                    st.caption(data['filename'])
    
    # Detailed results table
    with st.expander("📋 Detailed Results Table"):
        df = pd.DataFrame([{
            'Filename': r['filename'],
            'Class': r['class'].replace('_', ' ').title(),
            'Confidence': f"{r['confidence']:.2f}%",
            'Severity': r['severity'].title()
        } for r in results])
        
        st.dataframe(df, use_container_width=True)
    
    # Export options
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        df_export = pd.DataFrame([{
            'filename': r['filename'],
            'class': r['class'],
            'confidence': r['confidence'],
            'severity': r['severity'],
            **{f'prob_{k}': v for k, v in r['probabilities'].items()}
        } for r in results])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="batch_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Summary report
        report_text = generate_batch_report(results, class_counts)
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name="batch_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # JSON export
        import json
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="batch_analysis_results.json",
            mime="application/json",
            use_container_width=True
        )

def generate_batch_report(results, class_counts):
    """Generate text report for batch analysis"""
    
    total = len(results)
    avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0
    
    report = f"""
Traffic Anomaly Detection - Batch Analysis Report
{'='*60}

SUMMARY:
- Total Images Processed: {total}
- Average Confidence: {avg_confidence:.2f}%

CLASS DISTRIBUTION:
"""
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        report += f"- {class_name.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
    
    # Critical/Warning counts
    critical_count = sum(1 for r in results if r['severity'] == 'critical')
    warning_count = sum(1 for r in results if r['severity'] == 'warning')
    
    report += f"\nSEVERITY ANALYSIS:\n"
    report += f"- Critical: {critical_count} ({critical_count/total*100:.1f}%)\n"
    report += f"- Warning: {warning_count} ({warning_count/total*100:.1f}%)\n"
    report += f"- Normal: {total-critical_count-warning_count} ({(total-critical_count-warning_count)/total*100:.1f}%)\n"
    
    report += f"\n\nReport generated by Traffic Anomaly Detection System v1.0\n"
    
    return report