"""
Visualization utilities for charts and plots
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.config import AppConfig

config = AppConfig()

def create_probability_bar_chart(probabilities, highlight_class=None):
    """
    Create horizontal bar chart for class probabilities
    
    Args:
        probabilities: Dictionary of class probabilities
        highlight_class: Class to highlight (optional)
    
    Returns:
        Plotly figure
    """
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = [
        'crimson' if cls == highlight_class else 'lightblue'
        for cls in classes
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Traffic Condition",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_pie_chart(class_distribution, title="Class Distribution"):
    """
    Create pie chart for class distribution
    
    Args:
        class_distribution: Dictionary of class counts
        title: Chart title
    
    Returns:
        Plotly figure
    """
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    
    colors_rgb = [config.get_color_rgb(cls) for cls in classes]
    colors_hex = [f'rgb({c[0]},{c[1]},{c[2]})' for c in colors_rgb]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=classes,
            values=counts,
            marker=dict(colors=colors_hex),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_timeline_chart(events, duration):
    """
    Create timeline chart for anomaly events
    
    Args:
        events: List of event dictionaries
        duration: Total video duration in seconds
    
    Returns:
        Plotly figure
    """
    if not events:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomaly events detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=300)
        return fig
    
    df = pd.DataFrame(events)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='timestamp',
        y='class',
        color='class',
        size='confidence',
        hover_data=['frame', 'confidence'],
        title="Anomaly Events Timeline",
        labels={'timestamp': 'Time (seconds)', 'class': 'Anomaly Type'}
    )
    
    fig.update_layout(
        height=400,
        xaxis=dict(range=[0, duration]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_confidence_trend(frame_predictions, window=50):
    """
    Create line chart showing confidence trends over time
    
    Args:
        frame_predictions: List of frame prediction dictionaries
        window: Moving average window size
    
    Returns:
        Plotly figure
    """
    if not frame_predictions:
        return go.Figure()
    
    df = pd.DataFrame(frame_predictions)
    
    fig = go.Figure()
    
    # Add line for each class
    for class_name in config.CLASS_NAMES:
        class_frames = df[df['class'] == class_name]
        
        if not class_frames.empty:
            fig.add_trace(go.Scatter(
                x=class_frames['timestamp'],
                y=class_frames['confidence'],
                mode='lines+markers',
                name=class_name.replace('_', ' ').title(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Confidence Trends Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence (%)",
        height=400,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_batch_summary_chart(results):
    """
    Create summary charts for batch analysis
    
    Args:
        results: List of prediction result dictionaries
    
    Returns:
        Plotly figure with subplots
    """
    if not results:
        return go.Figure()
    
    df = pd.DataFrame(results)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Class Distribution', 'Average Confidence by Class'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    # Pie chart - class distribution
    class_counts = df['class'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            textinfo='label+percent'
        ),
        row=1, col=1
    )
    
    # Bar chart - average confidence
    avg_confidence = df.groupby('class')['confidence'].mean()
    fig.add_trace(
        go.Bar(
            x=avg_confidence.index,
            y=avg_confidence.values,
            text=[f"{v:.1f}%" for v in avg_confidence.values],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_heatmap(class_distribution_by_time):
    """
    Create heatmap for temporal class distribution
    
    Args:
        class_distribution_by_time: Dictionary mapping time bins to class counts
    
    Returns:
        Plotly figure
    """
    # Create matrix for heatmap
    time_bins = sorted(class_distribution_by_time.keys())
    classes = config.CLASS_NAMES
    
    matrix = []
    for class_name in classes:
        row = [class_distribution_by_time[t].get(class_name, 0) for t in time_bins]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=time_bins,
        y=classes,
        colorscale='YlOrRd',
        hovertemplate='Time: %{x}<br>Class: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Traffic Conditions Over Time",
        xaxis_title="Time Period",
        yaxis_title="Anomaly Type",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_confusion_matrix(true_labels, predicted_labels):
    """
    Create confusion matrix visualization
    
    Args:
        true_labels: List of true class labels
        predicted_labels: List of predicted class labels
    
    Returns:
        Plotly figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=config.CLASS_NAMES)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=config.CLASS_NAMES,
        y=config.CLASS_NAMES,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def format_alert_message(prediction):
    """
    Format alert message based on prediction
    
    Args:
        prediction: Prediction dictionary
    
    Returns:
        Formatted alert string
    """
    severity = prediction['severity']
    class_name = prediction['class'].replace('_', ' ').title()
    emoji = prediction['emoji']
    confidence = prediction['confidence']
    
    if severity == 'critical':
        return f"{emoji} **CRITICAL ALERT: {class_name}** (Confidence: {confidence:.1f}%)"
    elif severity == 'warning':
        return f"{emoji} **WARNING: {class_name}** (Confidence: {confidence:.1f}%)"
    else:
        return f"{emoji} **{class_name}** (Confidence: {confidence:.1f}%)"