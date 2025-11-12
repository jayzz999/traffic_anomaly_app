# Traffic Anomaly Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trafficanomalyapp-izbd7u6jptrccushtfsl2m.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-ready application for real-time detection and analysis of traffic anomalies from images, videos, and live streams. The system leverages deep learning models (TensorFlow) to identify events such as accidents, fire incidents, dense traffic, and more.

## 🚀 Features

- **Multi-Source Input**: Analyze traffic from uploaded images, videos, or live RTSP streams
- **Real-Time Detection**: Process video streams in real-time with customizable frame rates
- **Deep Learning Models**: Powered by TensorFlow and pre-trained CNN models
- **Multiple Detection Types**:
  - Traffic accidents
  - Fire incidents
  - Dense traffic conditions
  - Road anomalies
- **Interactive Dashboard**: Built with Streamlit for easy visualization and analysis
- **Batch Analysis**: Process multiple images or video files simultaneously
- **Analytics & Reporting**: Generate comprehensive reports with detection statistics
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 📋 Prerequisites

- Python 3.11 or higher
- TensorFlow 2.x
- OpenCV
- FFmpeg (for video processing)
- CUDA-capable GPU (optional, for faster processing)

## 🔧 Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/jayzz999/traffic_anomaly_app.git
cd traffic_anomaly_app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the application at http://localhost:8501
```

### Using the Setup Script

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

## 📖 Usage

### Image Detection

1. Navigate to the **Image Detection** page
2. Upload one or more traffic images
3. Click "Analyze" to detect anomalies
4. View results with highlighted detections and confidence scores

### Video Detection

1. Navigate to the **Video Detection** page
2. Upload a video file or provide an RTSP stream URL
3. Adjust frame processing rate if needed
4. Start analysis and monitor real-time results
5. Download processed video with annotations

### Real-Time Stream Analysis

1. Navigate to the **Real-Time Stream** page
2. Enter your RTSP stream URL
3. Configure detection parameters
4. Start monitoring and receive instant alerts

### Batch Analysis

1. Navigate to the **Batch Analysis** page
2. Upload multiple files for processing
3. Review comprehensive analytics and reports
4. Export results in CSV or JSON format

## 🏗️ Project Structure

```
traffic_anomaly_app/
├── app.py                  # Main Streamlit application
├── models/                 # Pre-trained models and weights
├── pages/                  # Streamlit pages for different features
│   ├── image_detection.py
│   ├── video_detection.py
│   ├── realtime_stream.py
│   └── batch_analysis.py
├── utils/                  # Utility functions and helpers
│   ├── config.py
│   └── detection.py
├── .streamlit/             # Streamlit configuration
├── requirements.txt        # Python dependencies
├── DockerFile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # This file
```

## ⚙️ Configuration

The application can be configured through:

- **Environment Variables**: Set in `.streamlit/config.toml`
- **Model Parameters**: Adjust detection thresholds and confidence levels
- **Processing Settings**: Configure frame rates, batch sizes, and GPU usage

Edit the `.streamlit/config.toml` file to customize:

```toml
[server]
port = 8501
maxUploadSize = 500

[theme]
base = "dark"
primaryColor = "#667eea"
```

## 🧪 Testing

Run the installation test:

```bash
python test_installation.py
```

## 📊 Model Information

The system uses state-of-the-art deep learning models:

- **Architecture**: Convolutional Neural Networks (CNN)
- **Framework**: TensorFlow 2.x
- **Training Data**: Diverse traffic scenarios and anomaly types
- **Accuracy**: High precision and recall rates for anomaly detection

## 🚦 Performance

- **Image Processing**: ~50-100ms per image (GPU)
- **Video Processing**: Real-time at 15-30 FPS (GPU)
- **Stream Latency**: <500ms for live RTSP streams
- **Batch Processing**: Parallel processing for multiple files

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**jayzz999**

- GitHub: [@jayzz999](https://github.com/jayzz999)

## 🌟 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the interactive web framework
- OpenCV for computer vision capabilities
- The open-source community for various tools and libraries

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on [GitHub Issues](https://github.com/jayzz999/traffic_anomaly_app/issues)
- Check out the [Project Overview](PROJECT_OVERVIEW.md) for more details
- Review the [Quick Start Guide](QUICKSTART.md) for getting started
- See the [Migration Guide](MIGRATION_GUIDE.md) for upgrading

## 🔗 Links

- [Live Demo](https://trafficanomalyapp-izbd7u6jptrccushtfsl2m.streamlit.app/)
- [Documentation](PROJECT_OVERVIEW.md)
- [Quick Start](QUICKSTART.md)
- [License](LICENSE)

---

**Built with ❤️ using Python, TensorFlow, and Streamlit**
