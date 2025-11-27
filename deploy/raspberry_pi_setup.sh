#!/bin/bash
# Raspberry Pi Setup Script for Traffic Anomaly Detection Edge Device
# Compatible with Raspberry Pi 4 and newer

set -e

echo "=========================================="
echo "Traffic Anomaly Detection - Edge Setup"
echo "RTA Dubai Integration"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This script is designed for Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    sqlite3 \
    git \
    wget \
    curl

# Install camera support
echo -e "${GREEN}Installing camera support...${NC}"
sudo apt-get install -y \
    libraspberrypi0 \
    libraspberrypi-dev \
    libcamera-dev \
    v4l-utils

# Enable camera interface (for legacy cameras)
if command -v raspi-config &> /dev/null; then
    echo -e "${GREEN}Enabling camera interface...${NC}"
    sudo raspi-config nonint do_camera 0 || true
fi

# Create application directory
APP_DIR="/opt/traffic-anomaly"
echo -e "${GREEN}Creating application directory at ${APP_DIR}...${NC}"
sudo mkdir -p ${APP_DIR}
sudo chown $USER:$USER ${APP_DIR}

# Copy application files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo -e "${GREEN}Copying application files...${NC}"
cp -r ${SCRIPT_DIR}/* ${APP_DIR}/

# Create Python virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
cd ${APP_DIR}
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install TensorFlow Lite for Raspberry Pi (lighter than full TensorFlow)
echo -e "${GREEN}Installing TensorFlow Lite...${NC}"
pip install tflite-runtime

# Install other Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
cat > requirements-edge.txt << 'EOF'
numpy>=1.21.0,<2.0
opencv-python-headless>=4.5.0
Pillow>=9.0.0
pywavelets>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.60.0
python-dateutil>=2.8.0
EOF

pip install -r requirements-edge.txt

# Create data directories
echo -e "${GREEN}Creating data directories...${NC}"
mkdir -p ${APP_DIR}/data
mkdir -p ${APP_DIR}/logs
mkdir -p ${APP_DIR}/models

# Create environment configuration file
echo -e "${GREEN}Creating environment configuration...${NC}"
cat > ${APP_DIR}/.env << 'EOF'
# Edge Device Configuration
EDGE_DEVICE_ID=rpi-edge-001
CAMERA_ID=cam-001
LOCATION_ID=intersection-001

# Camera Settings
CAMERA_SOURCE=0
# For RTSP: CAMERA_SOURCE=rtsp://username:password@ip:port/stream

# RTA Dubai API Configuration
RTA_API_URL=https://api.rta.ae/traffic/v1
RTA_AUTH_URL=https://auth.rta.ae/oauth2/token
RTA_CLIENT_ID=
RTA_CLIENT_SECRET=
RTA_API_KEY=

# Location Details
INTERSECTION_NAME=Traffic Intersection
CAMERA_LAT=25.2048
CAMERA_LON=55.2708

# Processing Settings
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=-1
EOF

# Create systemd service
echo -e "${GREEN}Creating systemd service...${NC}"
sudo cat > /etc/systemd/system/traffic-anomaly.service << EOF
[Unit]
Description=Traffic Anomaly Detection Edge Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/edge_runner.py --camera \${CAMERA_SOURCE}
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/logs/edge_runner.log
StandardError=append:${APP_DIR}/logs/edge_runner.error.log

[Install]
WantedBy=multi-user.target
EOF

# Create log rotation config
echo -e "${GREEN}Configuring log rotation...${NC}"
sudo cat > /etc/logrotate.d/traffic-anomaly << EOF
${APP_DIR}/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
}
EOF

# Enable and start service
echo -e "${GREEN}Enabling and starting service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable traffic-anomaly.service

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit ${APP_DIR}/.env with your RTA API credentials"
echo "2. Copy your trained model to ${APP_DIR}/models/"
echo "3. Start the service: sudo systemctl start traffic-anomaly"
echo "4. Check status: sudo systemctl status traffic-anomaly"
echo "5. View logs: tail -f ${APP_DIR}/logs/edge_runner.log"
echo ""
echo "Useful commands:"
echo "  sudo systemctl start traffic-anomaly    # Start service"
echo "  sudo systemctl stop traffic-anomaly     # Stop service"
echo "  sudo systemctl restart traffic-anomaly  # Restart service"
echo "  sudo journalctl -u traffic-anomaly -f   # View live logs"
echo ""
