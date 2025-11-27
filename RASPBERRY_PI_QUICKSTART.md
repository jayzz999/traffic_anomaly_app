# Raspberry Pi Quick Start Guide
## Traffic Anomaly Detection with RTA Dubai Integration

### Prerequisites

- Raspberry Pi 4 (4GB+ RAM recommended)
- MicroSD card (32GB+ recommended)
- Raspberry Pi OS (64-bit recommended)
- USB webcam or Raspberry Pi Camera Module
- Stable internet connection

---

## Step 1: Initial Setup

### 1.1 Update your Raspberry Pi
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Clone/Copy the project
```bash
# If using git
git clone <your-repo-url> /home/pi/traffic_anomaly_app
cd /home/pi/traffic_anomaly_app

# Or copy files via SCP from your Mac
# scp -r /Users/jayanthmuthina/Desktop/traffic_anomaly_app pi@<raspberry-pi-ip>:/home/pi/
```

### 1.3 Run the setup script
```bash
chmod +x deploy/raspberry_pi_setup.sh
./deploy/raspberry_pi_setup.sh
```

---

## Step 2: Configure the Edge Device

### 2.1 Edit the environment file
```bash
nano /opt/traffic-anomaly/.env
```

### 2.2 Set your configuration
```env
# Device identification
EDGE_DEVICE_ID=rpi-edge-001
CAMERA_ID=cam-dubai-001
LOCATION_ID=szr-trade-centre

# Camera source
CAMERA_SOURCE=0                    # USB camera
# CAMERA_SOURCE=/dev/video0        # Alternative
# CAMERA_SOURCE=rtsp://user:pass@ip:554/stream  # IP camera

# RTA Dubai API credentials
RTA_API_URL=https://api.rta.ae/traffic/v1
RTA_API_KEY=your_api_key_here

# Location
INTERSECTION_NAME=Sheikh Zayed Road - Trade Centre
CAMERA_LAT=25.2285
CAMERA_LON=55.2833
```

---

## Step 3: Copy the Model

Copy your trained model to the Raspberry Pi:
```bash
# From your Mac
scp /Users/jayanthmuthina/Desktop/traffic_anomaly_app/models/improved_model.h5 \
    pi@<raspberry-pi-ip>:/opt/traffic-anomaly/models/
```

---

## Step 4: Test the Setup

### 4.1 Test camera access
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera capture
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL'); cap.release()"
```

### 4.2 Run a quick test
```bash
cd /opt/traffic-anomaly
source venv/bin/activate
python edge_runner.py --camera 0 --debug
```

Press `Ctrl+C` to stop after verifying it works.

---

## Step 5: Start as a Service

### 5.1 Start the service
```bash
sudo systemctl start traffic-anomaly
```

### 5.2 Check status
```bash
sudo systemctl status traffic-anomaly
```

### 5.3 View logs
```bash
# Live logs
tail -f /opt/traffic-anomaly/logs/edge_runner.log

# Or using journalctl
sudo journalctl -u traffic-anomaly -f
```

---

## Useful Commands

| Command | Description |
|---------|-------------|
| `sudo systemctl start traffic-anomaly` | Start the service |
| `sudo systemctl stop traffic-anomaly` | Stop the service |
| `sudo systemctl restart traffic-anomaly` | Restart the service |
| `sudo systemctl status traffic-anomaly` | Check service status |
| `sudo systemctl enable traffic-anomaly` | Enable auto-start on boot |
| `sudo journalctl -u traffic-anomaly -f` | View live logs |

---

## Network Status Indicators

The system automatically adapts based on network quality:

| Status | Latency | Behavior |
|--------|---------|----------|
| **EXCELLENT** | <50ms | Full quality, real-time transmission |
| **GOOD** | <100ms | High quality, real-time transmission |
| **FAIR** | <200ms | Medium compression, batched every 5s |
| **POOR** | <500ms | Heavy compression, priority frames only |
| **CRITICAL** | >500ms | Minimal quality, batched every 30s |
| **OFFLINE** | No connection | Store locally, sync when back online |

---

## Troubleshooting

### Camera not detected
```bash
# Check if camera is connected
lsusb

# Check video devices
ls -la /dev/video*

# Add user to video group
sudo usermod -a -G video $USER
```

### Permission denied errors
```bash
sudo chown -R $USER:$USER /opt/traffic-anomaly
```

### Model loading issues
```bash
# Check model file exists
ls -la /opt/traffic-anomaly/models/

# Test model loading
python3 -c "from utils.model_utils import load_trained_model; print(load_trained_model())"
```

### Network connectivity issues
```bash
# Test internet connection
ping -c 3 8.8.8.8

# Test RTA API (replace with actual endpoint)
curl -I https://api.rta.ae/health
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI EDGE DEVICE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ Camera  │───▶│ Edge Runner  │───▶│ Anomaly Model   │   │
│  └─────────┘    └──────────────┘    └─────────────────┘   │
│                        │                     │             │
│                        ▼                     ▼             │
│              ┌─────────────────┐    ┌──────────────┐      │
│              │ Keyframe        │    │ Detection    │      │
│              │ Extractor       │    │ Result       │      │
│              └─────────────────┘    └──────────────┘      │
│                        │                     │             │
│                        ▼                     ▼             │
│              ┌─────────────────┐    ┌──────────────┐      │
│              │ Adaptive        │    │ RTA Event    │      │
│              │ Compressor      │    │ Reporter     │      │
│              └─────────────────┘    └──────────────┘      │
│                        │                     │             │
│                        ▼                     ▼             │
│              ┌─────────────────────────────────────┐      │
│              │        Network Monitor              │      │
│              │   (Checks connectivity & quality)   │      │
│              └─────────────────────────────────────┘      │
│                        │                                   │
│           ┌────────────┼────────────┐                     │
│           ▼            ▼            ▼                     │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│     │ Online   │ │ Degraded │ │ Offline  │               │
│     │ Send Now │ │ Compress │ │ Store    │               │
│     └──────────┘ └──────────┘ └──────────┘               │
│                                     │                     │
│                              ┌──────────────┐             │
│                              │ SQLite Queue │             │
│                              │ (Offline)    │             │
│                              └──────────────┘             │
│                                     │                     │
│                                     │ (When online)       │
│                                     ▼                     │
└─────────────────────────────────────┼─────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         RTA DUBAI SERVER        │
                    │   https://api.rta.ae/traffic    │
                    └─────────────────────────────────┘
```

---

## Support

For issues or questions:
1. Check the logs: `tail -f /opt/traffic-anomaly/logs/edge_runner.log`
2. Run in debug mode: `python edge_runner.py --debug`
3. Check system resources: `htop`
