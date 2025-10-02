# YOLO Artwork Recognition System

A comprehensive computer vision system for real-time artwork detection and recognition using custom-trained YOLO models. This project implements multiple deployment strategies including PC webcam inference, Raspberry Pi streaming, and dockerized solutions for robust artwork detection in gallery environments.

## Project Overview

This system uses custom-trained YOLOv8 models to detect and recognize three specific artworks:
- **"The Progress of a Soul: The Victory"** by Phoebe Anna Traquair (1902)
- **"The Execution of Lady Jane Grey"** by Paul Delaroche (1833)  
- **"Café Terrace at Night"** by Vincent van Gogh (1888)

The project supports multiple deployment configurations from simple PC testing to production-ready streaming solutions with Docker containerization.

## System Architecture

```
ISDN3001-YOLO-CV-Training/
├── Camera module/           # Main application code
├── runs/                   # YOLO training outputs
├── backup/                 # Dataset backups and experiments
├── requirements.txt        # Python dependencies
└── Model files (*.pt, *.onnx)
```

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- OpenCV
- Ultralytics YOLOv8
- Docker Desktop (for containerized deployment)

### Installation
```bash
git clone https://github.com/ChloeYuen895/ISDN3001-YOLO-CV-Training.git
cd ISDN3001-YOLO-CV-Training
pip install -r requirements.txt
```

## Deployment Options

### 1. PC Webcam Testing (`PC_artwork_recognition.py`)
Real-time artwork detection using your computer's webcam.

```python
# Quick test with PC camera
python "Camera module/PC_artwork_recognition.py"
```

**Features:**
- Uses secondary camera (index 1)
- Real-time inference at 75% confidence threshold
- CPU-optimized for testing
- 640px image resolution

### 2. Raspberry Pi Streaming (`mediamtx_raw/`)
Direct RTSP streaming from Raspberry Pi to PC for inference.

```python
# Connect to Pi stream
python "Camera module/mediamtx_raw/mediamtx_artwork_recognition.py"
```

**Configuration:**
- RTSP URL: `rtsp://192.168.1.108:8554/webcam`
- 30 FPS stream with buffer optimization
- 75% confidence, 70% IoU threshold

### 3. Docker MediaMTX Solution (`mediamtx_docker/`)
Production-ready containerized system with web API.

```bash
cd "Camera module/mediamtx_docker"
docker-compose up -d
```

**Services:**
- **MediaMTX Server**: RTSP streaming (port 8554)
- **Inference API**: Flask server with REST endpoints (port 5000)
- **Web Interface**: Real-time detection monitoring (port 8888)

**API Endpoints:**
- `GET /api/detections` - Current detections as JSON
- `GET /api/health` - System status  
- `GET /` - Web dashboard

## Model Training

### Training Scripts
The project includes multiple training iterations with progressive improvements:

#### AI_training_v3.py (Latest)
```python
# Enhanced training with regularization
python "Camera module/Training/AI_training_v3.py"
```

**Training Features:**
- YOLOv8 Nano base model
- 20 epochs with early stopping
- Advanced data augmentation
- GPU/CPU auto-detection
- Dropout and weight decay regularization
- ONNX export for Raspberry Pi

#### Key Training Parameters:
```yaml
epochs: 20
batch_size: 4
image_size: 640x640
patience: 5
confidence: 0.75
augmentations:
  - horizontal_flip: 0.5
  - vertical_flip: 0.5
  - rotation: ±15°
  - hsv_shifts: hue(±5%), sat(±70%), val(±40%)
  - perspective: 0.01
  - translation: ±10%
  - scale: ±50%
  - shear: ±2°
  - mosaic: 0.5
```

### Dataset Structure
```
Camera module/dataset/
├── data.yaml              # Dataset configuration
├── images/
│   ├── train/             # Training images (70%)
│   ├── val/               # Validation images (15%)
│   └── test/              # Test images (15%)
└── labels/
    ├── train/             # YOLO format annotations
    ├── val/
    └── test/
```

## Utilities & Tools

### Dataset Management
- **`split_dataset.py`**: Automatically splits images into train/val/test sets (70/15/15)
- **`autoannotatization.py`**: Semi-automatic annotation using pre-trained models

### Model Conversion
- **`pt_to_onnx_converter.py`**: Convert PyTorch models to ONNX for deployment

### System Checks
- **`check_gpu.py`**: Verify CUDA availability and GPU configuration
- **`test_camera.py`**: Test camera connectivity and settings

## Model Performance

### Training Results
- **Detection mAP50**: 0.85+ (validation set)
- **Inference Speed**: 
  - GPU: ~15ms per frame
  - CPU: ~45ms per frame
  - Raspberry Pi: ~120ms per frame
- **Model Size**: 6.2MB (ONNX format)

### Deployment Performance
- **PC Webcam**: Real-time 30 FPS processing
- **Pi Streaming**: 2 FPS inference (optimized for accuracy)
- **Docker System**: 2 FPS with REST API response < 50ms

## Configuration Files

### Docker Compose (`mediamtx_docker/docker-compose.yml`)
```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    ports: ["8554:8554", "8888:8888", "1935:1935"]
    volumes: ["./mediamtx.yml:/mediamtx.yml"]

  inference:
    image: python:3.9-slim
    ports: ["5000:5000"]
    volumes:
      - ./inference:/app
      - ../models:/models
    command: bash -c "pip install -r requirements.txt && python inference_server.py"
```

### MediaMTX Configuration (`mediamtx.yml`)
```yaml
rtspAddress: :8554
paths:
  cam1:
    source: publisher
    sourceProtocol: automatic
```

### Dataset Configuration (`data.yaml`)
```yaml
path: Camera module/dataset
train: images/train
val: images/val
test: images/test
nc: 3
names:
  - 'The Progress of a Soul: The Victory, Phoebe Anna Traquair 1902'
  - 'The Execution of Lady Jane Grey, Paul Delaroche 1833'
  - 'Café Terrace at Night, Vincent van Gogh 1888'
```

## Production Deployment

### Raspberry Pi Setup
1. **Install dependencies:**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

2. **Create streaming script:**
```bash
#!/bin/bash
LAPTOP_IP="192.168.1.100"  # Your PC IP
ffmpeg -f v4l2 -input_format h264 -video_size 1280x720 -framerate 30 \
       -i /dev/video0 -c copy -f rtsp -rtsp_transport tcp \
       "rtsp://$LAPTOP_IP:8554/cam1"
```

3. **Enable auto-start service:**
```bash
sudo systemctl enable camera-stream.service
```

### Windows PC Setup
1. **Start Docker services:**
```cmd
cd "Camera module\mediamtx_docker"
docker-compose up -d
```

2. **Verify services:**
- MediaMTX: http://localhost:8888/
- Inference API: http://localhost:5000/
- Stream test: `rtsp://localhost:8554/cam1` in VLC

## Monitoring & Debugging

### Check System Status
```bash
# Docker container status
docker ps

# View logs
docker logs mediamtx
docker logs yolo-inference -f

# API health check
curl http://localhost:5000/api/health
```

### Performance Monitoring
- **Detection Rate**: 2 inferences/second (configurable)
- **Stream Latency**: ~200-500ms
- **Memory Usage**: ~2GB (with Docker)
- **CPU Usage**: 15-30% (inference mode)

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Verify model file exists
ls -la "Camera module/models/best_detect.onnx"
```

**2. Stream Connection Failed**
```bash
# Check network connectivity
ping 192.168.1.108  # Pi IP
telnet localhost 8554  # MediaMTX port
```

**3. GPU Not Detected**
```python
# Run GPU check
python "Camera module/checkers/check_gpu.py"
```

**4. Low Detection Accuracy**
- Adjust confidence threshold (default: 0.75)
- Improve lighting conditions
- Retrain with more diverse dataset

## Dependencies

### Core Libraries
```txt
ultralytics==8.2.0
opencv-python==4.10.0
torch>=2.0.0
onnxruntime>=1.18.0
numpy>=1.24.0
flask>=2.3.0
```
