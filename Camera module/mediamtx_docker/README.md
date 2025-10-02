# Artwork Detection System with MediaMTX

A real-time artwork detection system that streams camera footage from a Raspberry Pi to a Windows laptop and performs object detection using a custom YOLO model to identify specific artworks on walls.

## Overview

This system uses:
- MediaMTX as the RTSP streaming server
- Custom YOLO ONNX model for artwork detection
- Flask API for real-time detection results
- Docker containers for easy deployment
- Raspberry Pi as the camera stream source

## Project Structure

```
Camera module/
└── mediamtx_docker/
    ├── docker-compose.yml
    ├── mediamtx.yml
    ├── inference/
    │   ├── requirements.txt
    │   └── inference_server.py
    └── models/
        └── best_detect.onnx (your custom model)
```

## Quick Start

### Prerequisites
- Windows laptop with Docker Desktop
- Raspberry Pi with camera module
- Both devices on same WiFi network
- Custom artwork detection model (`best_detect.onnx`)

### 1. Windows Laptop Setup

#### Clone and setup the project:
```cmd
cd "C:\Users\yueny\OneDrive\Documents\ISDN3001\Camera module\mediamtx_docker"
```

#### Start the services:
```cmd
docker-compose up -d
```

#### Verify services are running:
```cmd
docker ps
```

### 2. Raspberry Pi Setup

#### Connect to your Raspberry Pi:
```bash
ssh pi@raspberrypi.local
```

#### Install dependencies:
```bash
sudo apt update && sudo apt install -y ffmpeg
```

#### Create stream script:
```bash
nano ~/start_camera_stream.sh
```

Add the following content (replace with your laptop's IP):
```bash
#!/bin/bash
LAPTOP_IP="192.168.1.100"  # REPLACE WITH YOUR LAPTOP'S IP
STREAM_URL="rtsp://$LAPTOP_IP:8554/cam1"

ffmpeg -f v4l2 \
       -input_format h264 \
       -video_size 1280x720 \
       -framerate 30 \
       -i /dev/video0 \
       -c copy \
       -f rtsp \
       -rtsp_transport tcp \
       "$STREAM_URL"
```

#### Make executable and run:
```bash
chmod +x ~/start_camera_stream.sh
~/start_camera_stream.sh
```

## System Verification

### Check Services:

1. MediaMTX Status: http://localhost:8888/
2. Artwork Detection Server: http://localhost:5000/
3. Detection API: http://localhost:5000/api/detections
4. System Health: http://localhost:5000/api/health

### Test Stream:
- Open VLC Media Player
- Go to: Media -> Open Network Stream
- Enter: `rtsp://localhost:8554/cam1`

## Configuration Files

### docker-compose.yml
```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    ports: ["8554:8554", "8888:8888", "1935:1935"]
    volumes: ["./mediamtx.yml:/mediamtx.yml"]
    restart: unless-stopped

  inference:
    image: python:3.9-slim
    ports: ["5000:5000"]
    volumes:
      - ./inference:/app
      - ../models:/models
    working_dir: /app
    command: bash -c "pip install -r requirements.txt && python inference_server.py"
    restart: unless-stopped
```

### mediamtx.yml
```yaml
rtspAddress: :8554
rtmpAddress: :1935
hlsAddress: :8888
webrtcAddress: :8889

paths:
  cam1:
    source: publisher
    sourceProtocol: automatic
```

## Artwork Detection

The system uses a custom-trained YOLO model to detect 3 specific artworks:
- artwork1
- artwork2  
- artwork3

Detection Rate: 2 inferences per second (500ms interval)

## API Endpoints

### GET /
- Web interface showing real-time detection results

### GET /api/detections
- Returns JSON with current detections
```json
{
  "detections": [
    {
      "class": "artwork1",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 300],
      "timestamp": "2024-01-01T10:30:00"
    }
  ],
  "frame_count": 42,
  "timestamp": "2024-01-01T10:30:01"
}
```

### GET /api/health
- System status and model information

## Auto-start (Optional)

### On Raspberry Pi:
```bash
sudo nano /etc/systemd/system/camera-stream.service
```

Add:
```ini
[Unit]
Description=Artwork Detection Camera Stream
After=network.target

[Service]
Type=simple
User=pi
ExecStart=/home/pi/start_camera_stream.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable camera-stream.service
sudo systemctl start camera-stream.service
```

## Troubleshooting

### Common Issues:

1. Containers not starting
   ```cmd
   docker-compose down
   docker-compose up -d
   ```

2. Model not loading
   - Verify `best_detect.onnx` exists in `../models/`
   - Check Docker logs: `docker logs yolo-inference`

3. Stream connection failed
   - Verify laptop IP address in stream script
   - Check both devices on same WiFi network
   - Ensure Windows firewall allows ports 8554, 5000

4. Camera not detected
   ```bash
   # On Raspberry Pi
   v4l2-ctl --list-devices
   sudo raspi-config  # Enable camera interface
   ```

### Check Logs:
```cmd
# MediaMTX logs
docker logs mediamtx

# Inference server logs
docker logs yolo-inference -f
```

## Performance

- Stream Resolution: 1280x720
- Frame Rate: 30 FPS (stream), 2 FPS (detection)
- Latency: ~200-500ms
- Detection Confidence Threshold: 0.5

## Updating Artwork Classes

To modify detected artwork classes, edit `inference/inference_server.py`:
```python
self.classes = [
    'mona_lisa',
    'starry_night', 
    'the_scream'
]
```
