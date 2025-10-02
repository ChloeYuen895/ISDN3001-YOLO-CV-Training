import cv2
import onnxruntime as ort
import numpy as np
import time
import threading
from flask import Flask, jsonify, Response
import json
from datetime import datetime

app = Flask(__name__)

class YOLOONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])  # (640, 640)
        
        # COCO class names
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4

    def preprocess(self, image):
        # Resize image to model input size
        img = cv2.resize(image, self.input_size)
        img = img / 255.0  # Normalize
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, outputs, original_shape):
        predictions = outputs[0][0]  # First batch
        detections = []
        
        for pred in predictions:
            # Filter by confidence
            confidence = pred[4]
            if confidence < self.confidence_threshold:
                continue
            
            # Get class with highest score
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            # Combined confidence
            total_confidence = confidence * class_confidence
            
            if total_confidence > self.confidence_threshold:
                # Bounding box coordinates (center x, center y, width, height)
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                
                # Convert to corner coordinates
                x1 = int((cx - w/2) * original_shape[1])
                y1 = int((cy - h/2) * original_shape[0])
                x2 = int((cx + w/2) * original_shape[1])
                y2 = int((cy + h/2) * original_shape[0])
                
                detections.append({
                    'class': self.classes[class_id],
                    'class_id': int(class_id),
                    'confidence': float(total_confidence),
                    'bbox': [x1, y1, x2, y2],
                    'timestamp': datetime.now().isoformat()
                })
        
        return detections

    def detect(self, image):
        original_shape = image.shape[:2]
        input_tensor = self.preprocess(image)
        
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, original_shape)
        
        return detections

# Global variables
yolo_model = None
latest_results = []
frame_count = 0
last_inference_time = 0
INFERENCE_INTERVAL = 0.5  # 2 inferences per second

def initialize_model():
    global yolo_model
    try:
        yolo_model = YOLOONNX('C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/models/best_detect.onnx')
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

def process_frames():
    global latest_results, frame_count, last_inference_time
    
    # RTSP stream URL from MediaMTX
    stream_url = "rtsp://localhost:8554/cam1"
    
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Cannot connect to RTSP stream")
        return
    
    print("Connected to RTSP stream, starting inference...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            time.sleep(1)
            continue
        
        current_time = time.time()
        
        # Process 2 frames per second (every 0.5 seconds)
        if current_time - last_inference_time >= INFERENCE_INTERVAL:
            if yolo_model:
                try:
                    detections = yolo_model.detect(frame)
                    latest_results = detections
                    
                    # Print results to console
                    if detections:
                        print(f"Detections: {[(d['class'], d['confidence']) for d in detections]}")
                    else:
                        print("No detections")
                        
                except Exception as e:
                    print(f"Inference error: {e}")
            
            last_inference_time = current_time
            frame_count += 1
    
    cap.release()

# Flask routes
@app.route('/api/detections', methods=['GET'])
def get_detections():
    return jsonify({
        'detections': latest_results,
        'frame_count': frame_count,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': yolo_model is not None})

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>YOLO Inference Server</title>
            <meta http-equiv="refresh" content="2">
        </head>
        <body>
            <h1>YOLO Inference Server</h1>
            <p>Model: {}</p>
            <p>Frame Count: {}</p>
            <p>Latest Detections: {}</p>
            <p><a href="/api/detections">View JSON API</a></p>
        </body>
    </html>
    """.format(
        "Loaded" if yolo_model else "Not Loaded",
        frame_count,
        len(latest_results)
    )

if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Start frame processing in background thread
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    # Start Flask server
    print("Starting inference server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)