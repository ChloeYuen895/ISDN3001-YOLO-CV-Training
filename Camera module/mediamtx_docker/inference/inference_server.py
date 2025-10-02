import cv2
import onnxruntime as ort
import numpy as np
import time
import threading
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

class ArtworkDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])
        
        self.classes = [
            'The Progress of a Soul: The Victory, Phoebe Anna Traquair 1902',
            'The Execution of Lady Jane Grey, Paul Delaroche 1833',
            'Café Terrace at Night, Vincent van Gogh 1888'
        ]
        
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4

    def preprocess(self, image):
        img = cv2.resize(image, self.input_size)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, outputs, original_shape):
        predictions = outputs[0][0]
        detections = []
        
        for pred in predictions:
            confidence = pred[4]
            if confidence < self.confidence_threshold:
                continue
            
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            total_confidence = confidence * class_confidence
            
            if total_confidence > self.confidence_threshold:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                
                x1 = int((cx - w/2) * original_shape[1])
                y1 = int((cy - h/2) * original_shape[0])
                x2 = int((cx + w/2) * original_shape[1])
                y2 = int((cy + h/2) * original_shape[0])
                
                # Ensure bounding box is within image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_shape[1], x2)
                y2 = min(original_shape[0], y2)
                
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
artwork_model = None
latest_results = []
frame_count = 0
last_inference_time = 0
INFERENCE_INTERVAL = 0.5  # 2 inferences per second

def initialize_model():
    global artwork_model
    try:
        # Use your custom artwork detection model
        model_path = "C:/Users/yueny/OneDrive/Documents/ISDN3001/runs/detect/artwork_recognition/weights/best.onnx"
        artwork_model = ArtworkDetector(model_path)
        print("Artwork detection model loaded successfully")
        print(f"Model classes: {artwork_model.classes}")
    except Exception as e:
        print(f"Error loading model: {e}")

def process_frames():
    global latest_results, frame_count, last_inference_time
    
    # RTSP stream URL from MediaMTX
    stream_url = "rtsp://localhost:8554/cam1"
    
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Cannot connect to RTSP stream")
        print("Waiting for stream to become available...")
        time.sleep(5)
        return
    
    print("Connected to RTSP stream, starting artwork detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from stream")
            time.sleep(1)
            continue
        
        current_time = time.time()
        
        # Process 2 frames per second (every 0.5 seconds)
        if current_time - last_inference_time >= INFERENCE_INTERVAL:
            if artwork_model:
                try:
                    detections = artwork_model.detect(frame)
                    latest_results = detections
                    
                    # Print results to console
                    if detections:
                        print("Artwork Detections:", [(d['class'], "{:.2f}".format(d['confidence'])) for d in detections])
                    else:
                        print("No artworks detected")
                        
                except Exception as e:
                    print(f"Detection error: {e}")
            
            last_inference_time = current_time
            frame_count += 1
    
    cap.release()

# Flask routes
@app.route('/api/detections', methods=['GET'])
def get_detections():
    return jsonify({
        'detections': latest_results,
        'frame_count': frame_count,
        'timestamp': datetime.now().isoformat(),
        'model_loaded': artwork_model is not None
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': artwork_model is not None,
        'model_classes': artwork_model.classes if artwork_model else []
    })

@app.route('/')
def index():
    detection_summary = ""
    if latest_results:
        detection_summary = "<ul>"
        for detection in latest_results:
            detection_summary += f"<li>{detection['class']} - {detection['confidence']:.2f}</li>"
        detection_summary += "</ul>"
    else:
        detection_summary = "<p>No artworks detected</p>"
    
    return f"""
    <html>
        <head>
            <title>Artwork Detection Server</title>
            <meta http-equiv="refresh" content="2">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .detection {{ background: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Artwork Detection Server</h1>
            <p><strong>Model Status:</strong> {"Loaded" if artwork_model else "Not Loaded"}</p>
            <p><strong>Frame Count:</strong> {frame_count}</p>
            <p><strong>Artworks Detected:</strong> {len(latest_results)}</p>
            <div class="detections">
                <h3>Latest Detections:</h3>
                {detection_summary}
            </div>
            <p><a href="/api/detections">View JSON API</a> | <a href="/api/health">System Health</a></p>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Start frame processing in background thread
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    # Start Flask server
    print("Starting artwork detection server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)