import cv2
import numpy as np
import threading
import time
import queue
import subprocess
from datetime import datetime

class USBCameraInferencePipeline:
    def __init__(self):
        self.rtsp_url = "rtsp://localhost:8554/mystream"
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.labels = []
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_inference(self):
        self.is_running = True
        
        capture_thread = threading.Thread(target=self.optimized_capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        inference_thread = threading.Thread(target=self.optimized_inference)
        inference_thread.daemon = True
        inference_thread.start()
        
        print("USB Camera Inference Pipeline Started")
        print("Waiting for stream from Raspberry Pi...")
    
    def optimized_capture_frames(self):
        """Capture with USB camera compatibility"""
        # Try different OpenCV backends for better compatibility
        backends = [
            cv2.CAP_FFMPEG,
            cv2.CAP_ANY
        ]
        
        cap = None
        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.rtsp_url, backend)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test if capture works
                ret, test_frame = cap.read()
                if ret:
                    print(f"Successfully opened stream with backend: {backend}")
                    break
                else:
                    cap.release()
                    cap = None
            except:
                cap = None
        
        if cap is None:
            print("Failed to open stream with any backend")
            return
        
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while self.is_running:
            ret, frame = cap.read()
            if ret:
                reconnect_attempts = 0  # Reset on successful frame
                
                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Drop frame to maintain low latency
            else:
                print(f"Failed to capture frame - attempt {reconnect_attempts + 1}/{max_reconnect_attempts}")
                reconnect_attempts += 1
                
                if reconnect_attempts >= max_reconnect_attempts:
                    print("Max reconnection attempts reached. Restarting capture...")
                    break
                
                time.sleep(1)
        
        cap.release()
        
        # Auto-restart capture if failed
        if self.is_running:
            print("Restarting capture thread...")
            time.sleep(2)
            self.optimized_capture_frames()
    
    def optimized_inference(self):
        """Optimized inference with USB camera considerations"""
        no_frame_count = 0
        
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=2.0)
                no_frame_count = 0  # Reset counter
                
                self.frame_count += 1
                
                # Calculate FPS every second
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = current_time
                
                # Perform inference
                labels = self.usb_camera_detection(frame)
                self.labels = labels
                
                latency = time.time() - self.last_frame_time
                self.last_frame_time = time.time()
                
                # Display frame info
                height, width = frame.shape[:2]
                print(f"FPS: {self.fps:5.1f} | Latency: {latency:.3f}s | Size: {width}x{height} | Detected: {labels}")
                
            except queue.Empty:
                no_frame_count += 1
                if no_frame_count % 10 == 0:  # Print every 10 empty checks
                    print("No frames received - waiting for Raspberry Pi stream...")
            except Exception as e:
                print(f"Inference error: {e}")
    
    def usb_camera_detection(self, frame):
        """Enhanced detection for USB camera characteristics"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Get frame properties
        height, width = frame.shape[:2]
        avg_brightness = np.mean(frame)
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Simple "detection" based on color characteristics
        saturation = np.mean(hsv[:,:,1])
        lightness = np.mean(lab[:,:,0])
        
        labels = [f"usb_cam_{timestamp}"]
        
        # Add frame characteristics
        if avg_brightness > 150:
            labels.append("bright")
        elif avg_brightness < 50:
            labels.append("dark")
        
        if saturation > 100:
            labels.append("high_sat")
        
        labels.append(f"size_{width}x{height}")
        
        return labels
    
    def get_current_labels(self):
        return self.labels.copy()
    
    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    pipeline = USBCameraInferencePipeline()
    try:
        pipeline.start_inference()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping USB camera inference pipeline...")
        pipeline.stop()