import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/models/best_detect.onnx')  # Replace with 'path/to/custom.pt' for trained model

# RTSP stream from MediaMTX
cap = cv2.VideoCapture('rtsp://localhost:8554/pi_stream')

if not cap.isOpened():
    print("Error opening stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended")
        break

    # Run YOLO inference
    results = model(frame, conf=0.75)  # Adjust confidence for artwork detection

    # Draw detections
    annotated_frame = results[0].plot()

    # Display (for testing; in production, process detections)
    cv2.imshow('Artwork Recognition', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()