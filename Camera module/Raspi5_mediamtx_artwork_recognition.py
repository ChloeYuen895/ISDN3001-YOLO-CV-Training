import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("C:/Users/yueny/OneDrive/Documents/ISDN3001/runs/detect/artwork_recognition/weights/best.onnx")
rtsp_url = 'rtsp://10.108.4.156:8554/webcam'  # Pi IP

cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    results = model(frame, conf=0.5, iou=0.7, imgsz=640)
    annotated = results[0].plot()
    cv2.imshow('YOLO Artwork Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()