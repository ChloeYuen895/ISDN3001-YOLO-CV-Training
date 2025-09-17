from ultralytics import YOLO
import torch

# Load the trained YOLO model
model = YOLO('C:/Users/yueny/OneDrive/Documents/ISDN3001/runs/segment/artwork_segmentation9/weights/best.pt')  # Path to your .pt file

# Export to ONNX
model.export(format='onnx', imgsz=[640, 640], dynamic=True, opset=11)

# Verify the ONNX model
import onnx
onnx_model = onnx.load('best.onnx')  # Adjust to the output path if different
onnx.checker.check_model(onnx_model)
print("ONNX model verified successfully!")