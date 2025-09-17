from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained YOLOv8 nano segmentation model
    model = YOLO('yolov8n-seg.pt')

    # Train on your dataset
    results = model.train(
        data='C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/dataset/data.yaml',
        epochs=25,          # Reasonable for small/medium datasets
        imgsz=640,          # Balance accuracy and speed
        batch=8,            # Adjust to 4 or 2 if memory-limited
        device=device,
        name='artwork_segmentation',
        task='segment',
        patience=10,        # Early stopping after 10 epochs of no improvement
        # Necessary and valid augmentations
        fliplr=0.5,         # 50% chance of horizontal flip
        flipud=0.5,         # 50% chance of vertical flip
        degrees=10.0,       # ±10° rotation
        hsv_v=0.15,         # ±15% brightness (also approximates contrast/noise)
        perspective=0.001   # Slight perspective distortion
    )

    # Evaluate on validation set
    metrics = model.val(task='segment')
    print(metrics)

    # Export to ONNX for Raspberry Pi
    model.export(format='onnx', dynamic=True)