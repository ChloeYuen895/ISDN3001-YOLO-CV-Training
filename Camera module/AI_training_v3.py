from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained YOLOv8 nano detection model
    model = YOLO('yolov8n.pt')

    # Train on your dataset
    results = model.train(
        data='C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/dataset/data.yaml',
        epochs=20,          # Reduced to 20 to avoid overfitting
        imgsz=640,          # Standard image size for YOLOv8
        batch=4,            # Reduced for Raspberry Pi compatibility
        device=device,
        name='artwork_regonition',
        patience=5,         # Early stopping after 5 epochs of no improvement
        # Enhanced augmentations to improve generalization
        fliplr=0.5,         # 50% chance of horizontal flip
        flipud=0.5,         # 50% chance of vertical flip
        degrees=15.0,       # ±15° rotation for more variety
        hsv_h=0.05,         # ±5% hue shift
        hsv_s=0.7,          # ±70% saturation shift
        hsv_v=0.4,          # ±40% brightness shift
        perspective=0.01,  # Slight perspective distortion
        translate=0.1,      # ±10% translation
        scale=0.5,          # ±50% zoom
        shear=2.0,          # ±2° shear
        mosaic=0.5,         # 50% chance of mosaic augmentation
        # Regularization
        dropout=0.3,        # Dropout for regularization
        weight_decay=0.01   # L2 regularization
    )

    # Evaluate on validation set
    metrics = model.val(task='detect')
    print(metrics)

    # Export to ONNX for Raspberry Pi
    model.export(format='onnx', dynamic=True)