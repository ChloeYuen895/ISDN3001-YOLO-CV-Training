from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

def plot_training_results(results):
    """Plot training and validation loss/accuracy curves."""
    train_loss = results.metrics['train/loss']
    val_loss = results.metrics['val/loss']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

if __name__ == '__main__':
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained YOLOv8 nano segmentation model
    model = YOLO('yolov8n-seg.pt')

    # Train on your dataset
    results = model.train(
        data='C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/dataset/data.yaml',
        epochs=20,          # Reduced to 20 to avoid overfitting
        imgsz=640,          # Standard image size for YOLOv8
        batch=4,            # Reduced for Raspberry Pi compatibility
        device=device,
        name='artwork_segmentation',
        task='segment',
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

    # Plot training results
    plot_training_results(results)
    print("Training loss plot saved as 'loss_plot.png'")

    # Evaluate on validation set
    metrics = model.val(task='segment')
    print("Validation Metrics:")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50:95: {metrics.box.map:.4f}")
    print(f"Segmentation mAP@50: {metrics.seg.map50:.4f}")
    print(f"Segmentation mAP@50:95: {metrics.seg.map:.4f}")

    # Export to ONNX for Raspberry Pi (optimized for inference)
    model.export(
        format='onnx',
        dynamic=True,       # Dynamic input shapes for flexibility
        simplify=True,      # Simplify ONNX model for faster inference
        opset=12           # Use ONNX opset 12 for Raspberry Pi compatibility
    )
    print("Model exported to ONNX format for Raspberry Pi deployment")