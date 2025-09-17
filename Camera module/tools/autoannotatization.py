from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the trained YOLOv8 segmentation model
model = YOLO("C:/Users/yueny/OneDrive/Documents/ISDN3001/runs/segment/artwork_segmentation3/weights/best.onnx", task="segment")

# Define class names based on your dataset
class_names = [
    'The Progress of a Soul: The Victory, Phoebe Anna Traquair 1902',
    'The Execution of Lady Jane Grey, Paul Delaroche 1833',
    'Café Terrace at Night, Vincent van Gogh 1888'
]

# Function to save segmentation annotations in YOLO format with confidence filter
def save_annotations(image_path, results, output_dir, conf_min=0.75, conf_max=1.0):
    image_name = os.path.basename(image_path)
    txt_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    h, w, _ = img.shape

    with open(txt_path, "w") as f:
        for result in results:
            if result.masks is None:
                print(f"No segmentation masks found for {image_name}")
                continue
            masks = result.masks.xy  # Polygon coordinates (x, y pairs) for each mask
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            for mask, score, cls in zip(masks, scores, classes):
                # Filter by confidence score (0.75 to 1.0)
                if not (conf_min <= score <= conf_max):
                    continue
                # Normalize polygon coordinates to 0-1 range
                normalized_coords = []
                for x, y in mask:
                    x_norm = x / w
                    y_norm = y / h
                    normalized_coords.extend([x_norm, y_norm])
                # Write to file: class_id x1 y1 x2 y2 ... xn yn
                class_id = int(cls)
                coords_str = " ".join([f"{coord:.6f}" for coord in normalized_coords])
                f.write(f"{class_id} {coords_str}\n")
                print(f"Saved segmentation for {class_names[class_id]} in {image_name} (confidence: {score:.2f})")

# Autoannotate images in a folder
def autoannotate_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    supported_extensions = (".jpg", ".png", ".jpeg")
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(supported_extensions):
            img_path = os.path.join(input_dir, img_file)
            try:
                # Run inference with segmentation and confidence threshold
                results = model(img_path, task="segment", conf=0.75)
                # Save annotations (polygons) with confidence filter
                save_annotations(img_path, results, output_dir, conf_min=0.75, conf_max=1.0)
                # Save visualized image with segmentation masks
                annotated_img = results[0].plot(labels=True, conf=True)  # Draw masks, boxes, and labels
                output_img_path = os.path.join(output_dir, f"annotated_{img_file}")
                cv2.imwrite(output_img_path, annotated_img)
                print(f"Saved annotated image: {output_img_path}")
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")

# Example usage
input_dir = "C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/dataset/images"  # Test set directory
output_dir = "C:/Users/yueny/OneDrive/Documents/ISDN3001/Camera module/dataset/labels"  # Output directory
autoannotate_images(input_dir, output_dir)