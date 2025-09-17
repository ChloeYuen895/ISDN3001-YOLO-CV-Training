import cv2
from ultralytics import YOLO

# Load the ONNX model
model = YOLO("C:/Users/yueny/OneDrive/Documents/ISDN3001/runs/detect/artwork_recognition/weights/best.onnx")

# Open the camera (0 for default camera, 1 for secondary, etc.)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set image size and confidence threshold
imgsz = 640
conf = 0.75

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform segmentation on CPU
    results = model.predict(source=frame, imgsz=imgsz, conf=conf, device="cpu")

    # Process results
    annotated_frame = results[0].plot()  # Plot segmentation masks and labels

    # Display the frame with segmentation
    cv2.imshow("Artwork Recognition", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()