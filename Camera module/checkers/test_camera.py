import cv2
cap = cv2.VideoCapture(1)
if cap.isOpened():
    print("Camera is accessible")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Camera not accessible")