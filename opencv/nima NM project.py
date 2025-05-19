from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Replace with 'best.pt' if you have a custom-trained model

import cv2
from ultralytics import YOLO

# Load YOLO model (replace 'yolov8n.pt' with your trained model e.g., 'best.pt')
model = YOLO("yolov8n.pt")

# Start webcam or video file
cap = cv2.VideoCapture(0)  # 0 for webcam. You can also use "video.mp4"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Real-Time Traffic Sign Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
