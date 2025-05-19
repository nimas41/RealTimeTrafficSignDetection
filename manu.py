import cv2
import numpy as np

def classify_shape(contour, approx_area, bounding_box):
    aspect_ratio = bounding_box[2] / bounding_box[3]
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return "Unknown"

    solidity = approx_area / hull_area

    # Heuristics for basic signs
    if solidity > 0.9:
        return "Fist"  # compact shape
    elif 0.5 < solidity < 0.85 and aspect_ratio < 1.2:
        return "Open Palm"
    elif aspect_ratio > 1.5 and solidity < 0.6:
        return "Call Me"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Basic skin color range (adjust for glove or skin tone)
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sign = "No hand"
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 2000:
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sign = classify_shape(max_contour, area, (x, y, w, h))

    cv2.putText(frame, f"Sign: {sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    cv2.imshow("Sign Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()