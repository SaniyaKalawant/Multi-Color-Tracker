import cv2
import numpy as np

# Start video capture
cap = cv2.VideoCapture(0)

# Define HSV color ranges
color_ranges = {
    "red": ([0, 120, 70], [10, 255, 255]),
    "green": ([40, 40, 40], [70, 255, 255]),
    "blue": ([110, 50, 50], [130, 255, 255]),
    "purple": ([130, 50, 50], [170, 255, 255]),
    "black": ([0, 0, 0], [180, 255, 30]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "pink": ([140, 50, 50], [170, 255, 255]),
    "yellow": ([20, 100, 100], [30, 255, 255]),
    "cyan": ([85, 100, 100], [100, 255, 255]),
    "orange": ([10, 100, 100], [25, 255, 255])
}

# BGR color mapping for drawing rectangles
color_bgr = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "purple": (128, 0, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "pink": (255, 0, 215),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "orange": (0, 165, 255)
}

cv2.namedWindow("Object Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Object Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Read each frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Process each color
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)

        # Create mask
        mask = cv2.inRange(hsv, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes and display color name
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Ignore small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2  # Center of the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 3)
                cv2.putText(frame, color_name, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr[color_name], 2)

    # Show results
    cv2.imshow("Object Tracking", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
