import warnings
import cv2
import easyocr
import numpy as np
import re
from collections import deque

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning, module='easyocr')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Open webcam
# cap = cv2.VideoCapture(0)
# Open video file (testing with video path)
video_path = r"C:\Users\ashwi\OneDrive\Desktop\Finalcopy\Final\videos\VID_20240913_183358.mp4"
cap = cv2.VideoCapture(video_path)

# Get the FPS of the video for real-time playback
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # Delay in milliseconds per frame for real-time speed

# Set minimum and maximum brightness thresholds
min_brightness = 70 
max_brightness = 200 

# Store detected text to maintain stability and prevent blinking
persistent_text = None

# Function to perform Gamma Correction
def apply_gamma_correction(frame, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

# Function to detect whether the frame is too bright or too dark and adjust accordingly
def adjust_brightness(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    
    if avg_brightness < min_brightness:
        # Too dark, apply gamma correction to brighten
        return apply_gamma_correction(frame, gamma=2.0)
    elif avg_brightness > max_brightness:
        # Too bright, reduce brightness
        return apply_gamma_correction(frame, gamma=0.5) 
    else:
        # Acceptable brightness, no adjustment needed
        return frame

# Function to perform OCR on a specific region of the frame
def run_ocr(frame, box=None):
    if box is not None:
        x, y, w, h = box
        # Ensure the box is within the frame dimensions
        x, y = max(0, x), max(0, y)
        w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)
        frame = frame[y:y+h, x:x+w]  # Crop to bounding box
    
    # Preprocessing: Adjust brightness dynamically
    adjusted_frame = adjust_brightness(frame)
    
    # Run OCR on the processed area
    result = reader.readtext(adjusted_frame, detail=0)
    text = " ".join(result)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)  # Clean the text
    return text

# Function to detect black horizontal boxes
def detect_black_horizontal_boxes(frame):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Adjust the HSV range for darker black shades to ignore shadows
    lower_black = np.array([0, 0, 0])  # Pure black with low saturation and brightness
    upper_black = np.array([180, 255, 30])  # Very dark shades of black

    # Create mask to isolate black regions
    black_mask = cv2.inRange(hsv_frame, lower_black, upper_black)
    
    # Find contours of black regions
    contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area
    horizontal_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        area = w * h
        if aspect_ratio > 2.5 and 1000 < area < 50000:  # Horizontal box, valid size
            horizontal_boxes.append((x, y, w, h))
    
    return horizontal_boxes

# Initialize deque to keep track of the last detected texts
text_history = deque(maxlen=5)  # Keep track of the last 5 detections

frame_counter = 0
frame_interval = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect black horizontal boxes
    boxes = detect_black_horizontal_boxes(frame)
    
    if boxes:
        for box in boxes:
            x, y, w, h = box
            # Run OCR on the detected box
            detected_text = run_ocr(frame, box)
            
            # Only consider boxes where text is detected (not empty)
            if detected_text.strip():
                # Add detected text to history
                text_history.append(detected_text)
                
                # Determine the most common text in the history
                most_common_text = max(set(text_history), key=text_history.count)
                
                # Ensure text is stable and does not blink or change if it's the same
                if persistent_text is None or persistent_text != most_common_text:
                    persistent_text = most_common_text  # Update persistent text
                
                # Draw bounding box around the detected black box and display the text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, persistent_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        # If no boxes are detected, reset the persistent text
        persistent_text = None

    # Display the frame with detected text and bounding box
    cv2.imshow('Real-Time Video', frame)

    # Delay the frame display to match the video's FPS
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
