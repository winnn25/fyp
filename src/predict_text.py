import os
import cv2
import easyocr
import numpy as np
import re
from collections import deque
from ultralytics import YOLO

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Define directories and video paths
video_path = r"C:\Users\ashwi\OneDrive\Desktop\Final\videos\VID_20240913_183358.mp4"
video_path_out = '{}_out.mp4'.format(video_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open the video file at {video_path}.")
    exit(1)

# Read the first frame
ret, frame = cap.read()
H, W, _ = frame.shape

# Define video writer for the output video
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load the YOLOv8 model for box detection
model_path = os.path.join('.', 'sign_train', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # Load the custom model

# Set minimum and maximum brightness thresholds
min_brightness = 70
max_brightness = 200

# Store detected text to maintain stability and prevent blinking
persistent_text = None
text_history = deque(maxlen=5)  # Keep track of the last 5 detections

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
        return apply_gamma_correction(frame, gamma=2.0)  # Brighten the frame
    elif avg_brightness > max_brightness:
        return apply_gamma_correction(frame, gamma=0.5)  # Darken the frame
    else:
        return frame  # No adjustment needed

# Function to perform OCR on a specific region of the frame
def run_ocr(frame, box=None):
    if box is not None:
        x1, y1, x2, y2 = box
        frame = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop to bounding box
    
    # Preprocessing: Adjust brightness dynamically
    adjusted_frame = adjust_brightness(frame)
    
    # Run OCR on the processed area
    result = reader.readtext(adjusted_frame, detail=0)
    text = " ".join(result)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)  # Clean the text
    return text

# Process every frame (no skipping)
while ret:
    # Convert frame to RGB for YOLOv8 input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get model predictions on the frame
    results = model(frame_rgb)[0]

    # Iterate through detected boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # If confidence score is above threshold, process the box and run OCR
        if score > 0.3:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            
            # Run OCR on the detected region (ROI)
            detected_text = run_ocr(frame, (x1, y1, x2, y2))

            # Only consider boxes where text is detected (not empty)
            if detected_text.strip():
                # Add detected text to history for stability
                text_history.append(detected_text)
                
                # Determine the most common text in the history
                most_common_text = max(set(text_history), key=text_history.count)

                # Ensure text is stable and does not blink or change if it's the same
                if persistent_text is None or persistent_text != most_common_text:
                    persistent_text = most_common_text  # Update persistent text
                
                # Display the stable text on the frame
                cv2.putText(frame, persistent_text, (int(x1), int(y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    # Write processed frame to output video
    out.write(frame)

    # Read next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
