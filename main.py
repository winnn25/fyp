import os
import cv2
import easyocr
import numpy as np
import re
from collections import deque
from ultralytics import YOLO
import pyttsx3

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Initialize the TTS engine
engine = pyttsx3.init()

# Set speech rate and volume for the TTS engine
engine.setProperty('rate', 125)  # Slow down speech rate
engine.setProperty('volume', 1)  # Set volume level (1 is the max)

# Function to speak the detected text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Test with webcam (use 0 for default camera, change to 1 if needed)
cap = cv2.VideoCapture(1)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit(1)

# Get the video's FPS and calculate delay
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)

# Get frame dimensions
ret, frame = cap.read()
H, W, _ = frame.shape

# Load the YOLOv8 model for box detection
model_path = os.path.join('.', 'sign_train', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # Load the custom model

# Set minimum and maximum brightness thresholds
min_brightness = 70
max_brightness = 200

# Store detected text to maintain stability and prevent blinking
persistent_text = None
text_history = deque(maxlen=5)  # Keep track of the last 5 detections
detected_text_stable = None
previous_bounding_box = None 
text_stability_counter = 0  # Counter to track stable text
stability_threshold = 3  # Threshold for text to be considered stable (e.g., 3 consecutive frames)

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

# Frame counter and interval to reduce processing load
frame_counter = 0
frame_interval = 5

# Process every frame from the video in real-time
while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image from video.")
        break

    # Rotate the frame 90 degrees clockwise to make it vertical
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # we're focusing on the center region and we crop 10% from each side
    width = frame.shape[1]
    crop_width = int(width * 0.1)  # 10% from each side
    frame_cropped = frame[:, crop_width:width - crop_width]

    frame_counter += 1

    # Skip frames to reduce processing load
    if frame_counter % frame_interval != 0:
        continue

    # Convert frame to RGB for YOLOv8 input
    frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)

    # Get model predictions on the frame
    results = model(frame_rgb)[0]

    current_bounding_box = None  # Track current frame's bounding box

    # Iterate through detected boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # If confidence score is above threshold, process the box and run OCR
        if score > 0.3:
            current_bounding_box = (x1, y1, x2, y2)

            # Draw bounding box
            cv2.rectangle(frame_cropped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            # Run OCR and detect text
            detected_text = run_ocr(frame_cropped, (x1, y1, x2, y2))

            # Only consider boxes where text is detected (not empty)
            if detected_text.strip():
                # Add detected text to history for stability
                text_history.append(detected_text)

                # Determine the most common text in the history
                most_common_text = max(set(text_history), key=text_history.count)

                # Check if the text is stable (detected consistently for several frames)
                if persistent_text == most_common_text:
                    text_stability_counter += 1
                else:
                    text_stability_counter = 0  # Reset counter if the text changes

                persistent_text = most_common_text  # Update persistent text

                # If the text has been stable for more than the threshold, store it
                if text_stability_counter >= stability_threshold:
                    detected_text_stable = persistent_text  # Update stable text

                    # Speak the detected stable text
                    speak_text(detected_text_stable)
            
            # Display the stable text on the frame only if it is confirmed stable
            if detected_text_stable:
                cv2.putText(frame_cropped, detected_text_stable, 
                            (int(x1), int(y1 - 10)),  # Position above the bounding box
                            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                            0.8,  # Smaller font scale for reduced text size
                            (255, 0, 0),
                            2,  
                            cv2.LINE_AA)  

    # Detect object exit: If there was a bounding box in the previous frame, but not in this frame
    if previous_bounding_box and current_bounding_box is None:
        # Reset the detected text and history when the object exits
        detected_text_stable = None
        text_history.clear()

    # Update previous bounding box
    previous_bounding_box = current_bounding_box

    # Display the cropped frame with detected text and bounding boxes
    cv2.imshow('Video OCR Detection', frame_cropped)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
