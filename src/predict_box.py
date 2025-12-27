import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
from ultralytics import YOLO
import cv2

# Define directories and video paths
VIDEOS_DIR = os.path.join('.', 'videos')
video_path = r"C:\Users\ashwi\OneDrive\Desktop\Final\videos\VID_20240913_183358.mp4"
video_path_out = '{}_out.mp4'.format(video_path)

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: The video file at {video_path} does not exist.")
    exit(1)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open the video file at {video_path}.")
    exit(1)

# Read the first frame
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Failed to read the first frame from the video.")
    exit(1)

H, W, _ = frame.shape

# Define video writer for the output video
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load the YOLOv8 model
model_path = os.path.join('.', 'sign_train', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # Load the custom model

threshold = 0.3  # Set a lower detection threshold

while ret:
    # Convert frame to RGB for YOLOv8 input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get model predictions on the frame
    results = model(frame_rgb)[0]

    # Iterate through detected boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # If confidence score is above threshold, draw the bounding box
        if score > threshold:
            # Draw rectangle and put label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write processed frame to output video
    out.write(frame)

    # Read next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
