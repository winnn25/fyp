import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model for box detection
model_path = os.path.join('.', 'sign_train', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)

# Use the model to make predictions on an image
image_path = r"C:\Users\ashwi\OneDrive\Pictures\WhatsApp Image 2024-09-14 at 22.27.41_3aa618e7.jpg"
results = model(image_path)

# Desired output width and aspect ratio (9:16)
output_width = 450
aspect_ratio = 9 / 16
output_height = int(output_width / aspect_ratio)

# Iterate over the results and show each image with detections
for result in results:
    # result.plot() draws the bounding boxes on the image, but you need to show it using OpenCV
    img_with_boxes = result.plot()  # Plot the result (bounding boxes will be drawn on the image)
    
    # Resize the image to 360 width and maintain 9:16 aspect ratio
    resized_image = cv2.resize(img_with_boxes, (output_width, output_height))
    
    # Display the resized image using OpenCV
    cv2.imshow('Last.pt', resized_image)
    cv2.waitKey(0)  # Press any key to close the image window
    cv2.destroyAllWindows()

