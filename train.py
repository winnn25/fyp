from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

if __name__ == '__main__':
    # Ensure all the code that uses multiprocessing is inside this block
    results = model.train(data="config.yaml", epochs=50)  

