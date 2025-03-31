import cv2
import torch
import time

# Load the custom YOLOv5 model (modify the path to your custom model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gajjar/Downloads/best (1).pt')

# Set the model to evaluation mode
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(2)  # Change to 0 or 1 if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

start_time = time.time()
fps_list = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert to RGB for YOLOv5
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    curr_time = time.time()
    results = model(img_rgb)  # YOLOv5 inference
    inference_time = time.time() - curr_time
    
    # Calculate FPS
    fps = 1 / inference_time
    fps_list.append(fps)
    frame_count += 1
    print(f"Frame {frame_count}: {fps:.2f} FPS")
    
    # Stop after 5 seconds
    if time.time() - start_time >= 5:
        break

# Compute average FPS
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
print(f"Average FPS: {avg_fps:.2f}")

# Release resources
cap.release()
