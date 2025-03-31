import cv2
import torch
import time

# Load the custom YOLOv5 model (modify the path to your custom model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gajjar/Downloads/best (1).pt')

# Set the model to evaluation mode
model.eval()

# Initialize webcam (0 is the default webcam)
cap = cv2.VideoCapture(2)

# Set the desired resolution (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for FPS calculation
prev_time = time.time()  # Track the time at the start

# NMS parameters
confidence_threshold = 0.6 # Confidence threshold for filtering weak detections
iou_threshold = 1.0  # IoU threshold for NMS (you can adjust this value)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the image from BGR (OpenCV format) to RGB (YOLOv5 format)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform inference
    results = model(img_rgb)  # Perform inference on the frame

    # Get predictions for the current frame
    preds = results.pred[0]

    # Filter out low-confidence predictions
    preds = preds[preds[:, 4] > confidence_threshold]  # Confidence is at index 4

    # Apply NMS on the predictions
    results.pred[0] = results.pred[0][torch.ops.torchvision.nms(
        preds[:, :4],  # Bounding boxes
        preds[:, 4],   # Confidence scores
        iou_threshold  # IoU threshold
    )]

    # Frame data (class, confidence, bbox)
    frame_data = []

    # Process remaining predictions after NMS
    if len(results.pred[0]) > 0:  # Check if there are predictions
        for pred in results.pred[0]:
            # Extract class, confidence, and bounding box coordinates
            x1, y1, x2, y2, conf, cls = pred.tolist()

            # Store the data in the frame_data list
            frame_data.append([cls, conf, (x1, y1, x2, y2)])

        # Optionally, print the data for each frame
        #print(frame_data)
    else:
        # No predictions, skip printing
        pass

    # Render results on the frame (bounding boxes and labels)
    results.render()

    # Convert back to BGR for displaying with OpenCV
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame (top-left corner)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, f"FPS: {fps:.2f}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Live Webcam Feed', img_bgr)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
