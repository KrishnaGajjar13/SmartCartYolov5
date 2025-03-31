import cv2

# Function to check the available webcam indexes
def test_webcam_indexes():
    for index in range(1, 11):
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            print(f"Webcam found at index {index}")
            cap.release()  # Release the webcam after checking
        else:
            print(f"No webcam found at index {index}")

# Call the function to test indices
test_webcam_indexes()
