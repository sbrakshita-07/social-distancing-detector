import cv2
import os

# Paths
input_path = r"C:\Users\RAKSHITA\Desktop\social-distancing-detector\pedestrian.mp4"
output_path = r"C:\Users\RAKSHITA\Desktop\social-distancing-detector\pedestrian.avi"

# Open the video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("[ERROR] Cannot open video:", input_path)
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("[INFO] Processing video... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("Video Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Video saved as:", output_path)
