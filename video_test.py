import cv2
import os

# Path to your video
video_path = "social.mp4"  # replace with full path if needed
if not os.path.exists(video_path):
    print(f"[ERROR] Video not found: {video_path}")
    exit()

# Open the video
vs = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not vs.isOpened():
    print("[ERROR] Could not open video")
    exit()
else:
    print("[INFO] Video opened successfully")

# Loop through frames
while True:
    grabbed, frame = vs.read()
    if not grabbed:
        print("[INFO] End of video stream")
        break

    # Resize for display
    frame = cv2.resize(frame, (700, int(frame.shape[0] * 700 / frame.shape[1])))

    cv2.imshow("Video Test", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        print("[INFO] Exiting video stream")
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()
