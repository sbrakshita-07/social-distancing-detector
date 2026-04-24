# imports
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments 1qW
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if GPU is to be used or not
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# open input video if available else webcam stream
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# create directories to save detected persons and faces
if not os.path.exists("detected_persons"):
    os.makedirs("detected_persons")
if not os.path.exists("detected_faces"):
    os.makedirs("detected_faces")

# load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize counters for saving detected faces and persons
person_count = 0
face_count = 0

# loop over the frames from the video stream
while True:
    # read the next frame from the input video
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then that's the end fo the stream
    if not grabbed:
        break

    # resize the frame and then detect people (only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # initialize the set of indexes that violate the minimum social distance
    violate = set()

    # ensure there are at least two people detections (required in order to compute the
    # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        if i in violate:
            color = (0, 0, 255)  # Red color for violations
        else:
            color = (0, 255, 0)  # Green color for non-violations

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

        # add the label "Person" above the bounding box
        text = "Person"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # ensure the coordinates are within frame boundaries
        if startX >= 0 and startY >= 0 and endX <= frame.shape[1] and endY <= frame.shape[0]:
            person_roi = frame[startY:endY, startX:endX]
            if person_roi.size > 0:  # Ensure ROI is not empty
                person_img_path = os.path.join("detected_persons", f"person_{person_count}.jpg")
                person_count += 1
                cv2.imwrite(person_img_path, person_roi)

                # detect faces in the person bounding box
                gray_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_person_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (fx, fy, fw, fh) in faces:
                    fx_abs = startX + fx
                    fy_abs = startY + fy
                    face_img_path = os.path.join("detected_faces", f"face_{face_count}.jpg")
                    face_count += 1
                    cv2.imwrite(face_img_path, frame[fy_abs:fy_abs+fh, fx_abs:fx_abs+fw])
                    cv2.rectangle(frame, (fx_abs, fy_abs), (fx_abs + fw, fy_abs + fh), (255, 0, 0), 2)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video writer has not been
    # initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        print("[INFO] writing stream to output")
        writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
if writer is not None:
    writer.release()
vs.release()
cv2.destroyAllWindows()