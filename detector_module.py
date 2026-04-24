"""
Reusable module for social distancing detection
Can be used by both command-line script and web application
"""
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os


def get_model_path():
    """Get the absolute path to the YOLO model directory"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (parent of script_dir if script is in subdirectory)
    # If detector_module.py is in the root, use script_dir directly
    model_path = os.path.join(script_dir, config.MODEL_PATH)
    # If that doesn't exist, try parent directory (in case we're in a subdirectory)
    if not os.path.exists(model_path):
        parent_dir = os.path.dirname(script_dir)
        model_path = os.path.join(parent_dir, config.MODEL_PATH)
    return model_path


def initialize_detector():
    """Initialize and return the YOLO detector network"""
    # Get the correct model path
    model_path = get_model_path()
    
    # load the COCO class labels the YOLO model was trained on
    labelsPath = os.path.join(model_path, "coco.names")
    if not os.path.exists(labelsPath):
        raise FileNotFoundError(f"COCO names file not found at: {labelsPath}")
    LABELS = open(labelsPath).read().strip().split("\n")
    
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.join(model_path, "yolov3.weights")
    configPath = os.path.join(model_path, "yolov3.cfg")
    
    if not os.path.exists(weightsPath):
        raise FileNotFoundError(f"YOLO weights file not found at: {weightsPath}")
    if not os.path.exists(configPath):
        raise FileNotFoundError(f"YOLO config file not found at: {configPath}")
    
    # load the YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    # check if GPU is to be used or not
    if config.USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # determine only the "output" layer names that we need from YOLO
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, ln, LABELS


def process_video(input_path, output_path=None, user_id=None, display=False):
    """
    Process a video file for social distancing detection
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video (optional)
        user_id: User ID for organizing output files (optional)
        display: Whether to display frames during processing
    
    Returns:
        dict: Statistics about the detection (violations, people detected, etc.)
    """
    # Initialize detector
    net, ln, LABELS = initialize_detector()
    
    # Create user-specific directories only if saving images
    if getattr(config, "SAVE_PERSON_IMAGES", False):
        if user_id:
            user_persons_dir = os.path.join("detected_persons", f"user_{user_id}")
            user_faces_dir = os.path.join("detected_faces", f"user_{user_id}")
            os.makedirs(user_persons_dir, exist_ok=True)
            os.makedirs(user_faces_dir, exist_ok=True)
        else:
            user_persons_dir = "detected_persons"
            user_faces_dir = "detected_faces"
            os.makedirs(user_persons_dir, exist_ok=True)
            os.makedirs(user_faces_dir, exist_ok=True)
    
    # Load face detector only if needed
    face_cascade = None
    if getattr(config, "SAVE_PERSON_IMAGES", False):

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video stream
    vs = cv2.VideoCapture(input_path)
    writer = None
    
    try:
        # Get video properties for progress tracking
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vs.get(cv2.CAP_PROP_FPS)
        
        # Handle cases where video properties might not be available
        if total_frames <= 0:
            total_frames = 0  # Will be counted as we process
        if fps <= 0:
            fps = 25  # Default FPS
        
        frame_number = 0
        processed_frames = 0
        
        # Statistics
        stats = {
            'total_frames': 0,
            'total_violations': 0,  # Count of violation pairs (not individual people)
            'max_violations_in_frame': 0,
            'total_people_detected': 0,  # Will be average per frame
            'people_detected_sum': 0,  # Sum of people across all frames for calculating average
            'frames_with_violations': 0
        }
        
        person_count = 0
        face_count = 0
        
        print(f"\n{'='*60}")
        print(f"[INFO] Starting video processing...")
        print(f"[INFO] Video: {total_frames} frames at {fps:.2f} FPS")
        print(f"[INFO] Frame skip: {config.FRAME_SKIP} (processing every {config.FRAME_SKIP} frame(s))")
        print(f"[INFO] Estimated processing time: {total_frames / (fps * config.FRAME_SKIP) / 60:.1f} minutes")
        print(f"{'='*60}\n")
        
        # Process video frames
        while True:
            (grabbed, frame) = vs.read()
            
            if not grabbed:
                break
            
            frame_number += 1
            
            # Skip frames for faster processing
            if frame_number % config.FRAME_SKIP != 0:
                # Skip this frame (don't process, don't write to output)
                # This significantly speeds up processing
                continue
            
            processed_frames += 1
            stats['total_frames'] += 1
            
            # Print progress every 30 processed frames (or every 5% if video is long)
            if processed_frames % 30 == 0 or (total_frames > 0 and frame_number % max(1, total_frames // 20) == 0):
                progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                elapsed_frames = frame_number
                remaining_frames = total_frames - frame_number if total_frames > 0 else 0
                print(f"[INFO] Progress: {progress:.1f}% | Processed: {processed_frames} frames | Total: {frame_number}/{total_frames} frames")
            
            # Resize frame and detect people (smaller = faster)
            frame = imutils.resize(frame, width=config.FRAME_WIDTH)
            results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
            
            # Track people detected (sum for calculating average later)
            stats['people_detected_sum'] += len(results)
            
            # Initialize violation set and violation pairs counter
            violate = set()
            violation_pairs_this_frame = 0
            
            # Calculate distances if at least 2 people detected
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")
                
                # Check for violations - count pairs, not individuals
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        if D[i, j] < config.MIN_DISTANCE:
                            violate.add(i)
                            violate.add(j)
                            violation_pairs_this_frame += 1  # Count each pair as 1 violation
            
            # Update statistics
            if violation_pairs_this_frame > 0:
                stats['frames_with_violations'] += 1
                stats['total_violations'] += violation_pairs_this_frame
                if violation_pairs_this_frame > stats['max_violations_in_frame']:
                    stats['max_violations_in_frame'] = violation_pairs_this_frame
            
            # Draw bounding boxes and annotations
            for (i, (prob, bbox, centroid)) in enumerate(results):
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                
                # Color: red for violations, green for safe
                if i in violate:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
                
                text = "Person"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                # Save person images only if enabled (major speedup when disabled)
                if getattr(config, "SAVE_PERSON_IMAGES", False) and startX >= 0 and startY >= 0 and endX <= frame.shape[1] and endY <= frame.shape[0]:
                    person_roi = frame[startY:endY, startX:endX]
                    if person_roi.size > 0:
                        person_img_path = os.path.join(user_persons_dir, f"person_{person_count}.jpg")
                        person_count += 1
                        cv2.imwrite(person_img_path, person_roi)
                        
                        # Detect and save faces only if enabled
                        if getattr(config, "SAVE_FACE_IMAGES", False) and face_cascade is not None:
                            gray_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(
                                gray_person_roi, 
                                scaleFactor=1.1, 
                                minNeighbors=5, 
                                minSize=(30, 30)
                            )
                            
                            for (fx, fy, fw, fh) in faces:
                                fx_abs = startX + fx
                                fy_abs = startY + fy
                                face_img_path = os.path.join(user_faces_dir, f"face_{face_count}.jpg")
                                face_count += 1
                                cv2.imwrite(face_img_path, frame[fy_abs:fy_abs+fh, fx_abs:fx_abs+fw])
                                cv2.rectangle(frame, (fx_abs, fy_abs), (fx_abs + fw, fy_abs + fh), (255, 0, 0), 2)
            
            # Draw violation count on frame
            text = "Social Distancing Violations: {}".format(violation_pairs_this_frame)
            cv2.putText(frame, text, (10, frame.shape[0] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Display if requested
            if display:
                cv2.imshow("Output", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            
            # Write to output video if path provided
            if output_path and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                # Use original FPS or adjusted FPS based on frame skip
                output_fps = fps / config.FRAME_SKIP if fps > 0 else 25
                writer = cv2.VideoWriter(output_path, fourcc, int(output_fps), 
                                       (frame.shape[1], frame.shape[0]), True)
            
            if writer is not None:
                writer.write(frame)
        
        # Calculate average violations per frame and average people per frame
        if stats['total_frames'] > 0:
            stats['avg_violations_per_frame'] = stats['total_violations'] / stats['total_frames']
            stats['total_people_detected'] = stats['people_detected_sum'] / stats['total_frames']  # Average per frame
        else:
            stats['avg_violations_per_frame'] = 0
            stats['total_people_detected'] = 0
        
        # Remove temporary field
        if 'people_detected_sum' in stats:
            del stats['people_detected_sum']
        
        print(f"\n{'='*60}")
        print(f"[INFO] ✅ Processing complete!")
        print(f"[INFO] Processed {processed_frames} frames out of {frame_number} total frames")
        print(f"[INFO] Total violation pairs detected: {stats['total_violations']}")
        print(f"[INFO] Average people per frame: {stats['total_people_detected']:.2f}")
        print(f"[INFO] Frames with violations: {stats['frames_with_violations']}")
        print(f"{'='*60}\n")
        
    finally:
        # Always cleanup video resources, even if an error occurred
        if writer is not None:
            writer.release()
        if vs is not None:
            vs.release()
        if display:
            cv2.destroyAllWindows()
    
    return stats

