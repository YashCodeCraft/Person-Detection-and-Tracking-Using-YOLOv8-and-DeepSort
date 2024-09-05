import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSort tracker for multi-object tracking
model = YOLO('yolov8n.pt') 

# Initialize DeepSort tracker
tracker = DeepSort(
    max_age=300,  
    n_init=3, 
    max_iou_distance=0.6, 
    nn_budget=300
)

def process_video(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame)
        
         # Extract bounding boxes and prediction data
        boxes = results[0].boxes
        detections = boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = boxes.cls.cpu().numpy()  # Class IDs

        # Prepare detections for DeepSort
        detections_for_tracker = []
        for box, score, cls in zip(detections, confidences, class_ids):
            if score > 0.7 and cls == 0:  # Filter out low-confidence detections and non-person classes
                detections_for_tracker.append((box, score, cls))
        
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        
        # Display detections and tracking IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            label = f'ID {track_id}' # Label with unique ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Replace with your actual input and output video paths
video_path = r"Input_video.mp4"
output_path =  r"Output_video.mp4"

# video processing function
process_video(video_path, output_path)
