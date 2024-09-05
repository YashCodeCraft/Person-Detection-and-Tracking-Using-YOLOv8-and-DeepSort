# Person-Detection-and-Tracking-Using-YOLOv8-and-DeepSort
Person Detection and Tracking System for Analyzing Children with Autism Spectrum Disorder and Therapists. This project leverages YOLOv8 for detection and DeepSort for tracking, providing unique IDs for individuals in videos.

## Project Overview
This project focuses on detecting and tracking people (specifically children and therapists) in a video. Using YOLOv8 for detection and DeepSort for tracking, the model identifies and tracks individuals across video frames, even when they re-enter after leaving the frame or after occlusion (partial blocking of a person). The output video shows bounding boxes around detected individuals along with unique IDs to differentiate between them.

## How to Run the Project
### 1. Install Dependencies
To run the code, you need to install the required dependencies. These are included in the `requirements.txt` file. Run the following command in a bash terminal, VS Code terminal, or CMD prompt to install them:

```python
pip install -r requirements.txt
```

__The dependencies include:__

- YOLOv8 (for object detection)
- OpenCV (for video processing)
-  DeepSort (for tracking)
- PyTorch (to run YOLOv8)



### 2. Run the Code
You need to modify the video input and output paths in the script (`main.py`) according to your setup:

```python
video_path = r"Input_video.mp4"
output_path = r"Output_video.mp4"
```

Once the paths are set, run the following command in your terminal or command prompt:

```python
python main.py
```

This will process the input video and generate the output video with bounding boxes and unique IDs overlaid.

## Detailed Logic Behind Model Predictions
### A. Object Detection with YOLOv8
The YOLOv8 model is used to detect people in each frame of the video. Here's how it works:

1. **Frame-by-Frame Processing:** The video is read frame by frame. For each frame, the YOLOv8 model is applied to detect objects.

2. **Bounding Box Predictions:** YOLOv8 outputs bounding boxes, confidence scores, and class labels. A bounding box is a rectangular box that surrounds a detected object (in this case, people). The class label for people is `0` (class `0` represents the "person" category).

3. **Filtering Predictions:** The model may detect various objects, but we are only interested in people. Therefore, we filter out any object whose class is not "person" (class `0`). Additionally, we filter out detections with a confidence score lower than `0.7` to reduce false positives.
   - __Bounding Box Coordinates:__ Each bounding box is represented by four values (x1, y1, x2, y2), which define the top-left and bottom-right corners of the box.
   - __Confidence Score:__ This is a value between 0 and 1 indicating how confident the model is that the object detected is a person.
  
### B. Tracking with DeepSort
Once YOLOv8 detects people, their bounding boxes and confidence scores are passed to the DeepSort tracker, which tracks individuals across frames and assigns them unique IDs.

1. Tracking Individuals: DeepSort uses Kalman filters and object re-identification to assign a unique ID to each person detected in the video. This ensures that the same person has the same ID across different frames, even if they leave the frame and re-enter.

2. **Handling Occlusions and Re-entries:**

    - If a person is temporarily blocked (occluded) or leaves and re-enters the frame, DeepSort tries to maintain the same ID by matching the new detection to the previous one using bounding box coordinates and appearance features.
    - __Post-Occlusion Tracking:__ If a person is occluded (partially blocked) by another object or person, the tracker continues to track them once they reappear.
   
3. __Track Management:__ DeepSort keeps track of people for a defined number of frames, even if they temporarily leave the frame (`max_age=300`). This allows for continuous tracking even with interruptions.


### C. Combining Detection and Tracking
In each frame:

1. YOLOv8 detects the people and outputs their bounding boxes.
2. These detections are passed to DeepSort, which assigns unique IDs and updates the track of each person.
3. The bounding boxes and IDs are then drawn on the frame, showing the predictions visually.


### D. Visualizing Results
For each frame, after the detection and tracking are done:

- Bounding Boxes: Drawn around each detected person.
- Unique IDs: Each detected person is assigned an ID, which is displayed near the bounding box.

The processed frame is then written to the output video file.

### E. Thresholds and Parameters
- __Confidence Threshold:__ We use a confidence threshold of `0.7` to filter out low-confidence detections.
- __Tracking Parameters:__ The DeepSort tracker is configured with a maximum age of `300` frames to handle cases where people leave and re-enter the frame.

### F. Post-Processing
After the entire video is processed, the output video (with bounding boxes and unique IDs) is saved to the path you specified (`output_path`).


## Notes on Model Behavior
- The __YOLOv8 model__ is robust in detecting people, but certain challenges like occlusions, fast movement, or crowded scenes may still affect detection quality.
- __DeepSort__ performs well for tracking but may struggle if a person’s appearance changes drastically between frames (e.g., due to heavy occlusion or frame skips).
- __Post-occlusion handling__ ensures the model continues to track people even when they are partially blocked or leave and re-enter the scene.


## Results
1. __Dependencies:__ Make sure the required libraries are installed via `requirements.txt`.
2. __Test Video:__ Use the provided test video or your own video as input to check how well the detection and tracking work.
3. __Output:__ The output video will contain bounding boxes and unique IDs for the detected people.


## Deliverables
1. __process_video.py:__ The Python script for detecting and tracking people in a video.
2. __requirements.txt:__ The dependencies required to run the script.
3. __Output Video:__ The video with bounding boxes and unique IDs overlaid on the persons detected.
4. __README.md:__ This file explaining the logic and instructions for running the project.


## Troubleshooting
- If the YOLO model fails to load, make sure you have the correct version of PyTorch and YOLO installed.
- If the video processing is slow, consider reducing the video resolution or using a lighter model (e.g., YOLOv8n instead of larger variants).
- If DeepSort isn’t tracking people correctly, you can tweak parameters like `max_iou_distance` or `nn_budget` to improve performance.
