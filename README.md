# Person-Detection-and-Tracking-Using-YOLOv8-and-DeepSort
Person Detection and Tracking System for Analyzing Children with Autism Spectrum Disorder and Therapists. This project leverages YOLOv8 for detection and DeepSort for tracking, providing unique IDs for individuals in videos.

## Project Overview
This project focuses on detecting and tracking people (specifically children and therapists) in a video. Using YOLOv8 for detection and DeepSort for tracking, the model identifies and tracks individuals across video frames, even when they re-enter after leaving the frame or after occlusion (partial blocking of a person). The output video shows bounding boxes around detected individuals along with unique IDs to differentiate between them.

## How to Run the Project
### 1. Install Dependencies
To run the code, you need to install the required dependencies. These are included in the __requirements.txt__ file. Run the following command in a bash terminal, VS Code terminal, or CMD prompt to install them:

![image](https://github.com/user-attachments/assets/7d9ff08c-da39-42bc-8e44-305b99a53e22)

__The dependencies include:__

1. YOLOv8 (for object detection)
2. OpenCV (for video processing)
3. DeepSort (for tracking)
4. PyTorch (to run YOLOv8)



### 2. Run the Code
You need to modify the video input and output paths in the script (main.py) according to your setup:

![image](https://github.com/user-attachments/assets/be57ed46-cb0d-4a87-a270-e30c7ccd8b0a)

Once the paths are set, run the following command in your terminal or command prompt:

![image](https://github.com/user-attachments/assets/1dc439d7-decc-4db3-8141-d4d99ed0aefb)

This will process the input video and generate the output video with bounding boxes and unique IDs overlaid.


This will display the code block with syntax highlighting:

```python
def hello_world():
    print("Hello, World!")


