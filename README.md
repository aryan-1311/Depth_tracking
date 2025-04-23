<h1><b>Depth Tracking: 3D Object Detection and Depth Estimation</h1></b>
This repository implements a 3D object detection system with depth estimation and bird's eye view (BEV) visualization. It uses YOLOv11 for object detection and Depth Anything v2 for depth estimation to process video input (from a webcam or file) and generate an output video with visualizations of detected objects, their estimated 3D bounding boxes, depth maps, and BEV representations.
Repository

GitHub: https://github.com/aryan-1311/Depth_tracking

Prerequisites

Python: Version 3.8 or higher
Operating System: Windows, macOS, or Linux
Hardware:
CPU: Multi-core processor (Intel or AMD)
GPU (optional but recommended): NVIDIA GPU with CUDA support for faster processing
RAM: At least 8GB (16GB or more recommended for large models)


Dependencies: Listed in requirements.txt

Installation

Clone the Repository:
git clone https://github.com/aryan-1311/Depth_tracking.git
cd Depth_tracking


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

Install the required Python packages listed in requirements.txt:pip install -r requirements.txt


Note: Ensure you have a compatible version of PyTorch with CUDA support if using a GPU. You may need to install PyTorch separately based on your CUDA version:pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Check the PyTorch website for the appropriate command for your system.


Verify Installation:

Ensure all dependencies are installed correctly by running:python -c "import torch, cv2, numpy, ultralytics, timm, filterpy; print('All dependencies installed')"





Project Structure

run.py: Main script to execute the 3D object detection pipeline.
detection_model.py: Implements object detection using YOLOv11.
depth_model.py: Implements depth estimation using Depth Anything v2.
bbox3d_utils.py: Utilities for 3D bounding box estimation and BEV visualization.
load_camera_params.py: Functions to load and apply camera parameters.
requirements.txt: List of required Python packages.
README.md: This file.

Usage
Running the System

Prepare Input:

Webcam: Use the default webcam by setting source = 0 in run.py.
Video File: Specify the path to a video file (e.g., source = "path/to/video.mp4").
Camera Parameters (optional): Provide a JSON file with camera intrinsic and extrinsic parameters (see load_camera_params.py). If not provided, default parameters are used.


Configure Settings:

Open run.py and modify the configuration variables in the main() function as needed:
source: Input source (webcam index or video file path, default: 0).
output_path: Path to save the output video (e.g., "output_me_3.mp4").
yolo_model_size: YOLOv11 model size (nano, small, medium, large, extra, default: large).
depth_model_size: Depth Anything v2 model size (small, base, large, default: large).
device: Computation device (cuda, cpu, or mps for Apple Silicon, default: cpu).
conf_threshold: Confidence threshold for object detection (default: 0.5).
iou_threshold: IoU threshold for non-maximum suppression (default: 0.45).
classes: Filter by class IDs (e.g., [0, 1, 2] for specific classes, default: None for all).
enable_tracking: Enable/disable object tracking (default: True).
enable_bev: Enable/disable bird's eye view visualization (default: True).
enable_pseudo_3d: Enable/disable pseudo-3D visualization (default: True).
camera_params_file: Path to camera parameters JSON file (default: None).




Execute the Script:

Run the main script:python run.py


The system will:
Load the specified models.
Process the input video or webcam feed.
Display three windows: Object Detection, Depth Map, and 3D Object Detection (with BEV if enabled).
Save the output video to the specified output_path.




Interact with the Program:

Press q or Esc to exit the program.
The output video will be saved to the specified path upon exit.



Example Commands

Process a video file:

Edit run.py to set source = "path/to/video.mp4" and output_path = "output.mp4", then run:python run.py




Use webcam:

Ensure source = 0 in run.py, then run:python run.py





Output

Output Video: Saved to the specified output_path (e.g., output_me_3.mp4).
Includes:
2D bounding boxes with object labels and IDs (if tracking is enabled).
Pseudo-3D bounding boxes with depth information.
Depth map in the top-left corner.
Bird's eye view visualization in the bottom-left corner (if enabled).




