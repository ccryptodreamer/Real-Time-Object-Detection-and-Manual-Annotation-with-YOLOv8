




![Screenshot 2025-03-27 003708](https://github.com/user-attachments/assets/858050d7-ae81-4251-bb91-693e1c5258f6)



# YOLOv8 Live Camera Object Detection and Annotation

## Overview
This project uses **YOLOv8** to detect objects in a live camera feed and allows users to manually draw, label, and manage bounding boxes on the video stream. It provides an interactive way to annotate objects using OpenCV's mouse and keyboard interactions.

## Features
- **Real-time Object Detection**: Uses YOLOv8 to detect objects in the live video feed.
- **Manual Bounding Box Creation**: Click and drag to draw bounding boxes.
- **Labeling Objects**: Assign and edit text labels for manually drawn boxes.
- **Save and Load Annotations**: Saves drawn boxes and labels to a file (`boxes.json`).
- **Delete or Modify Boxes**: Edit, remove, or clear all bounding boxes with keyboard shortcuts.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- OpenCV (`cv2`)
- Ultralytics YOLOv8 (`ultralytics`)
- Supervision (`supervision`)
- NumPy (`numpy`)

Install dependencies with:
```bash
pip install opencv-python argparse ultralytics supervision numpy
```

## How It Works
1. The program opens a camera feed (defaulting to **camera index 1**, suitable for OBS Virtual Camera).
2. **YOLOv8** detects objects and draws bounding boxes in real-time.
3. Users can draw custom boxes using the **mouse**.
4. Labels can be edited using the **keyboard**.
5. All boxes are saved and can be reloaded in future sessions.

## Running the Script
Run the script using:
```bash
python script.py --camera-index 1 --webcam-resolution 1920 1080
```
Options:
- `--camera-index`: Choose a camera (default is `1` for OBS Virtual Camera).
- `--webcam-resolution`: Set the video resolution (default: `1920x1080`).

## Mouse Controls
- **Left Click**: Start drawing a box.
- **Drag**: Define the size of the box.
- **Release**: Save the box.
- **Click inside a box**: Select a box to edit its label.

## Keyboard Controls
- **`ESC`**: Exit and save boxes.
- **`+`**: Enable text editing mode.
- **Backspace (`⌫`)**: Delete last character in label.
- **`<`**: Delete selected box.
- **`Enter`**: Save edited label.
- **`*`**: Clear all drawn boxes.

## File Management
- Bounding boxes and labels are stored in `boxes.json`.
- Boxes are **automatically saved** on exit.
- If the file exists, boxes are **loaded** on startup.
- Press `*` to **wipe all saved boxes**.

## Troubleshooting
- **Camera Not Opening**: Try changing `--camera-index` (e.g., `0` for a laptop webcam).
- **Permission Errors**: Run the script as an administrator.
- **YOLO Model Missing**: Ensure `yolov8m.pt` is correctly downloaded and placed in the working directory.

## Future Improvements
- Support for multiple object classes.
- Customizable detection models.
- Export labeled images for training datasets.

## Author
Created using **YOLOv8**, OpenCV, and Python for real-time object annotation.



