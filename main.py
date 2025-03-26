import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import json
import os

# Define a function to get the class names based on the class_id
def get_class_name(class_id):
    class_names = [
        'person'
    ]
    return class_names[class_id] if class_id < len(class_names) else 'GRID'

# Mouse callback function to handle the drawing of the bounding box
drawing = False
start_point = None
end_point = None
static_boxes = []
selected_box_index = -1  # No box selected initially
current_text = "Object"  # Default text for new boxes
editing_text = False  # Flag to determine if the text is being edited

# File path to save/load boxes
boxes_file = 'boxes.json'

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, static_boxes, selected_box_index, current_text, editing_text

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")  # Debugging

        # Check if the click is inside an existing box
        for i, box in enumerate(static_boxes):
            x1, y1, x2, y2 = box['box']
            if x1 < x < x2 and y1 < y < y2:
                selected_box_index = i  # Select the box
                current_text = box['label']  # Set the text to the label of the selected box
                print(f"Box {i} selected with label '{current_text}'")  # Debugging
                return  # Stop execution! Prevents creating a new box

        # If no box was selected, start drawing a new one
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            end_point = (x, y)

            # Ensure the box is not just a single point (dragging must happen)
            if abs(start_point[0] - end_point[0]) > 5 and abs(start_point[1] - end_point[1]) > 5:
                static_boxes.append({'box': (start_point[0], start_point[1], end_point[0], end_point[1]), 'label': current_text})
                print(f"New box added: {static_boxes[-1]}")  # Debugging
            else:
                print("Ignored tiny box creation")  # Debugging

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live camera processing")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,  # Default to 1 as it's likely the OBS Virtual Camera
        help="Index of the camera device (default: 1 for OBS Virtual Camera)"
    )
    parser.add_argument(
        "--webcam-resolution",
        default=[1920, 1080],
        nargs=2,
        type=int,
        help="Resolution of the camera frames"
    )
    args = parser.parse_args()
    return args

def save_boxes():
    # Save the boxes and labels to a JSON file
    with open(boxes_file, 'w') as file:
        json.dump(static_boxes, file)
    print(f"Boxes saved to {boxes_file}")

def load_boxes():
    # Load the boxes from a JSON file if it exists
    if os.path.exists(boxes_file):
        with open(boxes_file, 'r') as file:
            loaded_boxes = json.load(file)
            print(f"Loaded {len(loaded_boxes)} boxes from {boxes_file}")
            return loaded_boxes
    return []

def wipe_all_boxes():
    global static_boxes
    static_boxes = []  # Clear all drawn boxes
    if os.path.exists(boxes_file):
        os.remove(boxes_file)  # Delete the JSON file to wipe it completely
    print("All boxes wiped and JSON file deleted!")

def main():
    global static_boxes, selected_box_index, current_text, editing_text
    
    # Parse arguments
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Load previously saved boxes if available
    static_boxes = load_boxes()

    # Open the OBS Virtual Camera
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera_index}")
        return

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8m.pt")  # Load YOLOv8 model
    bounding_box_annotator = sv.BoxAnnotator()  # Create annotator instance

    # Set mouse callback for drawing bounding boxes
    cv2.namedWindow("YOLOv8 Live Feed")
    cv2.setMouseCallback("YOLOv8 Live Feed", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if frame is not captured (end of video)
        
        result = model(frame)[0]  # Run detection on the frame

        # Extract detections
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Convert class IDs to integers

        # Create supervision Detections object with the extracted values
        detections = sv.Detections(
            xyxy=boxes,  # Use xyxy bounding boxes
            confidence=confidences,
            class_id=class_ids
        )

        # Annotate the frame with bounding boxes
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)

        # Loop through all the detected objects and add the class names as text
        for i, box in enumerate(boxes):
            class_name = get_class_name(class_ids[i])
            x1, y1, x2, y2 = box
            text_position = (int(x1), int(y1) - 10)  # Slightly above the bounding box
            cv2.putText(
                frame, 
                class_name, 
                text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9,  # Font size
                (0, 255, 0),  # Green color for text
                2  # Thickness of the text
            )
        
        # Define a list of OpenCV colors
        colors = [
        (255, 0, 0),   # Blue
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Yellow
        (128, 0, 128), # Purple
        (0, 128, 128)  # Teal
        ]

        # Draw static boxes (the ones drawn by the user)
        for i, static_box in enumerate(static_boxes):
            x1, y1, x2, y2 = static_box['box']
            label = static_box['label']
            color = colors[i % len(colors)]  # Cycle through the color list
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Show the frame with bounding boxes and static boxes
        cv2.imshow("YOLOv8 Live Feed", frame)

        # Handle keyboard events to edit the text
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC key to exit
            save_boxes()  # Save boxes when exiting
            break
        elif key == ord('+'):
            print("Plus key pressed!")  # Debugging
            if selected_box_index != -1:
                editing_text = True  # Enable text editing
                print(f"Editing enabled for box {selected_box_index} with label '{static_boxes[selected_box_index]['label']}'")
        elif key in [8, 255]:  # Backspace to delete the last character# 255 is often the Backspace code on some layouts

            if selected_box_index != -1 and editing_text:
                current_text = current_text[:-1]
        elif key == 60:  # '<' key pressed to delete the selected box
            if selected_box_index != -1:
                print(f"Deleting box {selected_box_index} with label '{static_boxes[selected_box_index]['label']}'")
                del static_boxes[selected_box_index]  # Delete the selected box
                selected_box_index = -1  # Reset the selected box index
        elif key == 13:  # Enter key to confirm the new label
            if selected_box_index != -1:
                static_boxes[selected_box_index]['label'] = current_text
                selected_box_index = -1  # Deselect box after confirming text
                editing_text = False  # Disable editing mode
        elif key == 42:  # '*' key pressed to wipe all boxes
            wipe_all_boxes()  # Wipe all boxes and delete the JSON file

        # Handle typing any character while editing
        elif key >= 32 and key <= 126 and editing_text:  # Typing any character while editing
            if selected_box_index != -1:
                current_text += chr(key)  # Append the character to the current label
                static_boxes[selected_box_index]['label'] = current_text  # Update the label immediately


    cap.release()  # Release video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
