import os
import random
import cv2

def read_yolo_annotations(annotation_path):
    """
    Read YOLO annotations from a file and return bounding box information.

    Args:
        annotation_path (str): Path to the YOLO annotation file.

    Returns:
        list: A list of bounding boxes with format [class_id, x_center, y_center, width, height].
    """
    boxes = []
    with open(annotation_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure valid YOLO annotation format
                boxes.append([float(x) for x in parts])
    return boxes

def draw_bounding_boxes(image, boxes, class_names):
    """
    Draw bounding boxes on an image using YOLO annotations.

    Args:
        image (numpy.ndarray): The image to draw on.
        boxes (list): List of bounding boxes in YOLO format.
        class_names (dict): Dictionary of class IDs to class names.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn.
    """
    height, width, _ = image.shape
    for box in boxes:
        class_id, x_center, y_center, box_width, box_height = box
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)
        
        # Draw rectangle and label
        color = (0, 255, 0)  # Green color for bounding boxes
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        label = class_names.get(int(class_id), f"Class {int(class_id)}")
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def process_images(image_folder, annotation_folder, output_folder, class_names, sample_size=100):
    """
    Process random images, draw bounding boxes using YOLO annotations, and save them.

    Args:
        image_folder (str): Path to the folder containing images.
        annotation_folder (str): Path to the folder containing YOLO annotations.
        output_folder (str): Path to save images with bounding boxes.
        class_names (dict): Dictionary of class IDs to class names.
        sample_size (int): Number of images to process.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    selected_images = random.sample(image_files, min(sample_size, len(image_files)))
    
    for image_file in selected_images:
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')
        
        if not os.path.exists(annotation_path):
            print(f"No annotation found for {image_file}. Skipping...")
            continue
        
        # Load image and annotations
        image = cv2.imread(image_path)
        boxes = read_yolo_annotations(annotation_path)
        
        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image, boxes, class_names)
        
        # Save the processed image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image_with_boxes)
        #print(f"Saved: {output_path}")

# Paths for images, annotations, and output
image_folder = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\images_yolo\train'
annotation_folder = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\ann_yolo\train'
output_folder = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\ConversionTest'
class_names = {0: 'car', 1: 'pedestrian', 2: 'cyclist', 3:'truck', 4:'other vehicles'}  # Update with your class names

# Process and save images with bounding boxes
process_images(image_folder, annotation_folder, output_folder, class_names)
