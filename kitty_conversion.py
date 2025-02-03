import os
import json
import glob
import random
import shutil

###########################################################################################################################
#KITTY CONVERSION

# Input directories
input_ann_dir = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\ann'
input_img_dir = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\img'

# Output directories for annotations and images
output_ann_train = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\ann_yolo\train'
output_ann_val = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\ann_yolo\val'
output_img_train = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\images_yolo\train'
output_img_val = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\images_yolo\val'

# Ensure the output directories exist
os.makedirs(output_ann_train, exist_ok=True)
os.makedirs(output_ann_val, exist_ok=True)
os.makedirs(output_img_train, exist_ok=True)
os.makedirs(output_img_val, exist_ok=True)

# Define a mapping for object classes to YOLO class IDs
class_mapping = {
    'car': 0,
    'pedestrian': 1,
    'person sitting' : 1,
    'cyclist': 2,
    'truck': 3,
    'van': 4,
    # Skip 'dont care' class entirely
}

# Get all JSON files in the input annotations directory
json_files = glob.glob(os.path.join(input_ann_dir, '*.json'))

# Shuffle the files for randomness and create an 80-20 split
random.shuffle(json_files)
split_index = int(0.8 * len(json_files))
#print(split_index)
train_files = json_files[:split_index]
val_files = json_files[split_index:]

# Function to process annotations and move images
def process_files(file_list, output_ann_dir, output_img_dir):
    for json_file in file_list:
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Get image width and height
        img_width = data['size']['width']
        img_height = data['size']['height']

        # Prepare YOLO formatted data
        yolo_lines = []

        for obj in data['objects']:
            class_title = obj['classTitle']

            # Map the class to a YOLO class ID using the defined mapping
            if class_title in class_mapping:
                class_id = class_mapping[class_title]
            else:
                # Skip if the class is not in the mapping
                continue

            x_min, y_min = obj['points']['exterior'][0]
            x_max, y_max = obj['points']['exterior'][1]

            # Calculate YOLO format values
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Create a YOLO formatted line
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Extract the original filename (remove .json and ensure no .png remains)
        base_filename = os.path.basename(json_file).replace('.json', '')  # Remove .json
        if base_filename.endswith('.png'):
            base_filename = base_filename[:-4]  # Remove .png if it's there

        output_filename = f"{base_filename}.txt"  # Use the original filename for the .txt file

        # Save to a YOLO text file in the appropriate output directory
        with open(os.path.join(output_ann_dir, output_filename), 'w') as f:
            for line in yolo_lines:
                f.write(line + '\n')

        # Move the corresponding image file to the output image directory
        image_file = os.path.join(input_img_dir, f"{base_filename}.png")
        if os.path.exists(image_file):
            shutil.copy(image_file, os.path.join(output_img_dir, f"{base_filename}.png"))

# Process training and validation files
process_files(train_files, output_ann_train, output_img_train)
process_files(val_files, output_ann_val, output_img_val)

print(f"Processed {len(train_files)} annotation and image pairs to training directories")
print(f"Processed {len(val_files)} annotation and image pairs to validation directories")
#########################################################################################################################
