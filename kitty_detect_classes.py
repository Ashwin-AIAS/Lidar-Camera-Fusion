import os
import json

def extract_classes_from_json(folder_path):
    class_titles = set()
    
    # Iterate over all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):  # Only process JSON files
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    
                    # Check for objects and extract classTitle
                    if 'objects' in data:
                        for obj in data['objects']:
                            if 'classTitle' in obj:
                                class_titles.add(obj['classTitle'])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file {file_name}: {e}")
    
    return class_titles

# Folder containing the JSON annotation files
folder_path = r'C:\Users\britt\Desktop\Python\project_camera\datasets\skeleton\kitty\train\ann'

# Extract and display the unique class titles
classes = extract_classes_from_json(folder_path)
print("Unique classes found:", classes)
