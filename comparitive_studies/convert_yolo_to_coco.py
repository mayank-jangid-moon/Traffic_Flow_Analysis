import os
import json
import glob
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

def create_coco_structure():
    """Create the basic COCO JSON structure"""
    coco_output = {
        "info": {
            "description": "Converted from YOLO format",
            "version": "1.0",
            "year": 2023,
            "contributor": "converter",
            "date_created": ""
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    return coco_output

def yolo_to_coco_coordinates(bbox, image_width, image_height):
    """
    Convert YOLO bbox format (x_center, y_center, width, height) normalized
    to COCO bbox format (x_min, y_min, width, height) in absolute coordinates
    """
    x_center, y_center, width, height = bbox
    
    # Denormalize the coordinates
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    # Convert to COCO format (x_min, y_min, width, height)
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    
    return [float(x_min), float(y_min), float(width), float(height)]

def convert_yolo_to_coco(yolo_dir, coco_dir, class_names=None, split_name="train"):
    """
    Convert YOLO dataset to COCO format
    
    Args:
        yolo_dir (str): Source directory with YOLO format annotations
        coco_dir (str): Target directory for COCO format
        class_names (list): List of class names
        split_name (str): Name of the dataset split (train, val/valid, test)
    """
    # Create output directory if it doesn't exist
    coco_split_name = 'valid' if split_name == 'val' else split_name
    coco_split_dir = os.path.join(coco_dir, coco_split_name)
    os.makedirs(coco_split_dir, exist_ok=True)
    
    # Get images and labels paths
    images_dir = os.path.join(yolo_dir, "images", split_name)
    labels_dir = os.path.join(yolo_dir, "labels", split_name)
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return False
        
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return False
    
    # Get image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    
    # Verify that we found image files
    if len(image_files) == 0:
        print(f"No image files found in {images_dir}")
        return False
        
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Create COCO JSON structure
    coco_output = create_coco_structure()
    
    # Add categories based on given class names
    if not class_names:
        # Try to read classes from dataset.yaml
        yaml_path = os.path.join(yolo_dir, "data.yaml")
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    class_names = data.get('names', [])
                    if isinstance(class_names, dict):
                        # Convert dict to list if needed
                        max_id = max(class_names.keys())
                        class_list = ["unknown"] * (max_id + 1)
                        for class_id, class_name in class_names.items():
                            class_list[class_id] = class_name
                        class_names = class_list
            except Exception as e:
                print(f"Error reading yaml: {e}")
                class_names = []
    
    if not class_names:
        # Create generic class names if nothing is provided
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        unique_class_ids = set()
        for label_file in label_files:
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            unique_class_ids.add(class_id)
        
        class_names = [f"class_{i}" for i in range(max(unique_class_ids) + 1 if unique_class_ids else 0)]
    
    # Add categories to COCO structure
    for i, class_name in enumerate(class_names):
        category = {
            "id": i + 1,  # COCO uses 1-indexed category IDs
            "name": class_name,
            "supercategory": "none"
        }
        coco_output["categories"].append(category)
    
    print(f"Processing {len(image_files)} images")
    
    # Process each image
    annotation_id = 1  # COCO uses 1-indexed annotation IDs
    for image_id, image_path in enumerate(image_files, 1):  # COCO uses 1-indexed image IDs
        # Get the filename and extension
        file_name = os.path.basename(image_path)
        file_base = os.path.splitext(file_name)[0]
        
        # Try to open and get image dimensions
        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
        
        # Add image to COCO structure
        coco_image = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": 1
        }
        coco_output["images"].append(coco_image)
        
        # Copy image to COCO directory
        shutil.copy2(image_path, os.path.join(coco_split_dir, file_name))
        
        # Find corresponding label file
        label_path = os.path.join(labels_dir, f"{file_base}.txt")
        
        # If label file exists, process annotations
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class_id x_center y_center width height
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        
                        # Convert YOLO coordinates to COCO coordinates
                        x_min, y_min, box_width, box_height = yolo_to_coco_coordinates(bbox, width, height)
                        
                        # Add annotation to COCO structure
                        coco_annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # COCO uses 1-indexed category IDs
                            "bbox": [x_min, y_min, box_width, box_height],
                            "area": box_width * box_height,
                            "segmentation": [],
                            "iscrowd": 0
                        }
                        coco_output["annotations"].append(coco_annotation)
                        annotation_id += 1
    
    # Save the COCO JSON file
    with open(os.path.join(coco_split_dir, "_annotations.coco.json"), 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    print(f"Converted {len(image_files)} images with {annotation_id-1} annotations to COCO format.")
    print(f"COCO dataset saved to: {coco_split_dir}")
    print(f"Categories: {[cat['name'] for cat in coco_output['categories']]}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO format dataset to COCO format')
    parser.add_argument('--yolo-dir', type=str, required=True, help='Directory with YOLO dataset')
    parser.add_argument('--coco-dir', type=str, required=True, help='Output directory for COCO dataset')
    parser.add_argument('--classes', type=str, help='Comma-separated class names')
    args = parser.parse_args()
    
    class_names = None
    if args.classes:
        class_names = args.classes.split(',')
    
    # Convert each split (train, val, test)
    for split in ['train', 'val', 'test']:
        print(f"\nConverting {split} split to COCO format...")
        convert_yolo_to_coco(args.yolo_dir, args.coco_dir, class_names, split_name=split)

if __name__ == "__main__":
    main()
