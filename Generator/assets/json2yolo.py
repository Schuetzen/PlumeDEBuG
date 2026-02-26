import os
import json
import random
import shutil
import cv2
import numpy as np

# ========== Configuration Parameters ==========
INPUT_DIR = "../output/run1"  # Folder containing PNG and JSON files, modify to actual path
OUTPUT_DIR = "test"
TRAIN_RATIO = 0.7  # Training set ratio, remaining 30% for validation set

# Output directory structure
IMAGE_TRAIN_DIR = os.path.join(OUTPUT_DIR, "images", "train")
IMAGE_VAL_DIR   = os.path.join(OUTPUT_DIR, "images", "val")
LABEL_TRAIN_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
LABEL_VAL_DIR   = os.path.join(OUTPUT_DIR, "labels", "val")

# ========== Helper Functions ==========

def create_dirs():
    """Create required output directories"""
    os.makedirs(IMAGE_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMAGE_VAL_DIR, exist_ok=True)
    os.makedirs(LABEL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LABEL_VAL_DIR, exist_ok=True)

def simplify_polygon(contour, epsilon_factor=0.005):
    """
    Simplify polygon to reduce number of points
    contour: OpenCV contour points
    epsilon_factor: Approximation accuracy coefficient, larger values mean more simplification
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def convert_annotation(json_path, image_path, debug=False):
    """
    Read JSON annotation and convert to YOLO format string list.
    YOLO segmentation format per line:
      <class> <x1> <y1> <x2> <y2> <x3> <y3> ...
    Where all coordinates are normalized to [0-1] range.
    """
    # Read image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image {image_path}")
    img_h, img_w = img.shape[:2]
    
    # Read JSON annotation
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    boxes = data.get("boxes", [])
    masks = data.get("masks", None)
    
    yolo_lines = []
    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        
        x, y, w, h = box
        
        # If JSON contains masks info and this object has corresponding mask, use mask to create polygon
        if masks is not None and i < len(masks):
            mask_data = masks[i]
            
            # Convert mask data to numpy array
            mask_array = np.array(mask_data, dtype=np.uint8) * 255
            
            # Check if mask dimensions match box dimensions
            if mask_array.shape[0] != int(h) or mask_array.shape[1] != int(w):
                print(f"Warning: mask size {mask_array.shape} doesn't match box size {(h, w)}, using bounding box conversion")
                
                # Use bounding box corners as polygon points
                polygon_points = [
                    [x, y],                 # top-left
                    [x + w, y],             # top-right
                    [x + w, y + h],         # bottom-right
                    [x, y + h]              # bottom-left
                ]
            else:
                # Extract mask contours
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Select contour with largest area
                    contour = max(contours, key=cv2.contourArea)
                    
                    # Simplify contour to reduce number of points
                    contour = simplify_polygon(contour)
                    
                    # Convert contour from local (relative to box) to global coordinates
                    contour = contour.reshape(-1, 2)
                    contour[:, 0] += int(x)
                    contour[:, 1] += int(y)
                    
                    # Ensure contour is within image bounds
                    contour[:, 0] = np.clip(contour[:, 0], 0, img_w - 1)
                    contour[:, 1] = np.clip(contour[:, 1], 0, img_h - 1)
                    
                    polygon_points = contour.tolist()
                else:
                    # If no contour found, use bounding box
                    polygon_points = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
        else:
            # If no mask available, use bounding box
            polygon_points = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]
        
        # Build YOLO format annotation line
        yolo_line = ["0"]  # Class ID, fixed to 0
        
        # Add normalized polygon point coordinates
        for pt in polygon_points:
            x_norm = pt[0] / img_w
            y_norm = pt[1] / img_h
            yolo_line.append(f"{x_norm:.6f}")
            yolo_line.append(f"{y_norm:.6f}")
        
        yolo_lines.append(" ".join(yolo_line))
        
        if debug:
            print(f"Conversion result: {yolo_line}")
            
    return yolo_lines

# ========== Main Function ==========

def main():
    create_dirs()
    
    # Get all PNG files in input folder
    files = os.listdir(INPUT_DIR)
    image_files = [f for f in files if f.lower().endswith(".png")]
    base_names = [os.path.splitext(f)[0] for f in image_files]
    
    # Randomly shuffle sample order
    random.shuffle(base_names)
    split_index = int(len(base_names) * TRAIN_RATIO)
    train_names = base_names[:split_index]
    val_names = base_names[split_index:]
    
    print(f"Found {len(base_names)} samples total, training set: {len(train_names)}, validation set: {len(val_names)}")
    
    # Process training and validation sets separately
    for split, names in [("train", train_names), ("val", val_names)]:
        for base in names:
            image_src = os.path.join(INPUT_DIR, base + ".png")
            json_src = os.path.join(INPUT_DIR, base + ".json")
            
            if not os.path.exists(image_src):
                print(f"Warning: Image {image_src} does not exist, skipping.")
                continue
            if not os.path.exists(json_src):
                print(f"Warning: Annotation file {json_src} does not exist, skipping.")
                continue
            
            try:
                yolo_lines = convert_annotation(json_src, image_src)
            except Exception as e:
                print(f"Error converting {base}: {e}")
                continue
            
            # Determine output paths
            if split == "train":
                image_dst = os.path.join(IMAGE_TRAIN_DIR, base + ".png")
                label_dst = os.path.join(LABEL_TRAIN_DIR, base + ".txt")
            else:
                image_dst = os.path.join(IMAGE_VAL_DIR, base + ".png")
                label_dst = os.path.join(LABEL_VAL_DIR, base + ".txt")
            
            # Copy image
            shutil.copy2(image_src, image_dst)
            # Write YOLO format annotation file
            with open(label_dst, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
    
    print("Conversion and splitting completed.")

def test_conversion(json_file, image_file):
    """Test conversion function, print conversion results"""
    print(f"Testing files: {json_file}, {image_file}")
    yolo_lines = convert_annotation(json_file, image_file, debug=True)
    print(f"Conversion result: {len(yolo_lines)} objects")
    for line in yolo_lines:
        print(line)

if __name__ == "__main__":
    # You can uncomment the following code to test conversion results for a single file
    # test_sample = "sample_name"  # Replace with actual filename (without extension)
    # test_conversion(
    #     os.path.join(INPUT_DIR, test_sample + ".json"),
    #     os.path.join(INPUT_DIR, test_sample + ".png")
    # )
    
    main()