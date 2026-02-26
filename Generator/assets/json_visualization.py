import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path, brightness_beta=50):
    """
    Load an image, normalize non-uint8 data to 0-255, and adjust brightness.
    If the image is mono (single channel), convert to 3-channel RGB format.
    
    Parameters:
      brightness_beta: Value to control brightness enhancement (default 50), higher values make image brighter
    """
    # Use IMREAD_UNCHANGED to read original data
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    print(f"Original image dimensions: {image.shape}, dtype: {image.dtype}")

    # If data is not uint8, normalize to 0-255
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        print("Image data normalized to uint8.")
    
    # Check if single channel, convert to 3-channel RGB if so
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print("Single channel image converted to RGB format.")
    else:
        # If image is color (BGR), convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Adjust brightness: higher beta values make image brighter
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness_beta)
    print(f"Brightness increased with beta={brightness_beta}.")
    
    return image

def load_annotations(image_path):
    """
    Automatically load annotation file with same name as image but with .json extension.
    Expected JSON format contains:
      - "boxes": list, each element in format [x, y, width, height]
      - "masks": list, each element is a 2D list (same size as corresponding bounding box, non-zero values indicate mask region)
    """
    base, _ = os.path.splitext(image_path)
    json_path = base + ".json"
    if not os.path.exists(json_path):
        raise ValueError(f"Annotation file does not exist: {json_path}")
    print(f"Loading annotation file: {json_path}")
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    print(f"Annotation content keys: {list(annotations.keys())}")
    return annotations

def calculate_void_fraction_and_bubble_count(annotations, image_shape):
    """
    Calculate void fraction and bubble count from annotations.
    
    Parameters:
      annotations: Dictionary containing "boxes" and "masks"
      image_shape: Tuple of (height, width) of the original image
    
    Returns:
      void_fraction: Percentage of image covered by bubbles
      bubble_count: Number of detected bubbles
    """
    h, w = image_shape[:2]
    total_pixels = h * w
    bubble_pixels = 0
    
    boxes = annotations.get("boxes", [])
    masks = annotations.get("masks", [])
    bubble_count = len(boxes)
    
    for box, mask_list in zip(boxes, masks):
        x, y, w_box, h_box = box
        
        # Convert mask_list to numpy array
        mask = np.array(mask_list, dtype=np.uint8)
        
        # Check if mask dimensions match box dimensions
        if mask.shape[0] != h_box or mask.shape[1] != w_box:
            continue
            
        # Check if box is within image bounds
        if y + h_box > h or x + w_box > w:
            continue
            
        # Count non-zero pixels in mask
        bubble_pixels += np.count_nonzero(mask)
    
    void_fraction = (bubble_pixels / total_pixels) * 100
    return void_fraction, bubble_count

def plot_image_with_annotations(image, annotations, alpha_mask=0.5, save_path=None):
    """
    Draw bounding boxes and masks on image, and save as TIFF format (if save path is provided).
    The saved image dimensions match the original image without adding white borders.
    Display void fraction and bubble count in the corner.
    
    Parameters:
      image: RGB format image (numpy array)
      annotations: Dictionary containing "boxes" and "masks"
      alpha_mask: Transparency for mask overlay (0~1)
      save_path: Save path (if not None, save as TIFF file)
    """
    h, w = image.shape[:2]
    dpi = 300  # DPI for saving, ensures consistent size: output size = (w, h) pixels

    # Calculate void fraction and bubble count
    void_fraction, bubble_count = calculate_void_fraction_and_bubble_count(annotations, image.shape)
    
    # Set figure size based on original image dimensions (units: inches)
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.imshow(image, vmin=0, vmax=255)
    ax.axis('off')
    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print("Starting annotation overlay...")

    boxes = annotations.get("boxes", [])
    masks = annotations.get("masks", [])
    if len(boxes) != len(masks):
        print("Warning: Number of boxes and masks do not match!")

    # Create a new overlay with same size as original image for accumulating mask information
    overlay = np.zeros_like(image, dtype=np.uint8)
    # Use color #4878CF for masks, corresponding to RGB values [72, 120, 207]
    overlay_color = np.array([72, 120, 207], dtype=np.uint8)

    # Iterate through each annotation
    for idx, (box, mask_list) in enumerate(zip(boxes, masks)):
        # Box format is [x, y, width, height]
        x, y, w_box, h_box = box
        print(f"Annotation {idx}: box = {box}")
        # Draw bounding box using color #E24A33
        rect = patches.Rectangle((x, y), w_box, h_box, linewidth=2, edgecolor="#E24A33", facecolor='none')
        ax.add_patch(rect)

        # Convert mask_list to numpy array (assuming mask_list is a 2D list)
        mask = np.array(mask_list, dtype=np.uint8)
        print(f"Annotation {idx}: mask shape = {mask.shape}")

        # Check if mask dimensions match box dimensions
        if mask.shape[0] != h_box or mask.shape[1] != w_box:
            print(f"Annotation {idx}: mask dimensions {mask.shape} do not match box dimensions ({h_box}, {w_box}), skipping this annotation.")
            continue

        # Check if box is within image bounds
        if y + h_box > h or x + w_box > w:
            print(f"Annotation {idx}: box exceeds image boundaries, skipping.")
            continue

        # Binarize mask (non-zero values are foreground)
        mask_binary = (mask > 0).astype(np.uint8)
        # Accumulate mask region color in overlay
        overlay[y:y+h_box, x:x+w_box] = np.where(mask_binary[..., None], overlay_color, overlay[y:y+h_box, x:x+w_box])

    # Draw overlay once (mask overlaid with alpha_mask transparency)
    ax.imshow(overlay, alpha=alpha_mask, vmin=0, vmax=255)
    
    # Add text displaying void fraction and bubble count in the top-left corner
    text_str = f"Void Fraction: {void_fraction:.2f}%\nBubble Count: {bubble_count}"
    ax.text(10, 30, text_str, fontsize=12, color='white', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

    # Save image as TIFF format (if save path is specified), maintain same size as original without white borders
    if save_path is not None:
        fig.savefig(save_path, format='tiff', dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"Image saved as {save_path}")

    plt.close(fig)
    print("Image processing completed.")
    print(f"Void Fraction: {void_fraction:.2f}%")
    print(f"Bubble Count: {bubble_count}")

if __name__ == "__main__":
    # Modify this to actual image file path, e.g., "../src/TestDB/run11/synth_0001.png"
    image_path = "../output/run1/synth_0002.png"
    # Auto-generate save filename: original filename prefix + "_annotated.tif"
    base, _ = os.path.splitext(image_path)
    save_tif_path = base + "_annotated.tif"
    print(f"Processing image: {image_path}")

    try:
        # Load image (automatically converts to RGB, normalizes, adjusts brightness)
        image = load_image(image_path, brightness_beta=100)
        # Load JSON annotation file with same name as image
        annotations = load_annotations(image_path)
        # Draw image with annotations and save as TIFF format (same dimensions as original, no white borders)
        plot_image_with_annotations(image, annotations, alpha_mask=0.5, save_path=save_tif_path)
    except Exception as e:
        print(f"Error occurred: {e}")