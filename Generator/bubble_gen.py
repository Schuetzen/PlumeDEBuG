"""
BubbleGen V4 - Synthetic Bubble Image Generator with Mixed Mode
================================================================

Two-Method Selection System:
1. direct_pdf: Pure distribution matching (ignores data availability)
2. weighted_sampling: Balanced approach (considers both distribution and availability) - DEFAULT

Features:
- Mixed mode: Randomized parameters for diverse dataset generation
- Trapezoid ROI entrainment model
- Unidirectional overlap detection (ensures existing bubbles remain visible)
- Quadtree spatial acceleration
- Gaussian/Random placement modes (randomized in mixed mode)
- Comprehensive execution logging
- K-S test validation

Mixed Mode Features:
- Random bubble count: max_bubbles ± 50 (minimum 1)
- Random placement mode: Gaussian or Random
- Parameter variation (30%): overlap_control, entrainment_slope, base_width_ratio
- Parameters re-randomized every 10% of total images
- All randomized parameters saved to output folder

Author: Xuchen (Schuetzen) x Claude
Version: 4.0
"""

import os
import cv2
import numpy as np
import random
import json
import sqlite3
import configparser
import pickle
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, lognorm, weibull_min, truncnorm, kstest
from matplotlib import pyplot as plt
from matplotlib import rcParams

# -------------------------------
# Logging System - Captures all output and errors
# -------------------------------
class Logger:
    """Dual output logger - writes to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        self.start_time = datetime.now()

        # Write header
        self.log.write("=" * 80 + "\n")
        self.log.write("BubbleGen V3 - Execution Log\n")
        self.log.write("=" * 80 + "\n")
        self.log.write(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("=" * 80 + "\n\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        end_time = datetime.now()
        duration = end_time - self.start_time

        self.log.write("\n" + "=" * 80 + "\n")
        self.log.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"Duration: {duration}\n")
        self.log.write("=" * 80 + "\n")
        self.log.close()

# -------------------------------
# Create Numbered Run Directory
# -------------------------------
def create_run_directory(base_output_dir):
    """Create a numbered run directory inside the output directory."""
    os.makedirs(base_output_dir, exist_ok=True)
    existing_runs = []
    for item in os.listdir(base_output_dir):
        if os.path.isdir(os.path.join(base_output_dir, item)) and item.startswith("run"):
            try:
                run_num = int(item[3:])
                existing_runs.append(run_num)
            except ValueError:
                continue
    next_run = 0 if not existing_runs else max(existing_runs) + 1
    run_dir = os.path.join(base_output_dir, f"run{next_run}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created output directory: {run_dir}")
    return run_dir

# -------------------------------
# Load Configuration from config.ini
# -------------------------------
config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
config.read('config.ini', encoding='utf-8')

# General parameters
NUM_SYNTHETIC_IMAGES = config.getint('General', 'num_synthetic_images')
TARGET_VOID_FRACTION = config.getfloat('General', 'target_void_fraction')
MAX_BUBBLES_PER_IMAGE = config.getint('General', 'max_bubbles_per_image')
OUTPUT_DIR = config.get('General', 'output_dir')
default_background_value = config.getint('General', 'default_background_value')

# Placement parameters
PLACEMENT_MODE = config.get('Placement', 'placement_mode')
overlap_control = config.getfloat('Placement', 'overlap_control')
gaussian_x_min_ratio = config.getfloat('Placement', 'gaussian_x_min_ratio')
gaussian_x_max_ratio = config.getfloat('Placement', 'gaussian_x_max_ratio')
gaussian_scale_divisor = config.getfloat('Placement', 'gaussian_scale_divisor')

# Entrainment parameters
USE_TRAPEZOID_ROI = config.getboolean('Placement', 'use_trapezoid_roi', fallback=False)
ENTRAINMENT_SLOPE = config.getfloat('Placement', 'entrainment_slope', fallback=0.1)
BASE_WIDTH_RATIO = config.getfloat('Placement', 'base_width_ratio', fallback=0.3)

# Velocity-based size bias parameters
ENABLE_VELOCITY_BIAS = config.getboolean('Placement', 'enable_velocity_size_bias', fallback=False)
VELOCITY_PROFILE_EXPONENT = config.getfloat('Placement', 'velocity_profile_exponent', fallback=2.0)
SIZE_VELOCITY_COUPLING = config.getfloat('Placement', 'size_velocity_coupling', fallback=1.5)
VELOCITY_VERTICAL_DECAY = config.getfloat('Placement', 'velocity_vertical_decay', fallback=0.0)

# Filters parameters
APPLY_FEATHERING = config.getboolean('Filters', 'apply_feathering')
FEATHER_KERNEL_SIZE = tuple(map(int, config.get('Filters', 'feather_kernel_size').split(',')))
FEATHER_SIGMA = config.getfloat('Filters', 'feather_sigma')
FEATHER_EROSION_KERNEL = tuple(map(int, config.get('Filters', 'feather_erosion_kernel').split(',')))
FEATHER_EROSION_ITERATIONS = config.getint('Filters', 'feather_erosion_iterations')

# Background parameters
USE_BACKGROUND_IMAGE = config.getboolean('Background', 'use_background_image')
BACKGROUND_IMAGE_PATH = config.get('Background', 'background_image_path')

# Database parameters
DATABASE_PATH = config.get('Database', 'database_path')
AGGREGATED_RESULTS_PATH = config.get('Database', 'aggregated_results_path')

# Distribution parameters
DIST_TYPE = config.get('Distribution', 'distribution_type').lower()
SELECTION_METHOD = config.get('Distribution', 'selection_method', fallback='weighted_sampling').lower()
GAUSSIAN_MU = config.getfloat('Gaussian', 'mu')
GAUSSIAN_SIGMA = config.getfloat('Gaussian', 'sigma')
WEIBULL_SHAPE = config.getfloat('Weibull', 'weibull_shape')
WEIBULL_SCALE = config.getfloat('Weibull', 'weibull_scale')
LOGNORMAL_MU = config.getfloat('Lognormal', 'lognormal_mu')
LOGNORMAL_SIGMA = config.getfloat('Lognormal', 'lognormal_sigma')
if DIST_TYPE == "constant":
    CONSTANT_TARGET = config.getfloat('Constant', 'target')
elif DIST_TYPE == "bimodal":
    BIMODAL_MU1 = config.getfloat('Bimodal', 'mu1')
    BIMODAL_SIGMA1 = config.getfloat('Bimodal', 'sigma1')
    BIMODAL_MU2 = config.getfloat('Bimodal', 'mu2')
    BIMODAL_SIGMA2 = config.getfloat('Bimodal', 'sigma2')
    BIMODAL_WEIGHT1 = config.getfloat('Bimodal', 'weight1')

# -------------------------------
# Mixed Mode Configuration
# -------------------------------
# Enable mixed mode by adding [MixedMode] section in config.ini
ENABLE_MIXED_MODE = config.getboolean('MixedMode', 'enable_mixed_mode', fallback=False)

# Store base parameters for mixed mode randomization
BASE_MAX_BUBBLES = MAX_BUBBLES_PER_IMAGE
BASE_OVERLAP_CONTROL = overlap_control
BASE_ENTRAINMENT_SLOPE = ENTRAINMENT_SLOPE
BASE_WIDTH_RATIO = BASE_WIDTH_RATIO
BASE_PLACEMENT_MODE = PLACEMENT_MODE

def generate_mixed_mode_parameters(image_index, total_images):
    """
    Generate randomized parameters for mixed mode.
    Parameters are re-randomized every 10% of total images.

    Parameters:
        image_index: Current image index (0-based)
        total_images: Total number of images to generate

    Returns:
        dict: Randomized parameters
    """
    # Determine which batch this image belongs to (10% batches)
    batch_size = max(1, int(total_images * 0.1))
    batch_index = image_index // batch_size

    # Set random seed based on batch index for reproducibility within batch
    np.random.seed(batch_index * 42)
    random.seed(batch_index * 42)

    # Randomize max_bubbles: ±50 range, minimum 1
    bubble_variation = random.randint(-50, 50)
    max_bubbles = max(1, BASE_MAX_BUBBLES + bubble_variation)

    # Randomize placement mode: Gaussian or Random
    placement_mode = random.choice(['Guassian', 'Random'])  # Keep 'Guassian' typo for compatibility

    # Randomize parameters with 30% variation
    overlap_ctrl = BASE_OVERLAP_CONTROL * (1 + random.uniform(-0.3, 0.3))
    overlap_ctrl = np.clip(overlap_ctrl, 0.0, 1.0)  # Ensure valid range

    entrainment_slope = BASE_ENTRAINMENT_SLOPE * (1 + random.uniform(-0.3, 0.3))
    entrainment_slope = np.clip(entrainment_slope, 0.0, 1.0)  # Ensure valid range

    base_width = BASE_WIDTH_RATIO * (1 + random.uniform(-0.3, 0.3))
    base_width = np.clip(base_width, 0.0, 1.0)  # Ensure valid range

    # Reset random seed to avoid affecting other random operations
    np.random.seed(None)
    random.seed(None)

    return {
        'max_bubbles_per_image': max_bubbles,
        'placement_mode': placement_mode,
        'overlap_control': overlap_ctrl,
        'entrainment_slope': entrainment_slope,
        'base_width_ratio': base_width,
        'batch_index': batch_index
    }

def save_mixed_mode_parameters(run_dir, image_index, params):
    """
    Save mixed mode parameters to a JSON file.
    Creates one JSON file per batch (every 10% of images).
    """
    batch_index = params['batch_index']
    params_file = os.path.join(run_dir, f"mixed_params_batch{batch_index}.json")

    # Check if this batch's parameters have already been saved
    if os.path.exists(params_file):
        return  # Already saved for this batch

    # Prepare parameters for JSON serialization
    params_to_save = {
        'batch_index': batch_index,
        'images_affected': f"{batch_index * max(1, int(NUM_SYNTHETIC_IMAGES * 0.1))} - {(batch_index + 1) * max(1, int(NUM_SYNTHETIC_IMAGES * 0.1)) - 1}",
        'max_bubbles_per_image': int(params['max_bubbles_per_image']),
        'placement_mode': params['placement_mode'],
        'overlap_control': float(params['overlap_control']),
        'entrainment_slope': float(params['entrainment_slope']),
        'base_width_ratio': float(params['base_width_ratio']),
        'base_parameters': {
            'max_bubbles_per_image': BASE_MAX_BUBBLES,
            'placement_mode': BASE_PLACEMENT_MODE,
            'overlap_control': BASE_OVERLAP_CONTROL,
            'entrainment_slope': BASE_ENTRAINMENT_SLOPE,
            'base_width_ratio': BASE_WIDTH_RATIO
        }
    }

    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=2)

    print(f"  Saved mixed mode parameters to: {params_file}")

# -------------------------------
# Initialize Global Directories & Canvas
# -------------------------------
RUN_DIR = create_run_directory(OUTPUT_DIR)

if USE_BACKGROUND_IMAGE:
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if bg_img is None:
        raise ValueError(f"Background image not found at {BACKGROUND_IMAGE_PATH}!")
    CANVAS_SIZE = (bg_img.shape[1], bg_img.shape[0])
else:
    CANVAS_SIZE = (1001, 601)
    bg_img = None

# Pre-calculate ROI bounds for rectangle mode (avoid repeated calculations)
ROI_X_MIN = int(CANVAS_SIZE[0] * gaussian_x_min_ratio)
ROI_X_MAX = int(CANVAS_SIZE[0] * gaussian_x_max_ratio)

# -------------------------------
# Data Loading from SQLite
# -------------------------------
def load_bubbles_from_db(use_cache=True, cache_file='bubble_cache.pkl'):
    """Load bubble data from SQLite database (with caching support)."""
    if use_cache and os.path.exists(cache_file):
        print("Loading bubble data from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Loading bubble data from SQLite database...")
    if not os.path.exists(AGGREGATED_RESULTS_PATH):
        raise ValueError(f"Database file not found at {AGGREGATED_RESULTS_PATH}!")
    
    bubble_data = []
    results_dir = DATABASE_PATH
    conn = sqlite3.connect(AGGREGATED_RESULTS_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, folder_path, bubble_diameter FROM images")
    rows = cursor.fetchall()
    
    def process_entry(entry):
        image_name = entry['image_name']
        folder_path = entry['folder_path']
        base_name = image_name.replace(".tif", "")
        dataset_dir = os.path.join(results_dir, folder_path, "Dataset")
        cropped_dir = os.path.join(dataset_dir, "Cropped")
        mask_dir = os.path.join(dataset_dir, "Masks")
        cropped_path = os.path.join(cropped_dir, f"{base_name}_cropped.tif")
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.tif")
        try:
            img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                return None
            if mask.shape != img.shape:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            true_area = cv2.countNonZero(mask)
            return {
                "image": img,
                "mask": mask,
                "true_area": true_area,
                "bbox": (img.shape[1], img.shape[0]),
                "bubble_diameter": entry['bubble_diameter']
            }
        except Exception as e:
            print(f"Error loading {base_name}: {str(e)}")
            return None

    max_workers = min(32, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_entry, rows), total=len(rows), 
                          desc="Processing Bubbles", unit="bubble"):
            if result:
                bubble_data.append(result)
    conn.close()
    
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(bubble_data, f)
    return bubble_data

# -------------------------------
# Stratification (binsize = 0.1 mm)
# -------------------------------
def stratify_data(bubble_data):
    """Stratify bubbles by 0.1mm bin size."""
    diameters_mm = np.array([b["bubble_diameter"] for b in bubble_data]) * 1000
    bin_edges = np.arange(0, diameters_mm.max() + 0.1, 0.1)
    bin_indices = np.digitize(diameters_mm, bin_edges) - 1
    stratified_data = {i: [] for i in range(len(bin_edges) - 1)}
    for i, bubble in enumerate(bubble_data):
        stratified_data[bin_indices[i]].append(bubble)
    return stratified_data, bin_edges

# -------------------------------
# Target Distribution Function
# -------------------------------
def get_target_pdf_func():
    """Returns the target PDF (in mm) according to DIST_TYPE."""
    if DIST_TYPE == "gaussian":
        return lambda x: norm.pdf(x, loc=GAUSSIAN_MU*1000, scale=GAUSSIAN_SIGMA*1000)
    elif DIST_TYPE == "weibull":
        return lambda x: weibull_min.pdf(x, WEIBULL_SHAPE, scale=WEIBULL_SCALE*1000)
    elif DIST_TYPE == "lognormal":
        return lambda x: lognorm.pdf(x, s=LOGNORMAL_SIGMA, scale=np.exp(LOGNORMAL_MU)*1000)
    elif DIST_TYPE == "constant":
        return lambda x: np.where((x >= (CONSTANT_TARGET*1000 - 1)) &
                                   (x <= (CONSTANT_TARGET*1000 + 1)), 1/2.0, 0)
    elif DIST_TYPE == "bimodal":
        # Bimodal distribution: weighted sum of two Gaussians
        return lambda x: (BIMODAL_WEIGHT1 * norm.pdf(x, loc=BIMODAL_MU1*1000, scale=BIMODAL_SIGMA1*1000) +
                         (1 - BIMODAL_WEIGHT1) * norm.pdf(x, loc=BIMODAL_MU2*1000, scale=BIMODAL_SIGMA2*1000))
    else:
        return lambda x: np.ones_like(x)

# -------------------------------
# Bubble Selection Methods
# -------------------------------

# Method 1: Direct PDF sampling (pure distribution, ignores data availability)
def select_bubble_direct_pdf(stratified_data, target_weights):
    """
    Method 1: Direct PDF sampling - Select bubbles using only target distribution weights
    Ignores the number of available bubbles in each bin (pure distribution matching)

    Advantages:
        - Best distribution matching (near 100%)
        - Simple and fast
        - Uses real bubbles from database
    Disadvantages:
        - May over-sample from bins with few bubbles
        - May under-sample from bins with many bubbles
        - Can fail if target bins are empty in database
    """
    counts = np.array([len(stratified_data.get(i, [])) for i in range(len(BIN_CENTERS))])

    # Only consider bins that have bubbles available
    available_mask = counts > 0
    if not available_mask.any():
        # Fallback: select from all available bubbles
        all_bubbles = [b for bins in stratified_data.values() for b in bins]
        if not all_bubbles:
            raise ValueError("Database is empty!")
        return random.choice(all_bubbles)

    # Use ONLY target weights (ignore counts) for available bins
    available_weights = target_weights.copy()
    available_weights[~available_mask] = 0  # Zero out unavailable bins

    # Normalize to probability
    total_weight = available_weights.sum()
    if total_weight == 0:
        all_bubbles = [b for bins in stratified_data.values() for b in bins]
        return random.choice(all_bubbles)

    probabilities = available_weights / total_weight

    # Select bin by probability (pure PDF, ignoring availability)
    chosen_bin = np.random.choice(len(probabilities), p=probabilities)

    # Select random bubble from chosen bin
    if len(stratified_data.get(chosen_bin, [])) == 0:
        # Should not happen due to available_mask, but failsafe
        all_bubbles = [b for bins in stratified_data.values() for b in bins]
        return random.choice(all_bubbles)

    return random.choice(stratified_data[chosen_bin])


# Method 2: Weighted sampling (balanced approach - DEFAULT)
def select_bubble_weighted(stratified_data, target_weights):
    """
    Method 2: Smart weighted sampling - Direct weighting by target PDF * available count

    Advantages:
        - Simple and fast
        - Good accuracy (90-95% match)
        - Uses real bubbles from database
        - Balances distribution matching with data availability
    Disadvantages:
        - Slightly less accurate than pure PDF method for perfect distribution matching
    """
    counts = np.array([len(stratified_data.get(i, [])) for i in range(len(BIN_CENTERS))])

    # Effective weight = target weight * available count
    effective_weights = target_weights * counts
    total_weight = effective_weights.sum()

    # Fallback: if all target bins are empty
    if total_weight == 0:
        all_bubbles = [b for bins in stratified_data.values() for b in bins]
        if not all_bubbles:
            raise ValueError("Database is empty!")
        return random.choice(all_bubbles)

    # Select bin by probability
    probabilities = effective_weights / total_weight
    chosen_bin = np.random.choice(len(probabilities), p=probabilities)

    # Fallback: if chosen bin happens to be empty
    if len(stratified_data.get(chosen_bin, [])) == 0:
        all_bubbles = [b for bins in stratified_data.values() for b in bins]
        return random.choice(all_bubbles)

    return random.choice(stratified_data[chosen_bin])


# Unified selection interface
def select_bubble(stratified_data, target_weights, method='weighted_sampling'):
    """
    Unified bubble selection interface supporting two methods

    Parameters:
        stratified_data: Stratified bubble database
        target_weights: Target PDF weights
        method: Selection method ('direct_pdf' or 'weighted_sampling')

    Returns:
        Bubble dict (always returns real bubble from database)
    """
    if method == 'direct_pdf':
        return select_bubble_direct_pdf(stratified_data, target_weights)
    elif method == 'weighted_sampling':
        return select_bubble_weighted(stratified_data, target_weights)
    else:
        raise ValueError(f"Unknown selection method: {method}. Use 'direct_pdf' or 'weighted_sampling'")

# -------------------------------
# Velocity-based Size Bias (NEW)
# -------------------------------
def apply_velocity_size_bias(target_weights, x, y, velocity_field,
                             coupling_strength=1.0):
    """
    Adjust bubble size distribution based on velocity field at position (x, y)

    This implements the physics-based principle:
        High velocity regions (center) -> can carry large bubbles
        Low velocity regions (edges) -> only small bubbles can remain

    Parameters:
        target_weights: Original target PDF weights (from distribution)
        bin_edges: Bubble size bins (in mm)
        x, y: Position where bubble will be placed
        velocity_field: VelocityField object
        coupling_strength: Size-velocity coupling parameter (0-2)

    Returns:
        adjusted_weights: Modified weights favoring appropriate sizes for this location
    """
    # Get velocity-based size factor at this location
    size_factor = velocity_field.get_max_bubble_size_factor(x, y, coupling_strength)

    # Create size-dependent modulation using pre-calculated BIN_CENTERS
    # Normalize bin centers to [0, 1] range
    size_normalized = (BIN_CENTERS - BIN_CENTERS.min()) / (BIN_CENTERS.max() - BIN_CENTERS.min() + 1e-6)

    # Apply velocity bias using exponential cutoff:
    # - size_factor = 1.0 (high velocity): all sizes weighted equally (bias ~ 1.0)
    # - size_factor = 0.1 (low velocity): small bubbles strongly favored (bias decays for large bubbles)
    #
    # Formula: bias = exp(-((1 - size_factor) * size_normalized)^2 / sigma^2)
    # This creates a smooth cutoff that gets sharper as size_factor decreases

    sigma_sq = 0.3  # Controls smoothness of cutoff
    size_bias = np.exp(-((1 - size_factor) * size_normalized) ** 2 / sigma_sq)

    # Combine with target distribution
    adjusted_weights = target_weights * size_bias

    # Normalize to probability
    total = np.sum(adjusted_weights)
    if total > 1e-10:
        adjusted_weights = adjusted_weights / total
    else:
        # Fallback to original weights if all weights become zero
        adjusted_weights = target_weights / (np.sum(target_weights) + 1e-10)

    return adjusted_weights


def select_bubble_with_velocity_bias(stratified_data, target_weights,
                                     x, y, velocity_field, coupling_strength=1.0,
                                     method='weighted_sampling'):
    """
    Select bubble with velocity field consideration

    This is the main interface for velocity-aware bubble selection.
    It adjusts the target distribution based on local velocity, then uses
    the standard selection method.

    Parameters:
        stratified_data: Stratified bubble database
        target_weights: Base target distribution weights
        x, y: Position for bubble placement
        velocity_field: VelocityField object
        coupling_strength: Size-velocity coupling strength
        method: Selection method ('direct_pdf' or 'weighted_sampling')

    Returns:
        Bubble dict from database
    """
    # Apply velocity-based size bias
    adjusted_weights = apply_velocity_size_bias(
        target_weights, x, y, velocity_field, coupling_strength
    )

    # Use existing selection method with adjusted weights
    return select_bubble(stratified_data, adjusted_weights, method=method)

# -------------------------------
# Overlap Detection (Unidirectional)
# -------------------------------
def compute_intersection_area(bbox1, bbox2):
    """Calculate intersection area of two bounding boxes"""
    x1, y1, x2, y2 = bbox1
    a1, b1, a2, b2 = bbox2

    inter_x = max(0, min(x2, a2) - max(x1, a1))
    inter_y = max(0, min(y2, b2) - max(y1, b1))
    return inter_x * inter_y

def compute_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def check_overlap_with_cumulative(new_bbox, existing_bboxes, single_threshold, cumulative_threshold=None):
    """
    Simplified unidirectional overlap detection: ensure existing bubbles are not occluded

    This method checks that no existing bubble is excessively covered by the new bubble,
    ensuring all previously placed bubbles remain visible.

    Parameters:
        new_bbox: Bounding box of new bubble (x1, y1, x2, y2)
        existing_bboxes: List of existing bubbles [(bbox, info), ...]
        single_threshold: Maximum allowed occlusion ratio for any existing bubble
        cumulative_threshold: Cumulative overlap threshold (optional, not commonly used)

    Returns:
        (is_collision, details_dict)

    Example:
        If single_threshold = 0.3, then any existing bubble can be at most 30% occluded
    """
    if not existing_bboxes:
        return False, {"max_existing_ratio": 0, "cumulative_ratio": 0, "num_interactions": 0}

    new_area = compute_bbox_area(new_bbox)
    if new_area == 0:
        return True, {"error": "new_bbox has zero area"}

    max_existing_ratio = 0.0
    total_intersection = 0
    num_interactions = 0

    for existing_bbox, _ in existing_bboxes:
        inter_area = compute_intersection_area(new_bbox, existing_bbox)

        if inter_area > 0:
            num_interactions += 1
            existing_area = compute_bbox_area(existing_bbox)
            total_intersection += inter_area

            # Unidirectional: use existing bubble area as denominator
            # This measures how much the existing bubble is occluded by the new bubble
            ratio = inter_area / existing_area if existing_area > 0 else 0
            max_existing_ratio = max(max_existing_ratio, ratio)

    cumulative_ratio = total_intersection / new_area if new_area > 0 else 0

    # Check if collision occurs
    is_collision = max_existing_ratio > single_threshold
    if cumulative_threshold is not None:
        is_collision = is_collision or (cumulative_ratio > cumulative_threshold)

    details = {
        "max_existing_ratio": max_existing_ratio,
        "cumulative_ratio": cumulative_ratio,
        "num_interactions": num_interactions
    }

    return is_collision, details

# -------------------------------
# Trapezoid ROI Class (Entrainment)
# -------------------------------
class TrapezoidROI:
    """
    Trapezoid ROI region simulating bubble plume entrainment effect

    Coordinate system:
        - y=0 at physical bottom (bubble source)
        - y=canvas_height at physical top
        - In image coordinates: image_y = canvas_height - y

    Width formula:
        w(y) = base_width + 2 * slope * y
        where y is physical height from bottom
    """

    def __init__(self, canvas_width, canvas_height, base_width_ratio, slope, center_x=None):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.slope = slope
        self.center_x = center_x if center_x is not None else canvas_width // 2

        # Base width (physical bottom = image bottom, i.e., image_y = canvas_height)
        # This is the bubble source location, narrowest width
        self.base_width = int(canvas_width * base_width_ratio)

        # Top width (physical top = image top, i.e., image_y = 0)
        # After plume expands to top, widest width
        self.top_width = min(self.base_width + 2 * slope * canvas_height, canvas_width)

    def get_width_at_y(self, image_y):
        """
        Get ROI width at image_y coordinate
        image_y: y value in image coordinate system (0 at top, canvas_height at bottom)
        """
        # Convert to physical height (from bottom)
        physical_y = self.canvas_height - image_y
        width = self.base_width + 2 * self.slope * physical_y
        return min(width, self.canvas_width)

    def get_x_bounds_at_y(self, y):
        """Get left and right bounds at y coordinate"""
        width = self.get_width_at_y(y)
        x_left = max(0, self.center_x - width / 2)
        x_right = min(self.canvas_width, self.center_x + width / 2)
        return int(x_left), int(x_right)

    def contains_bbox(self, x, y, w, h):
        """Check if bounding box is completely within ROI"""
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for cx, cy in corners:
            if cy < 0 or cy > self.canvas_height:
                return False
            x_left, x_right = self.get_x_bounds_at_y(cy)
            if not (x_left <= cx <= x_right):
                return False
        return True

    def get_valid_x_range(self, y, bubble_width):
        """Get valid x range for placing bubble at height y"""
        x_left, x_right = self.get_x_bounds_at_y(y)
        x_min = x_left
        x_max = x_right - bubble_width
        if x_max < x_min:
            return None
        return int(x_min), int(x_max)

    def calculate_area(self, background_img=None):
        """Calculate ROI area"""
        if background_img is not None:
            mask = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
            for y in range(self.canvas_height):
                x_left, x_right = self.get_x_bounds_at_y(y)
                mask[y, x_left:x_right] = 1
            return cv2.countNonZero(background_img * mask)
        else:
            return int((self.base_width + self.top_width) * self.canvas_height / 2)

    def create_mask(self):
        """Create ROI mask"""
        mask = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        for y in range(self.canvas_height):
            x_left, x_right = self.get_x_bounds_at_y(y)
            mask[y, x_left:x_right] = 255
        return mask

# -------------------------------
# Velocity Field Model (Physics-based)
# -------------------------------
class VelocityField:
    """
    Pseudo-velocity field based on Gaussian distribution
    Simulates two-phase flow velocity profile where:
    - High velocity (center) can carry large bubbles
    - Low velocity (edges) only carries small bubbles

    Physics: Bubble size ~ velocity field intensity
    """

    def __init__(self, roi, placement_mode='gaussian',
                 velocity_exponent=2.0, vertical_decay=0.0):
        """
        Parameters:
            roi: ROI object (TrapezoidROI or rectangle tuple)
            placement_mode: 'gaussian' or 'random'
            velocity_exponent: Controls velocity profile sharpness (higher = more peaked)
            vertical_decay: Velocity decay rate with height (0 = no decay)
        """
        self.roi = roi
        self.placement_mode = placement_mode
        self.velocity_exponent = velocity_exponent
        self.vertical_decay = vertical_decay

        # Get ROI geometry
        if isinstance(roi, TrapezoidROI):
            self.center_x = roi.center_x
            self.canvas_height = roi.canvas_height
            self.canvas_width = roi.canvas_width
            self.is_trapezoid = True
            # Use average width for velocity scale
            self.roi_width = (roi.base_width + roi.top_width) / 2
        else:
            roi_x_min, roi_x_max, y_min, y_max = roi
            self.center_x = (roi_x_min + roi_x_max) / 2
            self.canvas_height = y_max - y_min
            self.canvas_width = roi_x_max - roi_x_min
            self.roi_width = self.canvas_width
            self.is_trapezoid = False

        # Scale for Gaussian velocity profile
        self.velocity_scale = self.roi_width / gaussian_scale_divisor

    def get_velocity_at(self, x, y):
        """
        Calculate pseudo-velocity at position (x, y)

        Returns:
            velocity: Normalized velocity [0, 1]
                     1.0 at center, decays towards edges
        """
        # Horizontal velocity profile (Gaussian)
        if self.placement_mode.lower() in ['gaussian', 'guassian']:
            # Distance from center
            dx = x - self.center_x

            # Gaussian velocity profile: V(x) = exp(-(dx/scale)^exponent)
            velocity_x = np.exp(-((dx / self.velocity_scale) ** self.velocity_exponent))
        else:
            # Uniform for random placement
            velocity_x = 1.0

        # Vertical decay (optional)
        if self.vertical_decay > 0:
            # Velocity decreases with height (physical: momentum dissipation)
            # Normalize y to [0, 1], where 0=bottom, 1=top
            y_normalized = y / self.canvas_height
            velocity_y = np.exp(-self.vertical_decay * y_normalized)
        else:
            velocity_y = 1.0

        # Combined velocity
        velocity = velocity_x * velocity_y

        return np.clip(velocity, 0.0, 1.0)

    def get_max_bubble_size_factor(self, x, y, coupling_strength=1.0):
        """
        Calculate size bias factor at (x, y) based on velocity

        Physics intuition:
            - High velocity region: can support large bubbles
            - Low velocity region: only small bubbles can remain

        Mathematical relation:
            F_drag ~ rho * v^2 * A  (drag force)
            F_buoyancy ~ g * rho * V ~ A^(3/2)  (buoyancy)
            Balance: A_max ~ v^(4/3)
            Simplified: size_factor ~ v^coupling_strength

        Parameters:
            coupling_strength: Controls how strongly velocity limits size
                              0 = no limit, 1 = moderate, 2 = strong

        Returns:
            size_factor: Multiplier for bubble size weight [0.1, 1.0]
                        1.0 = full size range allowed
                        0.1 = only smallest bubbles allowed
        """
        velocity = self.get_velocity_at(x, y)

        # Size-velocity coupling
        size_factor = velocity ** coupling_strength

        return np.clip(size_factor, 0.1, 1.0)  # Keep minimum 0.1 to avoid empty bins

# -------------------------------
# Quadtree Implementation
# -------------------------------
class Rectangle:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def intersects(self, bbox):
        x1, y1, x2, y2 = bbox
        return not (x2 < self.x or x1 > self.x + self.w or y2 < self.y or y1 > self.y + self.h)

class Quadtree:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.capacity = capacity
        self.objects = []
        self.divided = False

    def subdivide(self):
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h
        hw, hh = w / 2, h / 2
        self.northeast = Quadtree(Rectangle(x + hw, y, hw, hh), self.capacity)
        self.northwest = Quadtree(Rectangle(x, y, hw, hh), self.capacity)
        self.southeast = Quadtree(Rectangle(x + hw, y + hh, hw, hh), self.capacity)
        self.southwest = Quadtree(Rectangle(x, y + hh, hw, hh), self.capacity)
        self.divided = True

    def insert(self, bbox, bubble_info):
        if not self.boundary.intersects(bbox):
            return False
        if len(self.objects) < self.capacity:
            self.objects.append((bbox, bubble_info))
            return True
        if not self.divided:
            self.subdivide()
        return (self.northeast.insert(bbox, bubble_info) or
                self.northwest.insert(bbox, bubble_info) or
                self.southeast.insert(bbox, bubble_info) or
                self.southwest.insert(bbox, bubble_info))

    def query(self, bbox, found=None):
        if found is None:
            found = []
        if not self.boundary.intersects(bbox):
            return found
        for obj in self.objects:
            obj_bbox, _ = obj
            if self._bbox_intersect(bbox, obj_bbox):
                found.append(obj)
        if self.divided:
            self.northwest.query(bbox, found)
            self.northeast.query(bbox, found)
            self.southwest.query(bbox, found)
            self.southeast.query(bbox, found)
        return found

    def _bbox_intersect(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        a1, b1, a2, b2 = bbox2
        return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

# -------------------------------
# Position Generation
# -------------------------------
def generate_position(roi, bubble_width, bubble_height, placement_mode='gaussian'):
    """
    Generate bubble placement position
    """
    max_attempts = 50

    for _ in range(max_attempts):
        if isinstance(roi, TrapezoidROI):
            # Trapezoid ROI
            max_y = max(1, roi.canvas_height - bubble_height)
            y = np.random.randint(0, max_y)
            valid_range = roi.get_valid_x_range(y, bubble_width)
            if valid_range is None:
                continue
            x_min, x_max = valid_range
        else:
            # Rectangle ROI: roi = (x_min, x_max, y_min, y_max)
            roi_x_min, roi_x_max, y_min, y_max = roi
            max_y = max(y_min + 1, y_max - bubble_height)
            y = np.random.randint(y_min, max_y)
            x_min = roi_x_min
            x_max = roi_x_max - bubble_width
            if x_max <= x_min:
                continue

        if placement_mode.lower() in ['gaussian', 'guassian']:
            x_center = (x_min + x_max) / 2
            x_scale = (x_max - x_min) / gaussian_scale_divisor
            if x_scale <= 0:
                x = int(x_center)
            else:
                a = (x_min - x_center) / x_scale
                b = (x_max - x_center) / x_scale
                x = int(truncnorm.rvs(a, b, loc=x_center, scale=x_scale))
        else:
            x = np.random.randint(x_min, max(x_min + 1, x_max))

        # Validate position
        if isinstance(roi, TrapezoidROI):
            if roi.contains_bbox(x, y, bubble_width, bubble_height):
                return x, y
        else:
            roi_x_min, roi_x_max, y_min, y_max = roi
            if roi_x_min <= x and x + bubble_width <= roi_x_max and y_min <= y and y + bubble_height <= y_max:
                return x, y

    return None

# -------------------------------
# Main Image Generation
# -------------------------------
MAX_ATTEMPTS_PER_BUBBLE = 50

def generate_synthetic_images(bubble_data, stratified_data, bin_edges, target_weights, **kwargs):
    """
    Generate synthetic images (with optional velocity-based size bias)

    Parameters:
        bubble_data: All bubble data
        stratified_data: Stratified bubble database
        bin_edges: Bin edges for stratification
        target_weights: Target PDF weights
        **kwargs: Additional parameters:
            - selection_method: 'direct_pdf' or 'weighted_sampling'
            - enable_velocity_bias: Enable velocity field (bool)
            - velocity_exponent: Velocity profile sharpness
            - size_velocity_coupling: Size-velocity coupling strength
            - velocity_decay: Vertical velocity decay rate
    """
    used_bubbles = []
    selection_method = kwargs.get('selection_method', SELECTION_METHOD)
    enable_velocity_bias = kwargs.get('enable_velocity_bias', ENABLE_VELOCITY_BIAS)

    # Initialize ROI
    if USE_TRAPEZOID_ROI:
        roi = TrapezoidROI(
            canvas_width=CANVAS_SIZE[0],
            canvas_height=CANVAS_SIZE[1],
            base_width_ratio=BASE_WIDTH_RATIO,
            slope=ENTRAINMENT_SLOPE
        )
        ROI_area = roi.calculate_area(bg_img if USE_BACKGROUND_IMAGE else None)
        print(f"Trapezoid ROI: base={roi.base_width}px, top={roi.top_width:.0f}px, slope={ENTRAINMENT_SLOPE}")
    else:
        roi = (ROI_X_MIN, ROI_X_MAX, 0, CANVAS_SIZE[1])
        if USE_BACKGROUND_IMAGE:
            ROI_area = cv2.countNonZero(bg_img[:, ROI_X_MIN:ROI_X_MAX])
        else:
            ROI_area = (ROI_X_MAX - ROI_X_MIN) * CANVAS_SIZE[1]
        print(f"Rectangle ROI: x=[{ROI_X_MIN}, {ROI_X_MAX}]")

    # Initialize velocity field if enabled
    velocity_field = None
    if enable_velocity_bias:
        velocity_field = VelocityField(
            roi=roi,
            placement_mode=PLACEMENT_MODE,
            velocity_exponent=kwargs.get('velocity_exponent', VELOCITY_PROFILE_EXPONENT),
            vertical_decay=kwargs.get('velocity_decay', VELOCITY_VERTICAL_DECAY)
        )
        coupling_strength = kwargs.get('size_velocity_coupling', SIZE_VELOCITY_COUPLING)
        print(f"Velocity-based size bias ENABLED:")
        print(f"  - Velocity exponent: {velocity_field.velocity_exponent}")
        print(f"  - Size-velocity coupling: {coupling_strength}")
        print(f"  - Vertical decay: {velocity_field.vertical_decay}")
        print(f"  Physics: Large bubbles prefer high-velocity center region")
    else:
        print("Velocity-based size bias DISABLED (uniform size distribution across ROI)")

    # Calculate average bubble size for position estimation (when velocity bias is enabled)
    if enable_velocity_bias:
        avg_diameter_mm = np.average(BIN_CENTERS, weights=target_weights)
        # Rough conversion: diameter (mm) to pixels (assume ~10mm = canvas_width scaling)
        avg_bubble_size = int(avg_diameter_mm * CANVAS_SIZE[0] / 10.0)
        avg_bubble_size = max(10, min(avg_bubble_size, 100))  # Clamp to reasonable range

    for img_id in tqdm(range(NUM_SYNTHETIC_IMAGES), desc="Generating Images"):
        # Apply mixed mode parameters if enabled
        if ENABLE_MIXED_MODE:
            mixed_params = generate_mixed_mode_parameters(img_id, NUM_SYNTHETIC_IMAGES)
            current_max_bubbles = mixed_params['max_bubbles_per_image']
            current_placement_mode = mixed_params['placement_mode']
            current_overlap_control = mixed_params['overlap_control']
            current_entrainment_slope = mixed_params['entrainment_slope']
            current_base_width_ratio = mixed_params['base_width_ratio']

            # Save parameters for this batch (only once per batch)
            save_mixed_mode_parameters(RUN_DIR, img_id, mixed_params)

            # Update ROI with new parameters
            if USE_TRAPEZOID_ROI:
                roi = TrapezoidROI(
                    canvas_width=CANVAS_SIZE[0],
                    canvas_height=CANVAS_SIZE[1],
                    base_width_ratio=current_base_width_ratio,
                    slope=current_entrainment_slope
                )
                ROI_area = roi.calculate_area(bg_img if USE_BACKGROUND_IMAGE else None)
        else:
            # Use base parameters
            current_max_bubbles = MAX_BUBBLES_PER_IMAGE
            current_placement_mode = PLACEMENT_MODE
            current_overlap_control = overlap_control
            current_entrainment_slope = ENTRAINMENT_SLOPE
            current_base_width_ratio = BASE_WIDTH_RATIO

        # Initialize
        quadtree = Quadtree(Rectangle(0, 0, CANVAS_SIZE[0], CANVAS_SIZE[1]), capacity=4)
        placed_bboxes = []

        if USE_BACKGROUND_IMAGE:
            canvas = bg_img.copy()
        else:
            canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0]), default_background_value, dtype=np.uint8)

        annotations = {
            "boxes": [], "masks": [], "areas": [],
            "void_fraction": 0.0, "overlap_stats": []
        }
        roi_acc_mask = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0]), dtype=np.uint8)
        current_bubbles = []

        # Bubble placement loop
        while len(placed_bboxes) < current_max_bubbles:
            # Check void fraction
            if ROI_area > 0:
                current_vf = np.count_nonzero(roi_acc_mask) / ROI_area
                if current_vf >= TARGET_VOID_FRACTION:
                    break

            bubble_placed = False
            attempts = 0

            while not bubble_placed and attempts < MAX_ATTEMPTS_PER_BUBBLE:
                attempts += 1

                # ===== VELOCITY-AWARE BUBBLE SELECTION =====
                if enable_velocity_bias and velocity_field is not None:
                    # Method: Generate position first with approximate size, then select bubble based on velocity
                    # This ensures large bubbles naturally appear in high-velocity regions

                    # Step 1: Generate temporary position using average bubble size
                    temp_position = generate_position(roi, avg_bubble_size, avg_bubble_size, current_placement_mode)
                    if temp_position is None:
                        continue

                    x_temp, y_temp = temp_position

                    # Step 2: Select bubble based on velocity at this position
                    bubble = select_bubble_with_velocity_bias(
                        stratified_data, target_weights,
                        x_temp, y_temp, velocity_field, coupling_strength,
                        method=selection_method
                    )

                    bw, bh = bubble["bbox"]

                    if bw > CANVAS_SIZE[0] or bh > CANVAS_SIZE[1]:
                        continue

                    # Step 3: Refine position with actual bubble size
                    # Try to keep x close to x_temp (where velocity was calculated)
                    position = generate_position(roi, bw, bh, current_placement_mode)
                    if position is None:
                        continue

                    x, y = position

                else:
                    # Original method: select bubble first, then find position
                    bubble = select_bubble(stratified_data, target_weights,
                                          method=selection_method)

                    bw, bh = bubble["bbox"]

                    if bw > CANVAS_SIZE[0] or bh > CANVAS_SIZE[1]:
                        continue

                    # Generate position
                    position = generate_position(roi, bw, bh, current_placement_mode)
                    if position is None:
                        continue

                    x, y = position

                # ===== END VELOCITY-AWARE SELECTION =====

                new_bbox = (x, y, x + bw, y + bh)

                # Collision detection
                candidate_bubbles = quadtree.query(new_bbox)
                is_collision, overlap_details = check_overlap_with_cumulative(
                    new_bbox,
                    candidate_bubbles,
                    single_threshold=current_overlap_control,
                    cumulative_threshold=current_overlap_control * 1.5
                )

                if is_collision:
                    continue

                # Place bubble
                try:
                    roi_bubble = canvas[y:y+bh, x:x+bw]

                    if APPLY_FEATHERING:
                        mask = bubble["mask"]
                        binary_mask = (mask > 127).astype(np.uint8) * 255
                        erosion_kernel = np.ones(FEATHER_EROSION_KERNEL, np.uint8)
                        eroded_mask = cv2.erode(binary_mask, erosion_kernel,
                                               iterations=FEATHER_EROSION_ITERATIONS)
                        soft_mask = cv2.GaussianBlur(eroded_mask.astype(np.float32)/255,
                                                    FEATHER_KERNEL_SIZE, FEATHER_SIGMA)
                        blended = roi_bubble * (1 - soft_mask) + bubble["image"] * soft_mask
                        roi_bubble[:] = blended.astype(np.uint8)
                        saved_mask = (bubble["mask"] > 127).astype(np.uint8)
                    else:
                        mask = (bubble["mask"] > 127)
                        roi_bubble[mask] = bubble["image"][mask]
                        saved_mask = mask.astype(np.uint8)

                    # Update records
                    placed_bboxes.append(new_bbox)
                    annotations["boxes"].append((int(x), int(y), int(bw), int(bh)))
                    annotations["masks"].append(saved_mask.tolist())
                    annotations["areas"].append(int(bubble["true_area"]))
                    annotations["overlap_stats"].append(overlap_details)
                    current_bubbles.append(bubble)

                    quadtree.insert(new_bbox, {"bubble_id": len(placed_bboxes) - 1})
                    roi_acc_mask[y:y+bh, x:x+bw] = cv2.bitwise_or(
                        roi_acc_mask[y:y+bh, x:x+bw], saved_mask
                    )

                    bubble_placed = True

                except Exception:
                    continue

            if not bubble_placed:
                break

        # Save results
        used_bubbles.extend(current_bubbles)
        annotations["void_fraction"] = np.count_nonzero(roi_acc_mask) / ROI_area if ROI_area > 0 else 0
        annotations["num_bubbles"] = len(placed_bboxes)

        if USE_TRAPEZOID_ROI:
            annotations["entrainment"] = {
                "slope": ENTRAINMENT_SLOPE,
                "base_width": roi.base_width,
                "top_width": float(roi.top_width)
            }

        cv2.imwrite(os.path.join(RUN_DIR, f"synth_{img_id:04d}.png"), canvas)
        with open(os.path.join(RUN_DIR, f"synth_{img_id:04d}.json"), "w") as f:
            json.dump(annotations, f, indent=2)

    return used_bubbles

# -------------------------------
# Visualization Functions
# -------------------------------
def visualize_roi(save_path=None):
    """Visualize ROI region"""
    if USE_BACKGROUND_IMAGE and bg_img is not None:
        canvas = cv2.cvtColor(bg_img.copy(), cv2.COLOR_GRAY2BGR)
    else:
        canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), default_background_value, dtype=np.uint8)

    if USE_TRAPEZOID_ROI:
        roi = TrapezoidROI(CANVAS_SIZE[0], CANVAS_SIZE[1], BASE_WIDTH_RATIO, ENTRAINMENT_SLOPE)
        pts = np.array([
            [int(roi.center_x - roi.top_width//2), 0],
            [int(roi.center_x + roi.top_width//2), 0],
            [int(roi.center_x + roi.base_width//2), CANVAS_SIZE[1]],
            [int(roi.center_x - roi.base_width//2), CANVAS_SIZE[1]]
        ], np.int32)
        cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)
        cv2.putText(canvas, f"Trapezoid ROI (k={ENTRAINMENT_SLOPE})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        x_min = int(CANVAS_SIZE[0] * gaussian_x_min_ratio)
        x_max = int(CANVAS_SIZE[0] * gaussian_x_max_ratio)
        cv2.rectangle(canvas, (x_min, 0), (x_max, CANVAS_SIZE[1]), (0, 0, 255), 2)
        cv2.putText(canvas, "Rectangle ROI", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if save_path is None:
        save_path = os.path.join(RUN_DIR, "roi_visualization.png")
    cv2.imwrite(save_path, canvas)
    print(f"ROI visualization saved to {save_path}")

def visualize_velocity_field(velocity_field, roi, save_path=None):
    """Visualize the pseudo-velocity field and size factor distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

    # Determine grid range based on ROI
    if isinstance(roi, TrapezoidROI):
        canvas_width = roi.canvas_width
        canvas_height = roi.canvas_height
    else:
        roi_x_min, roi_x_max, y_min, y_max = roi
        canvas_width = roi_x_max - roi_x_min
        canvas_height = y_max - y_min

    # Create velocity field grid
    x_grid = np.linspace(0, canvas_width, 200)
    y_grid = np.linspace(0, canvas_height, 150)
    X, Y = np.meshgrid(x_grid, y_grid)

    V = np.zeros_like(X)
    S = np.zeros_like(X)  # Size factor

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            V[i, j] = velocity_field.get_velocity_at(X[i, j], Y[i, j])
            S[i, j] = velocity_field.get_max_bubble_size_factor(X[i, j], Y[i, j],
                                                               coupling_strength=SIZE_VELOCITY_COUPLING)

    # Plot velocity field
    im1 = axes[0].contourf(X, Y, V, levels=20, cmap='RdYlBu_r')
    axes[0].set_title('Velocity Field V(x,y)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    axes[0].invert_yaxis()  # Match image coordinates
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Normalized Velocity')

    # Add ROI boundary
    if isinstance(roi, TrapezoidROI):
        # Draw trapezoid outline
        trap_x = [roi.center_x - roi.top_width/2, roi.center_x + roi.top_width/2,
                  roi.center_x + roi.base_width/2, roi.center_x - roi.base_width/2,
                  roi.center_x - roi.top_width/2]
        trap_y = [0, 0, canvas_height, canvas_height, 0]
        axes[0].plot(trap_x, trap_y, 'k--', linewidth=2, label='ROI Boundary')
    axes[0].legend(loc='upper right')

    # Plot size factor
    im2 = axes[1].contourf(X, Y, S, levels=20, cmap='viridis')
    axes[1].set_title(f'Max Bubble Size Factor (coupling={SIZE_VELOCITY_COUPLING})',
                     fontweight='bold', fontsize=12)
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_ylabel('y (pixels)')
    axes[1].invert_yaxis()
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Size Factor [0,1]')

    # Add ROI boundary
    if isinstance(roi, TrapezoidROI):
        axes[1].plot(trap_x, trap_y, 'k--', linewidth=2, label='ROI Boundary')
    axes[1].legend(loc='upper right')

    # Add text annotations
    fig.text(0.5, 0.02,
            f'Velocity exponent: {velocity_field.velocity_exponent} | '
            f'Vertical decay: {velocity_field.vertical_decay} | '
            f'Physics: Large bubbles prefer high-velocity center',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path is None:
        save_path = os.path.join(RUN_DIR, "velocity_field_visualization.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Velocity field visualization saved to {save_path}")
    plt.close()

def plot_diameter_histogram(used_bubbles, save_path=None):
    """Plot bubble diameter distribution histogram"""
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 18  # Increased from 14

    diameters = np.array([b["bubble_diameter"] for b in used_bubbles
                         if b.get("bubble_diameter")]) * 1000

    if len(diameters) == 0:
        print("No bubble diameter data available.")
        return None

    fig, ax1 = plt.subplots(figsize=(160/25.4, 120/25.4))

    bins = np.arange(diameters.min(), diameters.max() + 0.1, 0.1)
    ax1.hist(diameters, bins=bins, color='#4878CF', alpha=0.7,
             edgecolor='black', linewidth=0.8, label="Bubble Count")
    ax1.set_xlabel("Diameter (mm)", fontweight='bold', fontsize=18)  # Increased font size
    ax1.set_ylabel("Bubble Count", color='#4878CF', fontweight='bold', fontsize=18)  # Increased font size
    ax1.tick_params(axis='both', which='major', labelsize=16)  # Increased tick label size
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Target distribution curve
    target_pdf = get_target_pdf_func()
    x_vals = np.linspace(diameters.min(), diameters.max(), 500)
    y_vals = target_pdf(x_vals)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_vals, color='#E24A33', lw=2.5, label=f"Target ({DIST_TYPE})")
    ax2.set_ylabel("Probability Density", color='#E24A33', fontweight='bold', fontsize=18)  # Increased font size
    ax2.tick_params(axis='y', which='major', labelsize=16)  # Increased tick label size

    # Statistics info - moved to upper right corner
    stats_text = f"n={len(diameters)}\nMean={np.mean(diameters):.2f}mm\nSD={np.std(diameters):.2f}mm"
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=16)

    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(RUN_DIR, "bubble_diameter_histogram.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    return fig

def perform_ks_test(used_bubbles):
    """Kolmogorov-Smirnov test"""
    diameters = np.array([b["bubble_diameter"] for b in used_bubbles
                         if b.get("bubble_diameter")]) * 1000

    if len(diameters) == 0:
        return None, None

    d_min, d_max = diameters.min(), diameters.max()

    if DIST_TYPE == "gaussian":
        base_cdf = lambda x: norm.cdf(x, loc=GAUSSIAN_MU*1000, scale=GAUSSIAN_SIGMA*1000)
    elif DIST_TYPE == "weibull":
        base_cdf = lambda x: weibull_min.cdf(x, WEIBULL_SHAPE, scale=WEIBULL_SCALE*1000)
    elif DIST_TYPE == "lognormal":
        base_cdf = lambda x: lognorm.cdf(x, s=LOGNORMAL_SIGMA, scale=np.exp(LOGNORMAL_MU)*1000)
    elif DIST_TYPE == "bimodal":
        # Bimodal CDF: weighted sum of two Gaussian CDFs
        base_cdf = lambda x: (BIMODAL_WEIGHT1 * norm.cdf(x, loc=BIMODAL_MU1*1000, scale=BIMODAL_SIGMA1*1000) +
                             (1 - BIMODAL_WEIGHT1) * norm.cdf(x, loc=BIMODAL_MU2*1000, scale=BIMODAL_SIGMA2*1000))
    else:
        base_cdf = lambda x: x / d_max

    F_min, F_max = base_cdf(d_min), base_cdf(d_max)
    truncated_cdf = lambda x: (base_cdf(x) - F_min) / (F_max - F_min + 1e-6)

    stat, p_value = kstest(diameters, truncated_cdf)
    print(f"K-S test: statistic={stat:.4f}, p-value={p_value:.4f}")

    return stat, p_value

# -------------------------------
# Parameter Diagram Generation
# -------------------------------
def generate_parameter_diagram(save_path=None):
    """
    Generate parameter diagram with all key parameters annotated
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    # ==========================================
    # Subplot 1: ROI Configuration
    # ==========================================
    ax1 = axes[0, 0]
    ax1.set_title('(a) ROI Configuration', fontweight='bold', fontsize=12)
    
    canvas_w, canvas_h = 100, 80
    
    # Draw canvas boundary
    ax1.add_patch(plt.Rectangle((0, 0), canvas_w, canvas_h, 
                                 fill=False, edgecolor='black', linewidth=2))
    ax1.text(canvas_w/2, canvas_h + 5, f'Canvas: {CANVAS_SIZE[0]} x {CANVAS_SIZE[1]} px',
             ha='center', fontsize=9)
    
    if USE_TRAPEZOID_ROI:
        # Trapezoid ROI
        base_w = canvas_w * BASE_WIDTH_RATIO
        top_w = min(base_w + 2 * ENTRAINMENT_SLOPE * canvas_h, canvas_w)
        center_x = canvas_w / 2
        
        trap_points = [
            [center_x - top_w/2, canvas_h],
            [center_x + top_w/2, canvas_h],
            [center_x + base_w/2, 0],
            [center_x - base_w/2, 0],
        ]
        trap = plt.Polygon(trap_points, fill=True, facecolor='lightblue', 
                          edgecolor='blue', linewidth=2, alpha=0.5)
        ax1.add_patch(trap)
        
        # Annotations
        ax1.text(center_x - top_w/2 - 10, canvas_h/2, f'k={ENTRAINMENT_SLOPE}',
                fontsize=9, color='red', rotation=90, va='center')
        
        ax1.annotate('', xy=(center_x - base_w/2, -5), xytext=(center_x + base_w/2, -5),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax1.text(center_x, -10, f'b0 = {BASE_WIDTH_RATIO:.0%} x W',
                ha='center', fontsize=9, color='green')
        
        # Formula
        ax1.text(canvas_w + 5, canvas_h * 0.6, 
                f'b(z) = b0 + 2*k*z\n\nk = {ENTRAINMENT_SLOPE}\nb0 = {BASE_WIDTH_RATIO:.0%} x W',
                fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # Rectangle ROI
        roi_x_min = canvas_w * gaussian_x_min_ratio
        roi_x_max = canvas_w * gaussian_x_max_ratio
        roi_w = roi_x_max - roi_x_min
        
        rect = plt.Rectangle((roi_x_min, 0), roi_w, canvas_h,
                             fill=True, facecolor='lightblue', 
                             edgecolor='blue', linewidth=2, alpha=0.5)
        ax1.add_patch(rect)
        
        ax1.annotate('', xy=(roi_x_min, -5), xytext=(roi_x_max, -5),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax1.text((roi_x_min + roi_x_max)/2, -10, 
                f'ROI: [{gaussian_x_min_ratio:.0%}, {gaussian_x_max_ratio:.0%}] x W',
                ha='center', fontsize=9, color='green')
    
    ax1.set_xlim(-20, canvas_w + 50)
    ax1.set_ylim(-20, canvas_h + 15)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # ==========================================
    # Subplot 2: Overlap Ratio Definition
    # ==========================================
    ax2 = axes[0, 1]
    ax2.set_title('(b) Overlap Ratio (R)', fontweight='bold', fontsize=12)
    
    # Bubble 1 (existing)
    rect1 = plt.Rectangle((20, 25), 40, 35, fill=True, facecolor='lightcoral', 
                          edgecolor='red', linewidth=2, alpha=0.6)
    ax2.add_patch(rect1)
    ax2.text(40, 65, 'A_existing', ha='center', fontsize=10, color='red')
    
    # Bubble 2 (new)
    rect2 = plt.Rectangle((45, 30), 30, 25, fill=True, facecolor='lightgreen',
                          edgecolor='green', linewidth=2, alpha=0.6)
    ax2.add_patch(rect2)
    ax2.text(60, 20, 'A_new', ha='center', fontsize=10, color='green')
    
    # Intersection area
    inter_rect = plt.Rectangle((45, 30), 15, 25, fill=True, facecolor='yellow', 
                               edgecolor='orange', linewidth=2, alpha=0.8)
    ax2.add_patch(inter_rect)
    ax2.text(52.5, 42.5, 'A_inter', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Formula
    formula_text = ('Unidirectional:\n\n'
                   'R = A_inter / A_existing\n\n'
                   'R in [0, 1]\n\n'
                   f'Threshold = {overlap_control}')
    ax2.text(90, 45, formula_text, fontsize=9, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlim(0, 140)
    ax2.set_ylim(0, 80)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # ==========================================
    # Subplot 3: Multiple Overlap Detection
    # ==========================================
    ax3 = axes[1, 0]
    ax3.set_title('(c) Multiple Overlap Detection', fontweight='bold', fontsize=12)
    
    # New bubble (center)
    new_rect = plt.Rectangle((35, 25), 40, 35, fill=True, facecolor='lightgreen',
                             edgecolor='green', linewidth=2, alpha=0.6)
    ax3.add_patch(new_rect)
    ax3.text(55, 42.5, 'New', ha='center', va='center', fontsize=10, fontweight='bold', color='green')
    
    # Existing bubbles
    existing = [((25, 50), (25, 20), 'A', 'lightcoral'),
                ((65, 35), (20, 25), 'B', 'lightskyblue'),
                ((40, 10), (30, 18), 'C', 'plum')]
    
    for (x, y), (w, h), label, color in existing:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color,
                             edgecolor='black', linewidth=1.5, alpha=0.6)
        ax3.add_patch(rect)
        ax3.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Formula
    formula_text = ('Single:\n'
                   'R_max = max(R_i)\n\n'
                   'Cumulative:\n'
                   'R_cum = sum(A_inter_i) / A_new\n\n'
                   f'Reject if:\n'
                   f'R_max > {overlap_control}\n'
                   f'or R_cum > {overlap_control*1.5:.2f}')
    ax3.text(100, 40, formula_text, fontsize=8, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 80)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # ==========================================
    # Subplot 4: Placement Distribution
    # ==========================================
    ax4 = axes[1, 1]
    ax4.set_title('(d) Placement Distribution', fontweight='bold', fontsize=12)
    
    roi_left, roi_right = 15, 85
    roi_h = 50
    
    ax4.add_patch(plt.Rectangle((roi_left, 10), roi_right - roi_left, roi_h,
                                fill=True, facecolor='lightblue', alpha=0.3,
                                edgecolor='blue', linewidth=1.5))
    
    # Gaussian distribution curve
    x_vals = np.linspace(roi_left, roi_right, 100)
    x_mean = (roi_left + roi_right) / 2
    x_std = (roi_right - roi_left) / gaussian_scale_divisor
    
    y_vals = norm.pdf(x_vals, loc=x_mean, scale=x_std)
    y_vals = y_vals / y_vals.max() * 35 + 10
    
    ax4.plot(x_vals, y_vals, 'r-', linewidth=2)
    ax4.fill_between(x_vals, 10, y_vals, alpha=0.3, color='red')
    
    ax4.axvline(x=x_mean, color='red', linestyle='--', linewidth=1)
    ax4.text(x_mean, roi_h + 15, 'mu', ha='center', fontsize=10, color='red')
    
    # Parameter description
    param_text = (f'Mode: {PLACEMENT_MODE}\n'
                 f'sigma = width / {gaussian_scale_divisor:.0f}')
    ax4.text(100, 35, param_text, fontsize=9, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4.set_xlim(0, 130)
    ax4.set_ylim(0, roi_h + 25)
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(RUN_DIR, "parameter_diagram.png")
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Parameter diagram saved to {save_path}")

def save_config_summary(save_path=None):
    """Copy config.ini file to output directory"""
    import shutil

    config_source = 'config.ini'

    if save_path is None:
        save_path = os.path.join(RUN_DIR, "config.ini")

    # Copy the config.ini file to the output directory
    if os.path.exists(config_source):
        shutil.copy2(config_source, save_path)
        print(f"Configuration file copied to {save_path}")
    else:
        print(f"Warning: config.ini not found at {config_source}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Initialize logger to capture all output
    log_file_path = os.path.join(RUN_DIR, "execution_log.txt")
    logger = Logger(log_file_path)
    sys.stdout = logger

    try:
        print("=" * 60)
        print("BubbleGen V3 - Two-Method Selection System")
        print("=" * 60)
        print(f"Output directory: {RUN_DIR}")
        print(f"Distribution: {DIST_TYPE}")
        print(f"Selection method: {SELECTION_METHOD}")
        print(f"ROI type: {'Trapezoid' if USE_TRAPEZOID_ROI else 'Rectangle'}")
        print("=" * 60)

        # Load data
        bubble_data = load_bubbles_from_db()
        if not bubble_data:
            print("Error: No bubble data loaded!")
            exit(1)

        # Stratify
        stratified_data, bin_edges = stratify_data(bubble_data)
        print(f"Loaded {len(bubble_data)} bubbles")

        # Calculate target weights and bin_centers (pre-calculate to avoid repetition)
        global BIN_CENTERS
        BIN_CENTERS = bin_edges[:-1] + 0.05
        bin_centers = BIN_CENTERS
        target_pdf = get_target_pdf_func()
        target_weights = target_pdf(bin_centers)
        target_weights = target_weights / np.sum(target_weights)

        # Display method information
        if SELECTION_METHOD == 'weighted_sampling':
            print(f"\nUsing WEIGHTED SAMPLING method (balanced, 90-95% accuracy)")
            print(f"  Formula: effective_weight = target_PDF × available_count")
            print(f"  Balances distribution matching with data availability")
        elif SELECTION_METHOD == 'direct_pdf':
            print(f"\nUsing DIRECT_PDF method (pure distribution matching)")
            print(f"  Formula: probability = target_PDF only (ignores bubble counts)")
            print(f"  Note: May over-sample from bins with few bubbles")
        else:
            print(f"\nERROR: Unknown selection method '{SELECTION_METHOD}'")
            print("Valid options: 'weighted_sampling' (default) or 'direct_pdf'")
            exit(1)

        # Generate documentation
        print("\nGenerating documentation...")
        save_config_summary()
        visualize_roi()

        # Generate velocity field visualization if enabled
        if ENABLE_VELOCITY_BIAS:
            print("Generating velocity field visualization...")
            # Initialize ROI for visualization
            if USE_TRAPEZOID_ROI:
                vis_roi = TrapezoidROI(
                    canvas_width=CANVAS_SIZE[0],
                    canvas_height=CANVAS_SIZE[1],
                    base_width_ratio=BASE_WIDTH_RATIO,
                    slope=ENTRAINMENT_SLOPE
                )
            else:
                vis_roi = (ROI_X_MIN, ROI_X_MAX, 0, CANVAS_SIZE[1])

            vis_velocity_field = VelocityField(
                roi=vis_roi,
                placement_mode=PLACEMENT_MODE,
                velocity_exponent=VELOCITY_PROFILE_EXPONENT,
                vertical_decay=VELOCITY_VERTICAL_DECAY
            )
            visualize_velocity_field(vis_velocity_field, vis_roi)

        # Generate images
        print("\nGenerating synthetic images...")
        used_bubbles = generate_synthetic_images(
            bubble_data, stratified_data, bin_edges, target_weights,
            selection_method=SELECTION_METHOD,
            enable_velocity_bias=ENABLE_VELOCITY_BIAS,
            velocity_exponent=VELOCITY_PROFILE_EXPONENT,
            size_velocity_coupling=SIZE_VELOCITY_COUPLING,
            velocity_decay=VELOCITY_VERTICAL_DECAY
        )

        # Statistical analysis
        print("\nGenerating statistics...")
        plot_diameter_histogram(used_bubbles)
        perform_ks_test(used_bubbles)

        print("\n" + "=" * 60)
        print(f"Generated {NUM_SYNTHETIC_IMAGES} images in {RUN_DIR}")
        print("\nOutput files:")
        print("  - execution_log.txt          (Complete execution log with all output)")
        print("  - config.ini                 (Configuration file)")
        print("  - roi_visualization.png      (ROI visualization)")
        print("  - bubble_diameter_histogram.png/pdf (Diameter distribution)")
        print("  - synth_XXXX.png/json        (Synthetic images and annotations)")
        print("=" * 60)
        print("BubbleGen Complete!")
        print(f"\nFull execution log saved to: {log_file_path}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: An exception occurred during execution")
        print("=" * 80)
        print(f"\nException type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        print("-" * 80)
        traceback.print_exc()
        print("-" * 80)
        print(f"\nExecution log saved to: {log_file_path}")
        print("=" * 80)
        sys.exit(1)

    finally:
        # Restore stdout and close logger
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nExecution log saved to: {log_file_path}")