import os
import cv2
import numpy as np
import random
import json
import sqlite3
import configparser
import math
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, lognorm, weibull_min, truncnorm, kstest
from matplotlib import pyplot as plt
from matplotlib import rcParams
import shutil

# -------------------------------
# Create Numbered Run Directory
# -------------------------------
def create_run_directory(base_output_dir):
    """
    Create a numbered run directory inside the output directory.
    If run0, run1, etc. already exist, it will create the next available run folder.
    """
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

    # Save a copy of config.ini in the new run directory
    config_path = 'config.ini'
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, 'config.ini'))
        print(f"Config file saved at: {os.path.join(run_dir, 'config.ini')}")
    
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
MIN_DISTANCE_RATIO = config.getfloat('Placement', 'min_distance_ratio')
overlap_control = config.getfloat('Placement', 'overlap_control')

# Filters parameters
APPLY_FEATHERING = config.getboolean('Filters', 'apply_feathering')
FEATHER_KERNEL_SIZE = tuple(map(int, config.get('Filters', 'feather_kernel_size').split(',')))
FEATHER_SIGMA = config.getfloat('Filters', 'feather_sigma')
FEATHER_EROSION_KERNEL = tuple(map(int, config.get('Filters', 'feather_erosion_kernel').split(',')))
FEATHER_EROSION_ITERATIONS = config.getint('Filters', 'feather_erosion_iterations')
APPLY_BILATERAL_FILTER = config.getboolean('Filters', 'apply_bilateral_filter')
BILATERAL_FILTER_DIAMETER = config.getint('Filters', 'bilateral_filter_diameter')
BILATERAL_FILTER_SIGMA_COLOR = config.getfloat('Filters', 'bilateral_filter_sigma_color')
BILATERAL_FILTER_SIGMA_SPACE = config.getfloat('Filters', 'bilateral_filter_sigma_space')
APPLY_GAUSSIAN_BLUR = config.getboolean('Filters', 'apply_gaussian_blur')

# Background parameters
USE_BACKGROUND_IMAGE = config.getboolean('Background', 'use_background_image')
BACKGROUND_IMAGE_PATH = config.get('Background', 'background_image_path')

# Database parameters
DATABASE_PATH = config.get('Database', 'database_path')
AGGREGATED_RESULTS_PATH = config.get('Database', 'aggregated_results_path')

# -------------------------------
# Distribution Selection Parameters
# -------------------------------
DIST_TYPE = config.get('Distribution', 'distribution_type').lower()
# Gaussian
USE_GAUSSIAN_SELECTION = config.getboolean('Gaussian', 'use_gaussian_selection')
GAUSSIAN_MU = config.getfloat('Gaussian', 'mu')
GAUSSIAN_SIGMA = config.getfloat('Gaussian', 'sigma')
# Weibull
WEIBULL_SHAPE = config.getfloat('Weibull', 'weibull_shape')
WEIBULL_SCALE = config.getfloat('Weibull', 'weibull_scale')
# Lognormal
LOGNORMAL_MU = config.getfloat('Lognormal', 'lognormal_mu')
LOGNORMAL_SIGMA = config.getfloat('Lognormal', 'lognormal_sigma')
# Constant distribution
if DIST_TYPE == "constant":
    # target 单位 m，例如 0.005 表示 5mm
    CONSTANT_TARGET = config.getfloat('Constant', 'target')

# -------------------------------
# Additional Placement Parameters
# -------------------------------
gaussian_x_min_ratio = config.getfloat('Placement', 'gaussian_x_min_ratio')
gaussian_x_max_ratio = config.getfloat('Placement', 'gaussian_x_max_ratio')
gaussian_scale_divisor = config.getfloat('Placement', 'gaussian_scale_divisor')

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

# -------------------------------
# Optimized Data Loading from SQLite
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
        for result in tqdm(executor.map(process_entry, rows), total=len(rows), desc="Processing Bubbles", unit="bubble"):
            if result:
                bubble_data.append(result)
    conn.close()
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(bubble_data, f)
    return bubble_data

# -------------------------------
# Stratification Functions (binsize = 0.1 mm)
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

def calculate_sample_sizes(bin_edges, total_samples):
    """Calculate sample sizes for each bin based on the target distribution."""
    bin_centers = bin_edges[:-1] + 0.05
    if DIST_TYPE == "gaussian":
        desired_density = norm.pdf(bin_centers, loc=GAUSSIAN_MU*1000, scale=GAUSSIAN_SIGMA*1000)
    elif DIST_TYPE == "weibull":
        desired_density = weibull_min.pdf(bin_centers, WEIBULL_SHAPE, scale=WEIBULL_SCALE*1000)
    elif DIST_TYPE == "lognormal":
        desired_density = lognorm.pdf(bin_centers, s=LOGNORMAL_SIGMA, scale=np.exp(LOGNORMAL_MU)*1000)
    elif DIST_TYPE == "constant":
        desired_density = np.where((bin_centers >= (CONSTANT_TARGET*1000 - 1)) & (bin_centers <= (CONSTANT_TARGET*1000 + 1)), 1/2.0, 0)
    else:
        desired_density = np.ones_like(bin_centers)
    densities_sum = np.sum(desired_density)
    sample_sizes = (desired_density / densities_sum * total_samples).astype(int)
    return sample_sizes

def stratified_sampling_without_replacement(stratified_data, sample_sizes):
    """Perform stratified sampling without replacement."""
    sampled_data = []
    for i, size in enumerate(sample_sizes):
        if size > 0 and len(stratified_data[i]) >= size:
            sampled_data.extend(random.sample(stratified_data[i], size))
        elif len(stratified_data[i]) > 0:
            sampled_data.extend(stratified_data[i])
    return sampled_data

# -------------------------------
# Auto Adjust Sampling Weights Function
# -------------------------------
def auto_adjust_sampling_weights(bubble_data, stratified_data, bin_edges, target_pdf_func, total_samples, 
                                 learning_rate=0.1, max_iter=10, eps=1e-6):
    """
    自动调整分层采样的权重，使得合成数据直方图更接近目标分布。
    """
    bin_centers = bin_edges[:-1] + 0.05  # 单位 mm
    diameters = np.array([b["bubble_diameter"] for b in bubble_data]) * 1000
    hist_counts, _ = np.histogram(diameters, bins=bin_edges)
    empirical_density = hist_counts / (np.sum(hist_counts) + eps)
    
    target_density = target_pdf_func(bin_centers)
    weights = target_density / (empirical_density + eps)
    weights = weights / np.sum(weights)

    for iteration in range(max_iter):
        sample_sizes = (weights * total_samples).astype(int)
        synthetic_samples = []
        for i, n in enumerate(sample_sizes):
            if len(stratified_data.get(i, [])) > 0 and n > 0:
                synthetic_samples.extend([bin_centers[i]] * n)
        synthetic_samples = np.array(synthetic_samples)
        if synthetic_samples.size == 0:
            print("Warning: No synthetic samples generated in iteration", iteration)
            break
        syn_counts, _ = np.histogram(synthetic_samples, bins=bin_edges)
        synthetic_density = syn_counts / (np.sum(syn_counts) + eps)
        diff = target_density - synthetic_density
        error = np.linalg.norm(diff)
        print(f"Iteration {iteration}: Error = {error:.6f}")
        if error < 1e-3:
            break
        adjustment = 1 + learning_rate * (diff / (target_density + eps))
        # 防止负值
        adjustment = np.clip(adjustment, 0, None)
        weights = weights * adjustment
        weights = weights / np.sum(weights)
    print("Final adjusted weights:", weights)
    return weights

def get_target_pdf_func():
    """
    根据 DIST_TYPE 返回目标概率密度函数（单位 mm）。
    """
    if DIST_TYPE == "gaussian":
        return lambda x: norm.pdf(x, loc=GAUSSIAN_MU*1000, scale=GAUSSIAN_SIGMA*1000)
    elif DIST_TYPE == "weibull":
        return lambda x: weibull_min.pdf(x, WEIBULL_SHAPE, scale=WEIBULL_SCALE*1000)
    elif DIST_TYPE == "lognormal":
        return lambda x: lognorm.pdf(x, s=LOGNORMAL_SIGMA, scale=np.exp(LOGNORMAL_MU)*1000)
    elif DIST_TYPE == "constant":
        return lambda x: np.where((x >= (CONSTANT_TARGET*1000 - 1)) & (x <= (CONSTANT_TARGET*1000 + 1)), 1/2.0, 0)
    else:
        return lambda x: 1

# -------------------------------
# New Bubble Selection Function: Stratified Sampling Based on Adjusted Weights
# -------------------------------
def select_bubble_stratified(stratified_data, bin_edges, adjusted_weights):
    """
    利用自动调整后的权重进行分层采样，随机选择一个分箱，再从该分箱中随机抽取一个气泡。
    """
    bin_centers = bin_edges[:-1] + 0.05
    counts = np.array([len(stratified_data.get(i, [])) for i in range(len(bin_centers))])
    weights_bins = adjusted_weights * counts
    total_weight = weights_bins.sum()
    if total_weight == 0:
        all_bubbles = [bubble for bubbles in stratified_data.values() for bubble in bubbles]
        return random.choice(all_bubbles)
    probabilities = weights_bins / total_weight
    chosen_bin = np.random.choice(len(probabilities), p=probabilities)
    if len(stratified_data[chosen_bin]) == 0:
        all_bubbles = [bubble for bubbles in stratified_data.values() for bubble in bubbles]
        return random.choice(all_bubbles)
    return random.choice(stratified_data[chosen_bin])

# -------------------------------
# Collision & Overlap Functions
# -------------------------------
def calculate_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
        return 0.0
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return intersection / min(area1, area2)

def compute_overlap_ratio(bbox_candidate, bbox_existing):
    """
    Compute the overlap ratio between the candidate bubble and an existing bubble.
    """
    x1, y1, x2, y2 = bbox_candidate
    a1, b1, a2, b2 = bbox_existing
    inter_x = max(0, min(x2, a2) - max(x1, a1))
    inter_y = max(0, min(y2, b2) - max(y1, b1))
    inter_area = inter_x * inter_y
    area_candidate = (x2 - x1) * (y2 - y1)
    if area_candidate == 0:
        return 0
    return inter_area / area_candidate

# -------------------------------
# Quadtree Implementation
# -------------------------------
class Rectangle:
    """Rectangle class representing the boundary of a quadtree node."""
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 >= self.x and y1 >= self.y and x2 <= self.x + self.w and y2 <= self.y + self.h)

    def intersects(self, bbox):
        x1, y1, x2, y2 = bbox
        return not (x2 < self.x or x1 > self.x + self.w or y2 < self.y or y1 > self.y + self.h)

class Quadtree:
    def __init__(self, boundary, capacity=4):
        """Initialize the quadtree node with the given boundary and capacity."""
        self.boundary = boundary
        self.capacity = capacity
        self.objects = []
        self.divided = False

    def subdivide(self):
        """Subdivide the current node into four child nodes."""
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h
        hw, hh = w / 2, h / 2
        ne = Rectangle(x + hw, y, hw, hh)
        nw = Rectangle(x, y, hw, hh)
        se = Rectangle(x + hw, y + hh, hw, hh)
        sw = Rectangle(x, y + hh, hw, hh)
        self.northeast = Quadtree(ne, self.capacity)
        self.northwest = Quadtree(nw, self.capacity)
        self.southeast = Quadtree(se, self.capacity)
        self.southwest = Quadtree(sw, self.capacity)
        self.divided = True

    def insert(self, bbox, bubble_info):
        """Insert a bounding box with associated bubble info into the quadtree."""
        if not self.boundary.intersects(bbox):
            return False
        if len(self.objects) < self.capacity:
            self.objects.append((bbox, bubble_info))
            return True
        else:
            if not self.divided:
                self.subdivide()
            if self.northeast.insert(bbox, bubble_info):
                return True
            if self.northwest.insert(bbox, bubble_info):
                return True
            if self.southeast.insert(bbox, bubble_info):
                return True
            if self.southwest.insert(bbox, bubble_info):
                return True
        return False

    def query(self, bbox, found=None):
        """Return a list of all objects that intersect with the given bbox.""" 
        if found is None:
            found = []
        if not self.boundary.intersects(bbox):
            return found
        for obj in self.objects:
            obj_bbox, info = obj
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
# Synthetic Image Generation using Quadtree with Fixed ROI
# -------------------------------
MAX_ATTEMPTS_PER_BUBBLE = 50

def generate_synthetic_images_quadtree(bubble_data, stratified_data, bin_edges, adjusted_weights):
    """
    生成合成图像，使用四叉树进行碰撞检测，
    并利用自动调整后的权重进行分层采样生成气泡。
    """
    used_bubbles = []
    ROI_x_min = int(CANVAS_SIZE[0] * gaussian_x_min_ratio)
    ROI_x_max = int(CANVAS_SIZE[0] * gaussian_x_max_ratio)
    ROI_width = ROI_x_max - ROI_x_min
    # 修正 ROI_area 计算：如果使用背景图，则计算背景图 ROI 区域的非零像素数
    if USE_BACKGROUND_IMAGE:
        roi_bg = bg_img[:, ROI_x_min:ROI_x_max]
        ROI_area = cv2.countNonZero(roi_bg)
    else:
        ROI_area = ROI_width * CANVAS_SIZE[1]
    x_mean = (ROI_x_min + ROI_x_max) / 2
    x_scale = (ROI_x_max - ROI_x_min) / gaussian_scale_divisor
    candidate_count = 10

    for img_id in tqdm(range(NUM_SYNTHETIC_IMAGES), desc="Generating Images (Quadtree)"):
        qt_boundary = Rectangle(ROI_x_min, 0, ROI_width, CANVAS_SIZE[1])
        quadtree = Quadtree(qt_boundary, capacity=4)
        placed_centers = []
        placed_bboxes = []
        if USE_BACKGROUND_IMAGE:
            canvas = bg_img.copy()
        else:
            canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0]), default_background_value, dtype=np.uint8)
        annotations = {"boxes": [], "masks": [], "areas": [], "void_fraction": 0.0}
        roi_acc_mask = np.zeros((CANVAS_SIZE[1], ROI_width), dtype=np.uint8)
        current_bubbles = []

        # 当背景图较小时，为保证至少放置 MAX_BUBBLES_PER_IMAGE 个气泡，可仅用气泡数量作为终止条件
        while (len(placed_bboxes) < MAX_BUBBLES_PER_IMAGE and 
               ( (ROI_area > 0 and (np.count_nonzero(roi_acc_mask) / ROI_area) < TARGET_VOID_FRACTION) or (ROI_area == 0))):
            bubble_placed = False
            attempts = 0
            while not bubble_placed and attempts < MAX_ATTEMPTS_PER_BUBBLE:
                attempts += 1
                bubble = select_bubble_stratified(stratified_data, bin_edges, adjusted_weights)
                bw, bh = bubble["bbox"]
                if bw > CANVAS_SIZE[0] or bh > CANVAS_SIZE[1]:
                    continue
                max_x = CANVAS_SIZE[0] - bw
                max_y = CANVAS_SIZE[1] - bh

                # 修改候选位置生成：使用截断正态分布（支持 'gaussian' 或 'guassian'）
                if PLACEMENT_MODE.lower() in ["gaussian", "guassian"]:
                    # 计算截断正态分布的 a, b 参数
                    a, b = (ROI_x_min - x_mean) / x_scale, (ROI_x_max - x_mean) / x_scale
                    candidate_x = truncnorm.rvs(a, b, loc=x_mean, scale=x_scale, size=candidate_count).astype(int)
                else:
                    candidate_x = np.random.randint(ROI_x_min, ROI_x_max - bw, size=candidate_count)
                candidate_y = np.random.randint(0, max_y, size=candidate_count)
                
                candidate_centers = np.column_stack((candidate_x + bw // 2, candidate_y + bh // 2))
                if placed_centers:
                    placed_arr = np.array(placed_centers)
                    diff = candidate_centers[:, np.newaxis, :] - placed_arr[np.newaxis, :, :]
                    distances = np.sqrt(np.sum(diff**2, axis=2))
                    valid_idx = np.where(np.min(distances, axis=1) >= MIN_DISTANCE_RATIO * max(bw, bh))[0]
                else:
                    valid_idx = np.arange(candidate_count)
                
                found_candidate = False
                chosen_x = chosen_y = None
                chosen_center = None
                for idx in valid_idx:
                    x = candidate_x[idx]
                    y = candidate_y[idx]
                    new_bbox = (x, y, x + bw, y + bh)
                    center = (x + bw // 2, y + bh // 2)
                    candidate_bubbles = quadtree.query(new_bbox)
                    collision = False
                    for existing_bbox, _ in candidate_bubbles:
                        if compute_overlap_ratio(new_bbox, existing_bbox) > overlap_control:
                            collision = True
                            break
                    if collision:
                        continue
                    chosen_x, chosen_y, chosen_center = x, y, center
                    found_candidate = True
                    break
                
                if found_candidate:
                    try:
                        roi_bubble = canvas[chosen_y:chosen_y+bh, chosen_x:chosen_x+bw]
                        if APPLY_FEATHERING:
                            mask = bubble["mask"]
                            binary_mask = (mask > 127).astype(np.uint8) * 255
                            erosion_kernel = np.ones(FEATHER_EROSION_KERNEL, np.uint8)
                            eroded_mask = cv2.erode(binary_mask, erosion_kernel, iterations=FEATHER_EROSION_ITERATIONS)
                            soft_mask = cv2.GaussianBlur(eroded_mask.astype(np.float32)/255, FEATHER_KERNEL_SIZE, FEATHER_SIGMA)
                            blended = roi_bubble * (1 - soft_mask) + bubble["image"] * soft_mask
                            roi_bubble[:] = blended.astype(np.uint8)
                            saved_mask = (bubble["mask"] > 127).astype(np.uint8)
                        else:
                            mask = (bubble["mask"] > 127)
                            roi_bubble[mask] = bubble["image"][mask]
                            saved_mask = mask.astype(np.uint8)
                        
                        placed_bboxes.append((chosen_x, chosen_y, chosen_x+bw, chosen_y+bh))
                        placed_centers.append(chosen_center)
                        annotations["boxes"].append((int(chosen_x), int(chosen_y), int(bw), int(bh)))
                        annotations["masks"].append(saved_mask.tolist())
                        annotations["areas"].append(int(bubble["true_area"]))
                        current_bubbles.append(bubble)
                        
                        quadtree.insert((chosen_x, chosen_y, chosen_x+bw, chosen_y+bh), {"center": chosen_center})
                        
                        overlap_x1 = max(chosen_x, ROI_x_min)
                        overlap_x2 = min(chosen_x+bw, ROI_x_max)
                        overlap_y1 = chosen_y
                        overlap_y2 = min(chosen_y+bh, CANVAS_SIZE[1])
                        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                            mask_region = saved_mask[overlap_y1-chosen_y:overlap_y2-chosen_y, overlap_x1-chosen_x:overlap_x2-chosen_x]
                            roi_acc_mask[overlap_y1:overlap_y2, overlap_x1-ROI_x_min:overlap_x2-ROI_x_min] = \
                                cv2.bitwise_or(roi_acc_mask[overlap_y1:overlap_y2, overlap_x1-ROI_x_min:overlap_x2-ROI_x_min], mask_region)
                        bubble_placed = True
                        break
                    except Exception as e:
                        continue
            if not bubble_placed:
                break
        
        used_bubbles.extend(current_bubbles)
        void_area = np.count_nonzero(roi_acc_mask)
        annotations["void_fraction"] = void_area / ROI_area if ROI_area > 0 else 0
        img_filename = os.path.join(RUN_DIR, f"synth_{img_id:04d}.png")
        json_filename = os.path.join(RUN_DIR, f"synth_{img_id:04d}.json")
        cv2.imwrite(img_filename, canvas)
        with open(json_filename, "w") as f:
            json.dump(annotations, f)
    
    return used_bubbles

# -------------------------------
# Plot Bubble Diameter Histogram
# -------------------------------
def plot_diameter_histogram(used_bubbles, save_path=None):
    """
    Plot a histogram of bubble diameters with expected distribution overlay.
    
    Parameters:
    -----------
    used_bubbles : list
        List of bubble dictionaries containing diameter information
    save_path : str, optional
        Custom path to save the figure, defaults to RUN_DIR/bubble_diameter_histogram.png
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    
    # Set publication-quality plot parameters with larger fonts for bigger figure
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 14       # Increased from 12
    rcParams['axes.labelsize'] = 16  # Larger axis labels
    rcParams['axes.titlesize'] = 18  # Larger title
    rcParams['xtick.labelsize'] = 14 # Larger tick labels
    rcParams['ytick.labelsize'] = 14 # Larger tick labels
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['lines.linewidth'] = 2.0 # Thicker lines
    
    # Extract bubble diameter data (m), convert to mm
    diameters_m = [bubble["bubble_diameter"] for bubble in used_bubbles if bubble.get("bubble_diameter") is not None]
    if not diameters_m:
        print("No bubble diameter data available for histogram.")
        return None
        
    diameters = np.array(diameters_m) * 1000  # Convert to mm
    
    # Calculate actual data range
    d_min, d_max = diameters.min(), diameters.max()
    
    # Create high-quality figure with larger dimensions
    fig, ax1 = plt.subplots(figsize=(160/25.4, 120/25.4))  # Size in mm (160mm × 120mm)
    
    # Set histogram bins with appropriate resolution
    bin_width = 0.1  # mm
    bins = np.arange(d_min, d_max + bin_width, bin_width)
    
    # Plot histogram with professional color scheme
    counts, bins, patches = ax1.hist(diameters, bins=bins, 
                                    color='#4878CF', alpha=0.7, 
                                    edgecolor='black', linewidth=0.8,
                                    label="Bubble Count")
    
    # Set primary axis properties
    ax1.set_xlabel("Diameter (mm)", fontweight='bold')
    ax1.set_ylabel("Bubble Count", color='#4878CF', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4878CF', width=1.5, length=4)
    ax1.tick_params(axis='x', width=1.5, length=4)
    ax1.set_xlim([d_min, d_max])
    
    # Add subtle grid for readability
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Define distribution functions based on DIST_TYPE
    if DIST_TYPE == "gaussian":
        mu_val = GAUSSIAN_MU * 1000
        sigma_val = GAUSSIAN_SIGMA * 1000
        dist_label = f"Gaussian\n($\mu$={mu_val:.1f} mm, $\sigma$={sigma_val:.1f} mm)"
        pdf_func = lambda x: norm.pdf(x, loc=mu_val, scale=sigma_val)
    elif DIST_TYPE == "weibull":
        weibull_scale_mm = WEIBULL_SCALE * 1000
        dist_label = f"Weibull\n($k$={WEIBULL_SHAPE:.2f}, scale={weibull_scale_mm:.1f} mm)"
        pdf_func = lambda x: (WEIBULL_SHAPE/weibull_scale_mm) * (x/weibull_scale_mm)**(WEIBULL_SHAPE - 1) * \
                           np.exp(-(x/weibull_scale_mm)**WEIBULL_SHAPE)
    elif DIST_TYPE == "lognormal":
        scale_mm = math.exp(LOGNORMAL_MU) * 1000
        dist_label = f"Lognormal\n($s$={LOGNORMAL_SIGMA:.2f}, scale={scale_mm:.1f} mm)"
        pdf_func = lambda x: lognorm.pdf(x, s=LOGNORMAL_SIGMA, scale=math.exp(LOGNORMAL_MU)*1000)
    elif DIST_TYPE == "constant":
        dist_label = f"Uniform-fixed\n(target={CONSTANT_TARGET*1000:.1f} mm ±1 mm)"
        pdf_func = lambda x: np.where((x >= (CONSTANT_TARGET*1000 - 1)) & (x <= (CONSTANT_TARGET*1000 + 1)), 1/2.0, 0)
    else:
        mu_val = GAUSSIAN_MU * 1000
        sigma_val = GAUSSIAN_SIGMA * 1000
        dist_label = f"Gaussian ($\mu$={mu_val:.1f} mm, $\sigma$={sigma_val:.1f} mm)"
        pdf_func = lambda x: norm.pdf(x, loc=mu_val, scale=sigma_val)
    
    # Calculate expected distribution with high resolution
    x_vals = np.linspace(d_min, d_max, 500)
    y_vals = np.array([pdf_func(x) for x in x_vals])
    
    # Create secondary axis for distribution curve
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_vals, color='#E24A33', lw=3.0, 
            label=f"Expected {dist_label}")
    ax2.set_ylabel("Probability Density", color='#E24A33', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#E24A33', width=1.5, length=4)
    ax2.set_xlim([d_min, d_max])
    
    # Create a combined legend in the upper right corner
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Remove any existing legends to avoid duplicates
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
        
    # Create new combined legend in upper right
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='lightgrey', fontsize=12)
    
    # Apply tight_layout only to the main axes, before adding the stats box
    fig.tight_layout()
    
    # CRITICAL: Create explicit stats box in lower left with absolute positioning
    # Add after tight_layout to avoid warnings
    
    # Get statistics for text box
    stats_text = (f"n = {len(diameters)}\n"
                 f"Mean = {np.mean(diameters):.2f} mm\n"
                 f"Median = {np.median(diameters):.2f} mm\n"
                 f"SD = {np.std(diameters):.2f} mm")
    
    # Add a separate axes for the stats text box to ensure it appears in the correct location
    # Increased the bottom value from 0.15 to 0.25 to raise the position
    stats_ax = fig.add_axes([0.15, 0.25, 0.2, 0.15])  # [left, bottom, width, height] in figure coordinates
    stats_ax.axis('off')  # Hide the axes
    stats_ax.text(0.05, 0.05, stats_text,
             horizontalalignment='left', verticalalignment='bottom',
             transform=stats_ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey', boxstyle='round,pad=0.5'),
             fontsize=12)
    
    # Save figure in publication-ready quality
    if save_path is None:
        save_path = os.path.join(RUN_DIR, "bubble_diameter_histogram.png")
        
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    
    # Also save as vector format for journal submission
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
    fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight', format='svg')
    
    plt.close()
    return fig

# -------------------------------
# K-S Test Function for Synthetic Data (考虑实际数据范围)
# -------------------------------
def perform_ks_test_synthetic(used_bubbles, eps=1e-6):
    """
    对生成的合成图像中气泡直径数据（单位 mm）与目标理论分布进行 K-S 检验，
    检验时根据实际数据范围 [d_min, d_max] 截断理论分布。
    """
    diameters = np.array([b["bubble_diameter"] for b in used_bubbles if b.get("bubble_diameter") is not None]) * 1000
    d_min, d_max = diameters.min(), diameters.max()
    if DIST_TYPE == "gaussian":
        base_cdf = lambda x: norm.cdf(x, loc=GAUSSIAN_MU*1000, scale=GAUSSIAN_SIGMA*1000)
    elif DIST_TYPE == "weibull":
        base_cdf = lambda x: weibull_min.cdf(x, WEIBULL_SHAPE, loc=0, scale=WEIBULL_SCALE*1000)
    elif DIST_TYPE == "lognormal":
        base_cdf = lambda x: lognorm.cdf(x, s=LOGNORMAL_SIGMA, loc=0, scale=math.exp(LOGNORMAL_MU)*1000)
    elif DIST_TYPE == "constant":
        base_cdf = lambda x: (x - (CONSTANT_TARGET*1000 - 1)) / 2.0
    else:
        base_cdf = lambda x: 1

    F_min = base_cdf(d_min)
    F_max = base_cdf(d_max)
    truncated_cdf = lambda x: (base_cdf(x) - F_min) / (F_max - F_min + eps)

    stat, p_value = kstest(diameters, truncated_cdf)
    print(f"K-S test for synthetic bubbles vs {DIST_TYPE} distribution (truncated [{d_min:.1f}, {d_max:.1f}] mm):")
    print(f"  Statistic = {stat:.4f}, p-value = {p_value:.4f}")
    return stat, p_value

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    print("BubbleGen Start")
    print(f"Using output directory: {RUN_DIR}")
    
    bubble_data = load_bubbles_from_db()
    if not bubble_data:
        print("Error: No bubble data loaded!")
        exit(1)
    
    stratified_data, bin_edges = stratify_data(bubble_data)
    avg_area = np.mean([b["true_area"] for b in bubble_data])
    print(f"Loaded {len(bubble_data)} bubbles | Average area: {avg_area:.1f} px")
    
    target_pdf_func = get_target_pdf_func()
    total_samples_for_adjustment = 10000
    adjusted_weights = auto_adjust_sampling_weights(bubble_data, stratified_data, bin_edges, 
                                                    target_pdf_func, total_samples_for_adjustment,
                                                    learning_rate=0.1, max_iter=10, eps=1e-6)
    
    used_bubbles = generate_synthetic_images_quadtree(bubble_data, stratified_data, bin_edges, adjusted_weights)
    plot_diameter_histogram(used_bubbles)
    
    perform_ks_test_synthetic(used_bubbles)
    
    print(f"Generated {NUM_SYNTHETIC_IMAGES} images in {RUN_DIR}")
    print("BubbleGen Complete")
