"""
BubbleGen Parameter Illustration Script
With Bimodal Distribution Support
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.stats import norm, truncnorm, lognorm, weibull_min
import configparser

# -------------------------------
# Load Configuration
# -------------------------------
config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
config.read('config.ini', encoding='utf-8')

# General parameters
NUM_SYNTHETIC_IMAGES = config.getint('General', 'num_synthetic_images', fallback=100)
TARGET_VOID_FRACTION = config.getfloat('General', 'target_void_fraction', fallback=0.15)
MAX_BUBBLES_PER_IMAGE = config.getint('General', 'max_bubbles_per_image', fallback=200)
CANVAS_WIDTH = config.getint('General', 'canvas_width', fallback=1001)
CANVAS_HEIGHT = config.getint('General', 'canvas_height', fallback=601)

# Placement parameters
PLACEMENT_MODE = config.get('Placement', 'placement_mode', fallback='gaussian')
overlap_control = config.getfloat('Placement', 'overlap_control', fallback=0.3)
gaussian_scale_divisor = config.getfloat('Placement', 'gaussian_scale_divisor', fallback=6)
gaussian_x_min_ratio = config.getfloat('Placement', 'gaussian_x_min_ratio', fallback=0.35)
gaussian_x_max_ratio = config.getfloat('Placement', 'gaussian_x_max_ratio', fallback=0.8)

# Entrainment parameters
USE_TRAPEZOID_ROI = config.getboolean('Placement', 'use_trapezoid_roi', fallback=True)
ENTRAINMENT_SLOPE = config.getfloat('Placement', 'entrainment_slope', fallback=0.15)
BASE_WIDTH_RATIO = config.getfloat('Placement', 'base_width_ratio', fallback=0.2)

# Distribution parameters
DIST_TYPE = config.get('Distribution', 'distribution_type', fallback='gaussian').lower()

# Gaussian parameters
GAUSSIAN_MU = config.getfloat('Gaussian', 'mu', fallback=0.003)
GAUSSIAN_SIGMA = config.getfloat('Gaussian', 'sigma', fallback=0.001)

# Weibull parameters
WEIBULL_SHAPE = config.getfloat('Weibull', 'weibull_shape', fallback=2.0)
WEIBULL_SCALE = config.getfloat('Weibull', 'weibull_scale', fallback=0.003)

# Lognormal parameters
LOGNORMAL_MU = config.getfloat('Lognormal', 'lognormal_mu', fallback=-5.8)
LOGNORMAL_SIGMA = config.getfloat('Lognormal', 'lognormal_sigma', fallback=0.5)

# Bimodal parameters (NEW)
BIMODAL_MU1 = config.getfloat('Bimodal', 'mu1', fallback=0.002)
BIMODAL_SIGMA1 = config.getfloat('Bimodal', 'sigma1', fallback=0.0005)
BIMODAL_MU2 = config.getfloat('Bimodal', 'mu2', fallback=0.005)
BIMODAL_SIGMA2 = config.getfloat('Bimodal', 'sigma2', fallback=0.001)
BIMODAL_WEIGHT1 = config.getfloat('Bimodal', 'weight1', fallback=0.6)  # Weight of first mode

# Constant parameters
CONSTANT_TARGET = config.getfloat('Constant', 'target', fallback=0.003)

# Uniform parameters (paper-aligned: f(r) = 1/(2*delta) over [r0-delta, r0+delta])
UNIFORM_R0 = config.getfloat('Uniform', 'uniform_r0', fallback=0.003)
UNIFORM_DELTA = config.getfloat('Uniform', 'uniform_delta', fallback=0.001)

# Visualization parameters - Simple size mapping
# Map diameter ranges to fixed pixel sizes (simpler logic)
# 1-2mm -> 6 pixels, 2-3mm -> 9 pixels, 3-4mm -> 12 pixels, etc.
BASE_PIXEL_SIZE = 0.8  # Base size for each mm (adjustable)

# -------------------------------
# Output Directory
# -------------------------------
ILLUSTRATION_DIR = "./parameter_illustrations"

# Clear existing images before generating new ones
if os.path.exists(ILLUSTRATION_DIR):
    for file in os.listdir(ILLUSTRATION_DIR):
        file_path = os.path.join(ILLUSTRATION_DIR, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg')):
            os.remove(file_path)
    print(f"Cleared existing images from {ILLUSTRATION_DIR}/")

os.makedirs(ILLUSTRATION_DIR, exist_ok=True)

# -------------------------------
# Color Scheme
# -------------------------------
COLOR_CANVAS = '#f5f5f5'
COLOR_ROI = '#3498db'
COLOR_ROI_FILL = '#85c1e9'
COLOR_BUBBLE_NEW = '#27ae60'
COLOR_BUBBLE_EXIST = '#e74c3c'
COLOR_INTERSECTION = '#f39c12'
COLOR_GAUSSIAN = '#9b59b6'
COLOR_SLOPE = '#e67e22'
COLOR_BUBBLE_DOT = '#2c3e50'


# -------------------------------
# Distribution Functions (Including Bimodal)
# -------------------------------
def bimodal_pdf(x, mu1, sigma1, mu2, sigma2, weight1):
    """
    Bimodal (mixture of two Gaussians) probability density function
    
    Parameters:
        x: array of values
        mu1, sigma1: mean and std of first mode
        mu2, sigma2: mean and std of second mode
        weight1: weight of first mode (0-1), second mode weight = 1 - weight1
    
    Returns:
        PDF values
    """
    weight2 = 1.0 - weight1
    pdf1 = norm.pdf(x, loc=mu1, scale=sigma1)
    pdf2 = norm.pdf(x, loc=mu2, scale=sigma2)
    return weight1 * pdf1 + weight2 * pdf2


def bimodal_sample(n, mu1, sigma1, mu2, sigma2, weight1, min_val=0, max_val=None):
    """
    Sample from bimodal distribution
    
    Parameters:
        n: number of samples
        mu1, sigma1: first mode parameters
        mu2, sigma2: second mode parameters
        weight1: probability of sampling from first mode
        min_val: minimum allowed value
        max_val: maximum allowed value
    
    Returns:
        Array of n samples
    """
    samples = []
    while len(samples) < n:
        # Choose which mode to sample from
        if np.random.random() < weight1:
            sample = np.random.normal(mu1, sigma1)
        else:
            sample = np.random.normal(mu2, sigma2)
        
        # Check bounds
        if sample >= min_val:
            if max_val is None or sample <= max_val:
                samples.append(sample)
    
    return np.array(samples)


def get_target_pdf_func():
    """
    Returns the target PDF function based on DIST_TYPE
    All parameters in meters, returns PDF in mm
    """
    if DIST_TYPE == "gaussian":
        mu_mm = GAUSSIAN_MU * 1000
        sigma_mm = GAUSSIAN_SIGMA * 1000
        return lambda x: norm.pdf(x, loc=mu_mm, scale=sigma_mm)
    
    elif DIST_TYPE == "weibull":
        scale_mm = WEIBULL_SCALE * 1000
        return lambda x: weibull_min.pdf(x, WEIBULL_SHAPE, scale=scale_mm)
    
    elif DIST_TYPE == "lognormal":
        scale_mm = np.exp(LOGNORMAL_MU) * 1000
        return lambda x: lognorm.pdf(x, s=LOGNORMAL_SIGMA, scale=scale_mm)
    
    elif DIST_TYPE == "bimodal":
        mu1_mm = BIMODAL_MU1 * 1000
        sigma1_mm = BIMODAL_SIGMA1 * 1000
        mu2_mm = BIMODAL_MU2 * 1000
        sigma2_mm = BIMODAL_SIGMA2 * 1000
        return lambda x: bimodal_pdf(x, mu1_mm, sigma1_mm, mu2_mm, sigma2_mm, BIMODAL_WEIGHT1)
    
    elif DIST_TYPE == "constant":
        target_mm = config.getfloat('Constant', 'target', fallback=0.003) * 1000
        return lambda x: np.where((x >= target_mm - 0.5) & (x <= target_mm + 0.5), 1.0, 0.0)
    
    elif DIST_TYPE == "uniform":
        # Paper-aligned: f(r) = 1/(2*delta) over [r0-delta, r0+delta]
        r0_mm = UNIFORM_R0 * 1000
        delta_mm = UNIFORM_DELTA * 1000
        return lambda x: np.where(
            (x >= (r0_mm - delta_mm)) & (x <= (r0_mm + delta_mm)),
            1.0 / (2.0 * delta_mm),
            0.0
        )

    else:
        return lambda x: np.ones_like(x)


def get_distribution_description():
    """Get text description of current distribution"""
    if DIST_TYPE == "gaussian":
        return f"Gaussian\nmu = {GAUSSIAN_MU*1000:.2f} mm\nsigma = {GAUSSIAN_SIGMA*1000:.2f} mm"
    
    elif DIST_TYPE == "weibull":
        return f"Weibull\nshape = {WEIBULL_SHAPE}\nscale = {WEIBULL_SCALE*1000:.2f} mm"
    
    elif DIST_TYPE == "lognormal":
        return f"Lognormal\nmu = {LOGNORMAL_MU}\nsigma = {LOGNORMAL_SIGMA}"
    
    elif DIST_TYPE == "bimodal":
        return (f"Bimodal (Gaussian Mixture)\n"
                f"Mode 1: mu={BIMODAL_MU1*1000:.2f}mm, sigma={BIMODAL_SIGMA1*1000:.2f}mm, w={BIMODAL_WEIGHT1:.0%}\n"
                f"Mode 2: mu={BIMODAL_MU2*1000:.2f}mm, sigma={BIMODAL_SIGMA2*1000:.2f}mm, w={1-BIMODAL_WEIGHT1:.0%}")
    
    else:
        return f"Distribution: {DIST_TYPE}"


# -------------------------------
# Simulated Database and Bubble Selection (Based on bubble_gen_v3 logic)
# -------------------------------
def create_simulated_database():
    """
    Create a simulated uniform bubble database with stratification
    Similar to bubble_gen_v3.py's stratify_data function

    Returns:
        stratified_data: dict mapping bin_index -> list of diameters (in mm)
        bin_edges: array of bin edges (in mm)
    """
    # Define bin edges (0.1mm bins from 0 to 10mm)
    bin_edges = np.arange(0, 10.1, 0.1)

    # Create uniform database: each bin has equal number of bubbles
    bubbles_per_bin = 100
    stratified_data = {}

    for i in range(len(bin_edges) - 1):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        # Generate uniform samples within this bin
        bin_samples = np.random.uniform(bin_edges[i], bin_edges[i+1], bubbles_per_bin)
        stratified_data[i] = bin_samples.tolist()

    return stratified_data, bin_edges


def select_bubble_from_database(stratified_data, bin_edges, target_weights):
    """
    Select a bubble from simulated database using target distribution weights
    Based on bubble_gen_v3.py's direct_pdf method

    Parameters:
        stratified_data: dict mapping bin_index -> list of diameters
        bin_edges: array of bin edges (in mm)
        target_weights: target PDF weights for each bin

    Returns:
        diameter_mm: selected bubble diameter in mm
    """
    # Normalize weights to probabilities
    probs = target_weights / np.sum(target_weights)

    # Select bin based on target distribution
    chosen_bin = np.random.choice(len(probs), p=probs)

    # Randomly select a bubble from this bin
    if chosen_bin in stratified_data and len(stratified_data[chosen_bin]) > 0:
        diameter_mm = np.random.choice(stratified_data[chosen_bin])
    else:
        # Fallback: sample from bin center
        diameter_mm = (bin_edges[chosen_bin] + bin_edges[chosen_bin + 1]) / 2

    return diameter_mm


# -------------------------------
# AABB Overlap Detection (matches bubble_gen_public.py Eq. 6-7)
# -------------------------------
def compute_bbox_intersection_area(b1, b2):
    """
    Compute intersection area of two axis-aligned bounding boxes.
    Each bbox: (x_min, y_min, x_max, y_max)
    """
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 > ix1 and iy2 > iy1:
        return (ix2 - ix1) * (iy2 - iy1)
    return 0.0


def check_overlap_aabb(new_bbox, existing_bboxes, single_threshold):
    """
    AABB-based overlap check matching bubble_gen_public.py Eq. 6-7.
    R_neighbor = A_intersection / A_neighbor > w_ol  ->  reject (True = overlap detected).
    Unidirectional: checks how much of each existing bubble is covered by the new one.
    """
    max_existing_ratio = 0.0
    for existing_bbox in existing_bboxes:
        inter_area = compute_bbox_intersection_area(new_bbox, existing_bbox)
        if inter_area > 0:
            existing_area = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
            if existing_area > 0:
                ratio = inter_area / existing_area
                max_existing_ratio = max(max_existing_ratio, ratio)
    return max_existing_ratio > single_threshold


# -------------------------------
# Bubble Placement Simulation
# -------------------------------
def simulate_bubble_placement(canvas_w, canvas_h, k, base_w, center_x,
                               max_bubbles, placement_mode, scale_divisor,
                               overlap_control_param, bubble_radius=1.5):
    """Simulate bubble placement using overlap_control parameter"""
    placed_bubbles = []
    max_attempts_per_bubble = 50
    
    np.random.seed(42)
    
    for _ in range(max_bubbles):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts_per_bubble:
            attempts += 1
            
            y = np.random.uniform(bubble_radius, canvas_h - bubble_radius)
            
            width_at_y = base_w + 2 * k * y
            roi_left = center_x - width_at_y / 2
            roi_right = center_x + width_at_y / 2
            
            if width_at_y < bubble_radius * 4:
                continue
            
            if placement_mode.lower() in ['gaussian', 'guassian']:
                x_mean = center_x
                x_std = width_at_y / scale_divisor
                
                a = (roi_left + bubble_radius - x_mean) / x_std
                b = (roi_right - bubble_radius - x_mean) / x_std
                
                try:
                    x = truncnorm.rvs(a, b, loc=x_mean, scale=x_std)
                except:
                    x = np.random.uniform(roi_left + bubble_radius, roi_right - bubble_radius)
            else:
                x = np.random.uniform(roi_left + bubble_radius, roi_right - bubble_radius)
            
            if x < roi_left + bubble_radius or x > roi_right - bubble_radius:
                continue
            
            # Convert overlap_control to minimum distance
            # overlap_control is allowed overlap ratio (0-1), convert to distance multiplier
            min_dist = bubble_radius * 2 * (1 - overlap_control_param)
            too_close = False
            
            for existing_x, existing_y in placed_bubbles:
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if dist < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                placed_bubbles.append((x, y))
                placed = True
        
        if not placed:
            break
    
    return placed_bubbles


# -------------------------------
# Figure: Size Distribution (with Bimodal support)
# -------------------------------
def plot_size_distribution():
    """
    Nature-style plot of bubble size distribution.
    Only shows parameters relevant to the current distribution type.
    """
    # Nature-style figure: main plot + parameter table
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.2)

    ax = fig.add_subplot(gs[0])
    ax_params = fig.add_subplot(gs[1])

    # ========== MAIN PLOT ==========
    # Get PDF function
    pdf_func = get_target_pdf_func()

    # Determine x range based on distribution type
    if DIST_TYPE == "gaussian":
        mu_mm = GAUSSIAN_MU * 1000
        sigma_mm = GAUSSIAN_SIGMA * 1000
        x_min = max(0, mu_mm - 4*sigma_mm)
        x_max = mu_mm + 4*sigma_mm

    elif DIST_TYPE == "bimodal":
        mu1_mm = BIMODAL_MU1 * 1000
        sigma1_mm = BIMODAL_SIGMA1 * 1000
        mu2_mm = BIMODAL_MU2 * 1000
        sigma2_mm = BIMODAL_SIGMA2 * 1000
        x_min = max(0, min(mu1_mm, mu2_mm) - 4*max(sigma1_mm, sigma2_mm))
        x_max = max(mu1_mm, mu2_mm) + 4*max(sigma1_mm, sigma2_mm)

    elif DIST_TYPE == "weibull":
        scale_mm = WEIBULL_SCALE * 1000
        x_min = 0
        x_max = scale_mm * 3

    elif DIST_TYPE == "lognormal":
        x_min = 0
        x_max = np.exp(LOGNORMAL_MU) * 1000 * 5

    else:
        x_min = 0
        x_max = 10

    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = pdf_func(x_vals)

    # Main distribution curve - minimal styling
    ax.fill_between(x_vals, 0, y_vals, color='#2b7bba', alpha=0.2)
    ax.plot(x_vals, y_vals, '#2b7bba', linewidth=2)

    # Type-specific annotations
    if DIST_TYPE == "bimodal":
        mu1_mm = BIMODAL_MU1 * 1000
        sigma1_mm = BIMODAL_SIGMA1 * 1000
        mu2_mm = BIMODAL_MU2 * 1000
        sigma2_mm = BIMODAL_SIGMA2 * 1000

        # Mode 1
        y1 = BIMODAL_WEIGHT1 * norm.pdf(x_vals, loc=mu1_mm, scale=sigma1_mm)
        ax.plot(x_vals, y1, color='#e63946', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.axvline(x=mu1_mm, color='#e63946', linestyle=':', linewidth=1, alpha=0.4)

        # Mode 2
        y2 = (1-BIMODAL_WEIGHT1) * norm.pdf(x_vals, loc=mu2_mm, scale=sigma2_mm)
        ax.plot(x_vals, y2, color='#06d6a0', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.axvline(x=mu2_mm, color='#06d6a0', linestyle=':', linewidth=1, alpha=0.4)

    elif DIST_TYPE == "gaussian":
        mu_mm = GAUSSIAN_MU * 1000
        sigma_mm = GAUSSIAN_SIGMA * 1000
        ax.axvline(x=mu_mm, color='#333333', linestyle='--', linewidth=1.2, alpha=0.5)
        ax.axvline(x=mu_mm - sigma_mm, color='#666666', linestyle=':', linewidth=1, alpha=0.4)
        ax.axvline(x=mu_mm + sigma_mm, color='#666666', linestyle=':', linewidth=1, alpha=0.4)

    # Clean axes styling
    ax.set_xlabel('Bubble Diameter (mm)', fontsize=11, family='serif')
    ax.set_ylabel('Probability Density', fontsize=11, family='serif')
    ax.set_title(f'Bubble Size Distribution', fontsize=12, fontweight='bold',
                loc='left', family='serif', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

    # ========== PARAMETER TABLE ==========
    ax_params.axis('off')
    ax_params.set_xlim(0, 1)
    ax_params.set_ylim(0, 1)

    param_data = []
    param_data.append(['Distribution Type', '', ''])
    param_data.append(['  type', f'{DIST_TYPE}', ''])
    param_data.append(['', '', ''])

    # Only show parameters for current distribution type
    if DIST_TYPE == 'gaussian':
        param_data.append(['Gaussian Parameters', '', ''])
        param_data.append(['  mu', f'{GAUSSIAN_MU*1000:.2f}', 'mm'])
        param_data.append(['  sigma', f'{GAUSSIAN_SIGMA*1000:.2f}', 'mm'])

    elif DIST_TYPE == 'bimodal':
        param_data.append(['Bimodal Parameters', '', ''])
        param_data.append(['  Mode 1', '', ''])
        param_data.append(['    mu1', f'{BIMODAL_MU1*1000:.2f}', 'mm'])
        param_data.append(['    sigma1', f'{BIMODAL_SIGMA1*1000:.2f}', 'mm'])
        param_data.append(['    weight1', f'{BIMODAL_WEIGHT1:.0%}', ''])
        param_data.append(['', '', ''])
        param_data.append(['  Mode 2', '', ''])
        param_data.append(['    mu2', f'{BIMODAL_MU2*1000:.2f}', 'mm'])
        param_data.append(['    sigma2', f'{BIMODAL_SIGMA2*1000:.2f}', 'mm'])
        param_data.append(['    weight2', f'{1-BIMODAL_WEIGHT1:.0%}', ''])

    elif DIST_TYPE == 'weibull':
        param_data.append(['Weibull Parameters', '', ''])
        param_data.append(['  shape', f'{WEIBULL_SHAPE:.2f}', ''])
        param_data.append(['  scale', f'{WEIBULL_SCALE*1000:.2f}', 'mm'])

    elif DIST_TYPE == 'lognormal':
        param_data.append(['Lognormal Parameters', '', ''])
        param_data.append(['  mu', f'{LOGNORMAL_MU:.3f}', ''])
        param_data.append(['  sigma', f'{LOGNORMAL_SIGMA:.3f}', ''])

    elif DIST_TYPE == 'constant':
        param_data.append(['Constant Parameters', '', ''])
        param_data.append(['  target', f'{CONSTANT_TARGET*1000:.2f}', 'mm'])

    # Create table
    table = ax_params.table(cellText=param_data,
                           colWidths=[0.60, 0.25, 0.15],
                           cellLoc='left',
                           loc='center',
                           bbox=[0.05, 0.3, 0.9, 0.5])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        cell.set_linewidth(0.5)
        cell.set_text_props(family='serif')

        # Section headers
        if j == 0 and param_data[i][0] and not param_data[i][0].startswith('  '):
            cell.set_text_props(weight='bold', size=10)
            cell.set_facecolor('#e8e8e8')
        # Empty rows
        elif param_data[i][0] == '':
            cell.set_facecolor('white')
            cell.set_linewidth(0)
        # Parameter names
        elif j == 0:
            cell.set_text_props(family='monospace', size=8)
            cell.set_facecolor('#f9f9f9')
        # Values
        elif j == 1:
            cell.set_text_props(family='monospace', size=8, weight='bold')
            cell.set_facecolor('white')
        # Units
        else:
            cell.set_text_props(size=8, style='italic', color='#666666')
            cell.set_facecolor('white')

    ax_params.text(0.5, 0.85, 'Parameters',
                  ha='center', va='top', fontsize=11, fontweight='bold',
                  family='serif', transform=ax_params.transAxes)

    save_path = os.path.join(ILLUSTRATION_DIR, "size_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# -------------------------------
# Figure: Distribution Comparison (All Types)
# -------------------------------
def plot_distribution_comparison():
    """
    Nature-style comparison of all available distribution types.
    Parameters are shown within each subplot. No separate parameter table needed.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    distributions = [
        ('gaussian', 'Gaussian', '#2b7bba'),
        ('bimodal', 'Bimodal', '#9b59b6'),
        ('weibull', 'Weibull', '#27ae60'),
        ('lognormal', 'Lognormal', '#e67e22'),
        ('constant', 'Constant', '#e74c3c'),
        ('uniform', 'Uniform', '#95a5a6'),
    ]

    x_range = np.linspace(0, 10, 500)  # 0-10 mm

    for ax, (dist_type, title, color) in zip(axes, distributions):
        # Nature-style title
        is_current = (dist_type == DIST_TYPE)
        title_text = f'{title}' + (' [Current]' if is_current else '')
        ax.set_title(title_text, fontsize=11, fontweight='bold',
                    loc='left', family='serif', color='#c0392b' if is_current else '#333333')

        if dist_type == 'gaussian':
            mu = GAUSSIAN_MU * 1000
            sigma = GAUSSIAN_SIGMA * 1000
            y = norm.pdf(x_range, loc=mu, scale=sigma)
            params = f'μ = {mu:.2f} mm\nσ = {sigma:.2f} mm'

        elif dist_type == 'bimodal':
            mu1 = BIMODAL_MU1 * 1000
            sigma1 = BIMODAL_SIGMA1 * 1000
            mu2 = BIMODAL_MU2 * 1000
            sigma2 = BIMODAL_SIGMA2 * 1000
            w1 = BIMODAL_WEIGHT1
            y = bimodal_pdf(x_range, mu1, sigma1, mu2, sigma2, w1)

            # Show components with minimal styling
            y1 = w1 * norm.pdf(x_range, loc=mu1, scale=sigma1)
            y2 = (1-w1) * norm.pdf(x_range, loc=mu2, scale=sigma2)
            ax.fill_between(x_range, 0, y1, color='#e63946', alpha=0.15)
            ax.fill_between(x_range, 0, y2, color='#06d6a0', alpha=0.15)
            ax.plot(x_range, y1, color='#e63946', linestyle='--', linewidth=1, alpha=0.5)
            ax.plot(x_range, y2, color='#06d6a0', linestyle='--', linewidth=1, alpha=0.5)

            params = f'μ1={mu1:.1f}, w1={w1:.0%}\nμ2={mu2:.1f}, w2={1-w1:.0%}'

        elif dist_type == 'weibull':
            shape = WEIBULL_SHAPE
            scale = WEIBULL_SCALE * 1000
            y = weibull_min.pdf(x_range, shape, scale=scale)
            params = f'k = {shape:.2f}\nλ = {scale:.2f} mm'

        elif dist_type == 'lognormal':
            s = LOGNORMAL_SIGMA
            scale = np.exp(LOGNORMAL_MU) * 1000
            y = lognorm.pdf(x_range, s=s, scale=scale)
            params = f'μ = {LOGNORMAL_MU:.3f}\nσ = {s:.3f}'

        elif dist_type == 'constant':
            target = CONSTANT_TARGET * 1000
            y = np.where((x_range >= target - 0.5) & (x_range <= target + 0.5), 2.5, 0)
            params = f'D = {target:.2f} mm'

        else:  # uniform
            y = np.ones_like(x_range) * 0.1
            params = 'Uniform\ndistribution'

        # Clean plotting
        ax.fill_between(x_range, 0, y, color=color, alpha=0.2)
        ax.plot(x_range, y, color=color, linewidth=1.8)

        ax.set_xlabel('Diameter (mm)', fontsize=9, family='serif')
        ax.set_ylabel('Probability Density', fontsize=9, family='serif')
        ax.set_xlim(0, 10)
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.12, linewidth=0.5)

        # Minimal parameter annotation
        ax.text(0.97, 0.97, params, transform=ax.transAxes, fontsize=8,
               va='top', ha='right', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#cccccc', linewidth=0.8, alpha=0.95))

        # Highlight current distribution with subtle background
        if is_current:
            ax.patch.set_facecolor('#fff9e6')
            ax.patch.set_alpha(0.5)

    plt.suptitle('Bubble Size Distribution Types', fontsize=13, fontweight='bold',
                family='serif', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(ILLUSTRATION_DIR, "distribution_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# -------------------------------
# Figure: Overlap Extreme Cases
# -------------------------------
def plot_overlap_extreme_cases():
    """
    Illustrate all realistic overlap cases using config parameters.
    All bubble sizes and distances are derived from config.ini settings.

    Cases generated based on:
    1. overlap_control constraint from config
    2. Distribution-based bubble sizes (using mu/sigma from config)
    3. R = A_inter / A_exist ∈ [0, 1]
    4. overlap_control threshold from config
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    # Generate representative bubble sizes from distribution config
    # Use mean ± standard deviation to get realistic size range
    if DIST_TYPE == 'gaussian':
        mean_diameter = GAUSSIAN_MU * 1000  # Convert to mm
        std_diameter = GAUSSIAN_SIGMA * 1000
    elif DIST_TYPE == 'bimodal':
        mean_diameter = (BIMODAL_MU1 * BIMODAL_WEIGHT1 + BIMODAL_MU2 * (1-BIMODAL_WEIGHT1)) * 1000
        std_diameter = max(BIMODAL_SIGMA1, BIMODAL_SIGMA2) * 1000
    elif DIST_TYPE == 'weibull':
        mean_diameter = WEIBULL_SCALE * 1000
        std_diameter = mean_diameter * 0.3  # Approximate
    elif DIST_TYPE == 'lognormal':
        mean_diameter = np.exp(LOGNORMAL_MU) * 1000
        std_diameter = mean_diameter * LOGNORMAL_SIGMA
    else:
        mean_diameter = 3.0  # Default
        std_diameter = 1.0

    # Convert to pixel radii for display (assuming scale factor)
    pixel_scale = 5  # pixels per mm
    r_medium = mean_diameter * pixel_scale / 2
    r_small = max(r_medium - std_diameter * pixel_scale, r_medium * 0.5)
    r_large = r_medium + std_diameter * pixel_scale

    # Calculate threshold distance based on overlap_control
    # Note: overlap_control is the allowed overlap ratio (0-1)
    # For visualization purposes, we use it to calculate minimum spacing
    min_dist_multiplier = 2 * (1 - overlap_control)  # Convert overlap ratio to distance multiplier

    # Define cases based on config parameters
    # (title, description, r_exist, r_new, distance_ratio)
    cases = [
        # Row 1: Equal size bubbles with different spacing
        ("Case 1: Min Distance",
         f"Equal sizes, min spacing\n(overlap_control={overlap_control})",
         r_medium, r_medium, min_dist_multiplier),

        ("Case 2: At Threshold",
         f"Equal sizes, at threshold\n(overlap_control={overlap_control})",
         r_medium, r_medium, 2 * (1 - overlap_control)),

        ("Case 3: Rejected Overlap",
         f"Equal sizes, overlap > threshold\nWould be REJECTED",
         r_medium, r_medium, 2 * (1 - overlap_control * 1.5)),

        # Row 2: Small new bubble (from lower distribution range)
        ("Case 4: Small New, Separated",
         f"Small new (μ-σ)\nMin distance",
         r_medium, r_small, min_dist_multiplier),

        ("Case 5: Small New, Overlap",
         f"Small new bubble\nPartial overlap",
         r_medium, r_small, 1.5),

        ("Case 6: Small Inside Large",
         f"Small new inside large existing\nR ≈ (small/large)²",
         r_medium, r_small, 0.3),

        # Row 3: Large new bubble (from upper distribution range)
        ("Case 7: Large New, Separated",
         f"Large new (μ+σ)\nNo overlap",
         r_medium, r_large, 2.5),

        ("Case 8: Large Covers Small",
         f"Large new covers small existing\nR ≈ 1.0 (existing fully inside)",
         r_small, r_large, 0.5),

        ("Case 9: Large Overlap",
         f"Large new, significant overlap\nExceeds threshold",
         r_medium * 0.8, r_large, 1.2),
    ]

    for ax, (title, desc, r_exist, r_new, dist_ratio) in zip(axes, cases):
        ax.set_title(f'{title}\n{desc}', fontsize=10, fontweight='bold', pad=10)
        ax.set_xlim(-5, 85)
        ax.set_ylim(-5, 70)
        ax.set_aspect('equal')
        ax.axis('off')

        # Position bubbles
        center_x = 40
        center_y = 35
        pos_exist = (center_x - dist_ratio * r_exist / 2, center_y)
        pos_new = (center_x + dist_ratio * r_new / 2, center_y)

        # Calculate distance and intersection
        dist = np.sqrt((pos_new[0] - pos_exist[0])**2 + (pos_new[1] - pos_exist[1])**2)

        # Calculate overlap ratio R
        if dist >= r_exist + r_new:
            R_value = 0.0  # No overlap
            inter_area = 0
        elif dist <= abs(r_exist - r_new):
            # One circle completely inside the other
            smaller_r = min(r_exist, r_new)
            R_value = (np.pi * smaller_r**2) / (np.pi * r_exist**2)
            inter_area = np.pi * smaller_r**2
        else:
            # Partial overlap - calculate intersection area
            # Using formula for two circles intersection
            d = dist
            r1, r2 = r_exist, r_new
            part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2*d*r1))
            part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2*d*r2))
            part3 = 0.5 * np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
            inter_area = part1 + part2 - part3
            R_value = inter_area / (np.pi * r_exist**2)

        # Draw intersection region first (if exists)
        if dist < r_exist + r_new:
            if dist <= abs(r_exist - r_new):
                # One inside the other - draw the smaller circle as intersection
                smaller_r = min(r_exist, r_new)
                smaller_pos = pos_exist if r_exist < r_new else pos_new
                inter_circle = plt.Circle(smaller_pos, smaller_r,
                                        facecolor=COLOR_INTERSECTION, alpha=0.7, zorder=4,
                                        edgecolor='orange', linewidth=2)
                ax.add_patch(inter_circle)
            else:
                # Partial overlap - draw lens shape
                r1, r2 = r_exist, r_new
                d = dist

                # Calculate intersection points
                a = (r1**2 - r2**2 + d**2) / (2*d)
                h = np.sqrt(max(0, r1**2 - a**2))

                cx = pos_exist[0] + a * (pos_new[0] - pos_exist[0]) / d
                cy = pos_exist[1] + a * (pos_new[1] - pos_exist[1]) / d

                dx = -(pos_new[1] - pos_exist[1]) / d
                dy = (pos_new[0] - pos_exist[0]) / d

                p1 = (cx + h * dx, cy + h * dy)
                p2 = (cx - h * dx, cy - h * dy)

                # Draw lens
                theta1 = np.arctan2(p1[1] - pos_exist[1], p1[0] - pos_exist[0])
                theta2 = np.arctan2(p2[1] - pos_exist[1], p2[0] - pos_exist[0])

                if theta2 > theta1:
                    theta1, theta2 = theta2, theta1

                angles_exist = np.linspace(theta1, theta2, 50)
                arc_exist_x = pos_exist[0] + r_exist * np.cos(angles_exist)
                arc_exist_y = pos_exist[1] + r_exist * np.sin(angles_exist)

                theta3 = np.arctan2(p1[1] - pos_new[1], p1[0] - pos_new[0])
                theta4 = np.arctan2(p2[1] - pos_new[1], p2[0] - pos_new[0])

                if theta3 > theta4:
                    theta3, theta4 = theta4, theta3

                angles_new = np.linspace(theta3, theta4, 50)
                arc_new_x = pos_new[0] + r_new * np.cos(angles_new)
                arc_new_y = pos_new[1] + r_new * np.sin(angles_new)

                lens_x = np.concatenate([arc_exist_x, arc_new_x[::-1]])
                lens_y = np.concatenate([arc_exist_y, arc_new_y[::-1]])

                ax.fill(lens_x, lens_y, color=COLOR_INTERSECTION, alpha=0.7, zorder=4,
                       edgecolor='orange', linewidth=2)

        # Draw existing bubble
        exist_circle = plt.Circle(pos_exist, r_exist,
                                 facecolor=COLOR_BUBBLE_EXIST,
                                 edgecolor='darkred', linewidth=2.5,
                                 alpha=0.5, zorder=2)
        ax.add_patch(exist_circle)
        # Convert pixel radius back to mm diameter for display
        exist_diameter_mm = (r_exist * 2) / pixel_scale
        ax.text(pos_exist[0], pos_exist[1], f'Existing\nD={exist_diameter_mm:.2f}mm',
               ha='center', va='center', fontsize=8, color='darkred', fontweight='bold')

        # Draw new bubble
        new_circle = plt.Circle(pos_new, r_new,
                               facecolor=COLOR_BUBBLE_NEW,
                               edgecolor='darkgreen', linewidth=2.5,
                               alpha=0.5, zorder=3)
        ax.add_patch(new_circle)
        # Convert pixel radius back to mm diameter for display
        new_diameter_mm = (r_new * 2) / pixel_scale
        ax.text(pos_new[0], pos_new[1], f'New\nD={new_diameter_mm:.2f}mm',
               ha='center', va='center', fontsize=8, color='darkgreen', fontweight='bold')

        # Decision text with detailed info
        if R_value <= overlap_control:
            decision = 'ACCEPT ✓'
            decision_color = 'green'
            bgcolor = 'lightgreen'
        else:
            decision = 'REJECT ✗'
            decision_color = 'red'
            bgcolor = 'lightcoral'

        info_text = (f'R = {R_value:.3f}\n'
                    f'Distance = {dist:.1f}\n'
                    f'Threshold = {overlap_control}\n'
                    f'Decision: {decision}')

        ax.text(42, 5, info_text, fontsize=8,
               va='bottom', ha='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor=bgcolor,
                        edgecolor=decision_color, linewidth=2, alpha=0.85))

    # Clean title - Nature style
    plt.suptitle('Bubble Overlap Cases - R = A_intersection / A_existing',
                fontsize=13, fontweight='bold', family='serif', y=0.98)

    # Add parameter box in top right corner of figure
    param_text_lines = [
        'Overlap Parameters:',
        f'  overlap_control = {overlap_control}',
        '',
        'Distribution:',
        f'  type = {DIST_TYPE}',
    ]

    if DIST_TYPE == 'gaussian':
        param_text_lines.append(f'  μ = {GAUSSIAN_MU*1000:.1f} mm')
        param_text_lines.append(f'  σ = {GAUSSIAN_SIGMA*1000:.1f} mm')
    elif DIST_TYPE == 'bimodal':
        param_text_lines.append(f'  μ1 = {BIMODAL_MU1*1000:.1f} mm')
        param_text_lines.append(f'  μ2 = {BIMODAL_MU2*1000:.1f} mm')
    elif DIST_TYPE == 'weibull':
        param_text_lines.append(f'  scale = {WEIBULL_SCALE*1000:.1f} mm')
    elif DIST_TYPE == 'lognormal':
        param_text_lines.append(f'  μ = {LOGNORMAL_MU:.2f}')
        param_text_lines.append(f'  σ = {LOGNORMAL_SIGMA:.2f}')

    param_text = '\n'.join(param_text_lines)
    fig.text(0.98, 0.95, param_text, fontsize=8, family='monospace',
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9f9',
                     edgecolor='#cccccc', linewidth=1, alpha=0.95))

    plt.tight_layout(rect=[0, 0, 0.96, 0.96])

    save_path = os.path.join(ILLUSTRATION_DIR, "overlap_extreme_cases.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# -------------------------------
# Figure: Unified Parameter Diagram
# -------------------------------
def plot_unified_parameter_diagram():
    """
    Simplified version: Generate bubbles from simulated database based on target distribution.
    Uses v3 logic: select bin based on PDF, then randomly sample diameter from bin.
    No right panel, no title text.
    """
    # Create simulated database and calculate target weights
    stratified_data, bin_edges = create_simulated_database()
    bin_centers = bin_edges[:-1] + 0.05
    target_pdf = get_target_pdf_func()
    target_weights = target_pdf(bin_centers)
    target_weights = target_weights / np.sum(target_weights)

    # Single full-width figure (no right panel)
    fig, ax_main = plt.subplots(figsize=(12, 9))

    # Remove title (as requested)
    # ax_main.set_title('...')  # Removed

    scale = 200 / CANVAS_WIDTH
    canvas_w = CANVAS_WIDTH * scale
    canvas_h = CANVAS_HEIGHT * scale

    k = ENTRAINMENT_SLOPE
    base_w = canvas_w * BASE_WIDTH_RATIO
    top_w = min(base_w + 2 * k * canvas_h, canvas_w)
    center_x = canvas_w / 2

    # Canvas background
    ax_main.add_patch(Rectangle((0, 0), canvas_w, canvas_h,
                                fill=True, facecolor='#f5f5f5',
                                edgecolor='#333333', linewidth=1.5, zorder=1))

    # ROI trapezoid
    trap_points = np.array([
        [center_x - top_w/2, canvas_h],
        [center_x + top_w/2, canvas_h],
        [center_x + base_w/2, 0],
        [center_x - base_w/2, 0],
    ])
    trap = Polygon(trap_points, fill=True, facecolor='#e8f4f8',
                  edgecolor='#2b7bba', linewidth=1.5, alpha=0.4, zorder=2)
    ax_main.add_patch(trap)

    # Slope lines
    ax_main.plot([center_x - base_w/2, center_x - top_w/2], [0, canvas_h],
                color='#666666', linewidth=1.2, linestyle='--', alpha=0.5, zorder=3)
    ax_main.plot([center_x + base_w/2, center_x + top_w/2], [0, canvas_h],
                color='#666666', linewidth=1.2, linestyle='--', alpha=0.5, zorder=3)

    # Generate bubbles using v3 logic: select from database based on target distribution
    placed_bubbles = []  # each entry: (x, y, radius_px, bbox)
    max_attempts_per_bubble = 50
    np.random.seed(42)

    # Void fraction tracking (matches public.py pixel-mask accumulation, approximated via circle bbox area)
    roi_area = (base_w + top_w) * canvas_h / 2.0  # trapezoid area
    roi_acc_area = 0.0

    for _ in range(MAX_BUBBLES_PER_IMAGE):
        # Void fraction stop condition (matches public.py)
        if roi_area > 0 and roi_acc_area / roi_area >= TARGET_VOID_FRACTION:
            break

        placed = False
        attempts = 0

        while not placed and attempts < max_attempts_per_bubble:
            attempts += 1

            # STEP 1: Select bubble diameter from database based on target distribution (v3 logic)
            diameter_mm = select_bubble_from_database(stratified_data, bin_edges, target_weights)

            # STEP 2: Convert diameter from mm to pixels using simple range mapping
            # 1-2mm -> 6px, 2-3mm -> 9px, 3-4mm -> 12px, etc.
            # Formula: diameter_px = ceil(diameter_mm) * BASE_PIXEL_SIZE
            bubble_diameter_px = int(np.ceil(diameter_mm)) * BASE_PIXEL_SIZE
            bubble_radius_px = bubble_diameter_px / 2.0

            # Skip if bubble is too large for canvas
            if bubble_radius_px * 2 >= canvas_h or bubble_radius_px * 2 >= canvas_w:
                continue

            # STEP 3: Generate position within trapezoid ROI
            y_min = max(bubble_radius_px, 0)
            y_max = max(canvas_h - bubble_radius_px, y_min + 1)
            y = np.random.uniform(y_min, y_max)

            width_at_y = base_w + 2 * k * y
            roi_left = center_x - width_at_y / 2
            roi_right = center_x + width_at_y / 2

            if PLACEMENT_MODE.lower() in ['gaussian', 'guassian']:
                x_mean = center_x
                x_std = width_at_y / gaussian_scale_divisor

                # Ensure valid range
                x_min = roi_left + bubble_radius_px
                x_max = roi_right - bubble_radius_px

                if x_max <= x_min or x_std <= 0:
                    # Fallback to uniform if range is invalid
                    x = np.random.uniform(x_min, x_max) if x_max > x_min else center_x
                else:
                    a = (x_min - x_mean) / x_std
                    b = (x_max - x_mean) / x_std

                    # Ensure a < b
                    if a >= b:
                        x = np.random.uniform(x_min, x_max)
                    else:
                        x = truncnorm.rvs(a, b, loc=x_mean, scale=x_std)
            else:
                x_min = roi_left + bubble_radius_px
                x_max = roi_right - bubble_radius_px
                x = np.random.uniform(x_min, x_max) if x_max > x_min else center_x

            # Check bounds
            if x < roi_left + bubble_radius_px or x > roi_right - bubble_radius_px:
                continue

            # STEP 4: AABB overlap check (matches public.py Eq. 6-7)
            # Bounding box for the simulated circle
            new_bbox = (x - bubble_radius_px, y - bubble_radius_px,
                        x + bubble_radius_px, y + bubble_radius_px)
            existing_bboxes = [bb for _, _, _, bb in placed_bubbles]

            if not check_overlap_aabb(new_bbox, existing_bboxes, overlap_control):
                placed_bubbles.append((x, y, bubble_radius_px, new_bbox))
                roi_acc_area += np.pi * bubble_radius_px ** 2  # circle area proxy for mask
                placed = True

    # Draw bubbles as circles with sizes from database
    for x, y, r, _ in placed_bubbles:
        circle = plt.Circle((x, y), r, facecolor='#2b7bba',
                           alpha=0.65, edgecolor='white', linewidth=0.8, zorder=4)
        ax_main.add_patch(circle)

    # Gaussian distribution overlay
    if PLACEMENT_MODE.lower() in ['gaussian', 'guassian']:
        sample_heights = [canvas_h * 0.2, canvas_h * 0.5, canvas_h * 0.8]
        gaussian_color = '#ff6b6b'

        for sample_z in sample_heights:
            sample_width = base_w + 2 * k * sample_z
            x_gaussian = np.linspace(center_x - sample_width/2, center_x + sample_width/2, 100)
            x_std = sample_width / gaussian_scale_divisor
            y_gaussian = norm.pdf(x_gaussian, loc=center_x, scale=x_std)

            curve_height = 12
            y_gaussian = y_gaussian / y_gaussian.max() * curve_height + sample_z

            ax_main.fill_between(x_gaussian, sample_z, y_gaussian,
                                color=gaussian_color, alpha=0.15, zorder=5)
            ax_main.plot(x_gaussian, y_gaussian, color=gaussian_color,
                        linewidth=1.0, alpha=0.4, zorder=5, linestyle='-')

    # Annotations
    TEXT_ZORDER = 10
    width_color = '#27ae60'
    slope_color = '#d68910'

    # Canvas width
    canvas_w_y = -18
    ax_main.annotate('', xy=(canvas_w, canvas_w_y), xytext=(0, canvas_w_y),
                    arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2),
                    zorder=TEXT_ZORDER)
    ax_main.text(canvas_w/2, canvas_w_y - 5, f'W = {CANVAS_WIDTH} px',
                ha='center', fontsize=9, color='#333333', family='serif',
                zorder=TEXT_ZORDER)

    # Canvas height
    ax_main.annotate('', xy=(-8, canvas_h), xytext=(-8, 0),
                    arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2),
                    zorder=TEXT_ZORDER)
    ax_main.text(-13, canvas_h/2, f'H = {CANVAS_HEIGHT} px',
                ha='center', va='center', fontsize=9, color='#333333',
                family='serif', rotation=90, zorder=TEXT_ZORDER)

    # ROI label
    roi_y = canvas_h + 8
    ax_main.text(center_x, roi_y, 'ROI',
                ha='center', va='bottom', fontsize=11, color='#2b7bba',
                family='serif', fontweight='bold', zorder=TEXT_ZORDER)

    # Width formula
    formula_y = roi_y - 7
    ax_main.text(center_x, formula_y, r'$w = w_0 + 2kH$',
                ha='center', va='bottom', fontsize=9, color=width_color,
                family='serif', fontweight='bold', zorder=TEXT_ZORDER)

    # Base width
    base_y = -5
    ax_main.annotate('', xy=(center_x - base_w/2, base_y), xytext=(center_x + base_w/2, base_y),
                    arrowprops=dict(arrowstyle='<->', color=width_color, lw=1.5),
                    zorder=TEXT_ZORDER)
    ax_main.text(center_x, base_y - 3, f'w₀ = {BASE_WIDTH_RATIO:.1f}W',
                ha='center', va='top', fontsize=8, color=width_color, family='serif',
                fontweight='bold', zorder=TEXT_ZORDER)

    # Slope annotation
    mid_y_slope = canvas_h * 0.6
    mid_x_left = center_x - (base_w + 2 * k * mid_y_slope) / 2

    arrow_start_y = mid_y_slope - 15
    arrow_end_y = mid_y_slope + 15
    arrow_start_x = center_x - (base_w + 2 * k * arrow_start_y) / 2
    arrow_end_x = center_x - (base_w + 2 * k * arrow_end_y) / 2

    ax_main.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
                    arrowprops=dict(arrowstyle='->', color=slope_color, lw=1.5),
                    zorder=TEXT_ZORDER)

    ax_main.text(mid_x_left - 15, mid_y_slope, f'k = {ENTRAINMENT_SLOPE}',
                ha='right', va='center', fontsize=8, color=slope_color,
                family='serif', fontweight='bold', zorder=TEXT_ZORDER)

    # Result count + void fraction display
    result_y = canvas_w_y - 12
    actual_count = len(placed_bubbles)
    target_count = MAX_BUBBLES_PER_IMAGE
    actual_vf = roi_acc_area / roi_area if roi_area > 0 else 0
    vf_stopped = (roi_area > 0 and actual_vf >= TARGET_VOID_FRACTION and actual_count < target_count)

    if vf_stopped:
        # Stopped by void fraction target (normal termination)
        count_text = f'n = {actual_count} bubbles  |  VF = {actual_vf:.3f} (target {TARGET_VOID_FRACTION:.3f} reached)'
        count_color = '#2ecc71'
        ax_main.text(canvas_w/2, result_y, count_text,
                    ha='center', fontsize=9, fontweight='bold', color=count_color,
                    family='serif', zorder=TEXT_ZORDER)
    elif actual_count < target_count:
        # Red warning if neither target was reached
        count_text = f'n = {actual_count}/{target_count} bubbles  |  VF = {actual_vf:.3f}'
        count_color = '#e74c3c'
        ax_main.text(canvas_w/2, result_y, count_text,
                    ha='center', fontsize=10, fontweight='bold', color=count_color,
                    family='serif', zorder=TEXT_ZORDER)
        warning_y = result_y - 8
        ax_main.text(canvas_w/2, warning_y,
                    'Check parameters: overlap_control, BASE_PIXEL_SIZE, or canvas size',
                    ha='center', fontsize=8, color=count_color, style='italic',
                    family='serif', zorder=TEXT_ZORDER)
    else:
        # Max bubble count reached
        count_text = f'n = {actual_count} bubbles  |  VF = {actual_vf:.3f}'
        count_color = '#333333'
        ax_main.text(canvas_w/2, result_y, count_text,
                    ha='center', fontsize=10, fontweight='bold', color=count_color,
                    family='serif', zorder=TEXT_ZORDER)

    # Adjust limits
    ax_main.set_xlim(-35, canvas_w + 20)
    ax_main.set_ylim(result_y - 8, formula_y + 10)
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    # Console summary
    print(f"  Bubbles placed: {actual_count}  |  Void fraction: {actual_vf:.4f} (target: {TARGET_VOID_FRACTION})")
    if vf_stopped:
        print(f"  Stopped: void fraction target reached.")
    elif actual_count < target_count:
        print(f"\n{'='*60}")
        print(f"WARNING: Target bubble count not reached!")
        print(f"{'='*60}")
        print(f"  Target:  {target_count} bubbles")
        print(f"  Actual:  {actual_count} bubbles")
        print(f"  Missing: {target_count - actual_count} bubbles ({(1 - actual_count/target_count)*100:.1f}% short)")
        print(f"\n  Possible solutions:")
        print(f"  • Increase overlap_control (current: {overlap_control})")
        print(f"  • Decrease BASE_PIXEL_SIZE (current: {BASE_PIXEL_SIZE})")
        print(f"  • Increase canvas size (current: {CANVAS_WIDTH}x{CANVAS_HEIGHT})")
        print(f"  • Decrease max_bubbles_per_image target")
        print(f"{'='*60}\n")

    # Save
    save_path = os.path.join(ILLUSTRATION_DIR, "unified_parameter_diagram.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def generate_all_illustrations():
    print("=" * 60)
    print("BubbleGen Parameter Illustration Generator")
    print(f"Current distribution type: {DIST_TYPE}")
    print("=" * 60)

    print("\n1. Generating unified parameter diagram...")
    plot_unified_parameter_diagram()

    print("\n2. Generating distribution comparison...")
    plot_distribution_comparison()

    print("\n" + "=" * 60)
    print(f"All illustrations saved to: {ILLUSTRATION_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_illustrations()