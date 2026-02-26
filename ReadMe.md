# PlumeDEBuG

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Plume-Data Empowered Bubble Generator (PlumeDEBuG)**
Fast synthetic plume generation with high-quality data.
.

<div align="center">
  <img src="Generator/assets/thumbnail_EFDL_logo_black.png" alt="PlumeDEBuG Mode Type" width="200">
  <p><em>Environment Fluid Dynamics Lab</em></p>
</div>

## Feature

<div align="center">
  <img src="Generator/assets/demo.gif" alt="PlumeDEBuG Bubble Generation Demo" width="600">
  <p><em>Example of synthetic bubble plume generation</em></p>
</div>


## Quick Start

### Installation

```bash
git clone https://github.com/YourUsername/PlumeDEBuG.git
cd PlumeDEBuG

# Fast setup (recommended)
cd Generator
./setup.bat

# Or setup environment manually
conda env create -f environment/environment.yml
conda activate PlumeDEBuG
```

### Usage

```bash
cd Generator

# Generate images
python bubble_gen.py

# Generate parameter illustrations
python bubble_gen_illustration.py
```

### Core Parameters

```ini
[General]
num_synthetic_images = 100        # Images to generate
target_void_fraction = 0.15       # Bubble coverage (0-1)
max_bubbles_per_image = 200       # Max bubbles per image

[Distribution]
distribution_type = gaussian      # gaussian | bimodal | weibull | lognormal | constant
selection_method = weighted_sampling  # weighted_sampling (default) | direct_pdf

# Sample distribution:

[Gaussian]
mu = 0.003                        # Mean diameter (m)
sigma = 0.001                     # Std deviation (m)

[Bimodal]
mu1 = 0.002                       # Mode 1 mean (m)
sigma1 = 0.0005                   # Mode 1 std (m)
mu2 = 0.005                       # Mode 2 mean (m)
sigma2 = 0.001                    # Mode 2 std (m)
weight1 = 0.6                     # Mode 1 weight (0-1)

[Placement]
placement_mode = Guassian         # Guassian | Random
overlap_control = 0.3             # Max overlap ratio (0-1)
entrainment_slope = 0.1           # Trapezoid expansion rate
base_width_ratio = 0.3            # Bottom width / canvas width

```

## Project Structure

```
PlumeDEBuG/
├── Generator/
│   ├── bubble_gen.py              # Main generator
│   ├── bubble_gen_illustration.py # Parameter visualizer
│   ├── config.ini                 # Config file
│   ├── bubble_cache.pkl           # Cached bubble database
│   ├── output/                    # Generated images (auto-numbered runs)
│   │   ├── run0/
│   │   │   ├── synth_0000.png/json
│   │   │   ├── execution_log.txt
│   │   │   ├── config.ini
│   │   │   ├── roi_visualization.png
│   │   │   ├── bubble_diameter_histogram.png/pdf
│   │   │   └── mixed_params_batch*.json (if mixed mode enabled)
│   │   └── run1/
│   └── parameter_illustrations/   # Auto-generated diagrams
│       ├── unified_parameter_diagram.png/pdf
│       ├── size_distribution.png/pdf
│       └── distribution_comparison.png/pdf
├── tools/                         # Database preprocessing tools
│   ├── mat_to_sqlite.py           # .mat → SQLite converter
│   ├── combine_datasets.m         # Merge multiple datasets
│   ├── sync_database.m            # Sync .mat/crops/masks
│   ├── filter_size_bins.m         # Balance size distribution
│   ├── plot_distribution.m        # Visualize distribution
│   └── ...                        # (See Database Tools section)
└── README.md
```

## Selection Methods

| Method | Accuracy | Description |
|--------|----------|-------------|
| `weighted_sampling` | 90-95% | Balances distribution matching with data availability.  |
| `direct_pdf` | ~100% | Pure distribution matching, ignores bubble counts per bin. May over-sample rare bins. |

## Mixed Mode (Optional)

When `enable_mixed_mode = True`, randomizes every 10% of images:
- Bubble count: `max_bubbles ± 50`
- Placement mode: randomly Gaussian or Random
- Parameters: `overlap_control`, `entrainment_slope`, `base_width_ratio` vary by ±30%
- Saves parameters to `mixed_params_batch*.json`

## Velocity-Based Size Bias (Optional)

When `enable_velocity_size_bias = True`:
- High-velocity center: can carry large bubbles
- Low-velocity edges: only small bubbles allowed
- Physics: bubble size ∝ velocity^coupling_strength

## Overlap Control

Unidirectional method: `R = A_intersection / A_existing`
- Protects existing bubbles from occlusion
- Rejects new bubble if any existing bubble is >30% covered (default threshold)
- Quadtree acceleration for collision detection

## Complete Workflow

**Quick Reference:**
```
New Images → Validate → Compute Metrics → Merge → Balance → SQLite → Generate
  (Step 1)    (Step 2)     (Step 3)      (Step 4)  (Step 5)  (Step 6)   (Step 7)
```

### Step 1: Add New Images to Dataset

**Required folder structure for each dataset:**
download the public dataset at [zenodo](https://zenodo.org/records/18793954).
```
dataset/
└── <folder_name>/              # e.g., "044", "601"
    ├── Dataset/
    │   ├── Cropped/
    │   │   └── <prefix>_cropped.tif
    │   └── Masks/
    │       └── <prefix>_mask.tif
    └── aggregated_results.mat  # Contains imgInfo struct array
```

**Add new bubble images:**
1. Create new folder: `dataset/701/`
2. Create subfolders: `Dataset/Cropped/` and `Dataset/Masks/`
3. Add images:
   - `Dataset/Cropped/bubble001_cropped.tif`
   - `Dataset/Masks/bubble001_mask.tif`
4. Create `aggregated_results.mat` in MATLAB:

```matlab
% Example: Create imgInfo struct array
imgInfo(1).imageName = 'bubble001';          % No extension
imgInfo(1).bubble_diameter = 0.0035;         % Diameter in meters (3.5mm)
imgInfo(2).imageName = 'bubble002';
imgInfo(2).bubble_diameter = 0.0028;

% Save to mat file
save('dataset/701/aggregated_results.mat', 'imgInfo');
```

**Required fields:**
- `imageName`: filename prefix (no `.tif` extension)
- `bubble_diameter`: diameter in meters

### Step 2: Validate Single Dataset

```matlab
% In MATLAB, set baseFolder to your dataset
cd tools

% Check file correspondence
% Edit check_file_correspondence.m: baseFolder = '../dataset/701'
check_file_correspondence

% Sync database (removes orphans)
% Edit sync_database.m: baseFolder = '../dataset/701'
sync_database

% Optional: Balance crops/masks
balance_crops  % or balance_masks
```

### Step 3: Compute Bubble Metrics

```matlab
% Batch compute ellipse fits for all bubbles
% Edit compute_ellipse_fits.m: add '701' to sourceDirs
compute_ellipse_fits
```

### Step 4: Merge Multiple Datasets

```matlab
% Combine all datasets into one aggregated file
% Edit combine_datasets.m: add folder names to sourceDirs
% Example: sourceDirs = {'044','601','701'};
combine_datasets

% Output: ../Aggregated_bubble_data/aggregated_results.mat
```

### Step 5: Balance Distribution (Optional)

```matlab
% Filter size bins to reduce peaks
% Edit filter_size_bins.m: configure targetRange and targetBinCount
filter_size_bins

% Visualize distribution
plot_distribution
```

### Step 6: Convert to SQLite

```bash
cd tools
python mat_to_sqlite.py

# Input:  ../Aggregated_bubble_data/aggregated_results.mat
# Output: ../Aggregated_bubble_data/aggregated_results.db
```

### Step 7: Generate Synthetic Images

```bash
cd Generator

# Update config.ini paths:
# [Database]
# database_path = ../Aggregated_bubble_data
# aggregated_results_path = ../Aggregated_bubble_data/aggregated_results.db

# Generate images
python bubble_gen.py

# Generate parameter diagrams
python bubble_gen_illustration.py

# Output: Generator/output/run0/
```

## Troubleshooting

**Problem: "imgInfo variable not found"**
- Solution: Ensure .mat file contains `imgInfo` struct (not `data` or other names)

**Problem: "Cropped/Mask file mismatch"**
- Solution: Run `sync_database.m` to remove orphaned files

**Problem: "Target bubble count not reached in illustration"**
- Solution: Increase `overlap_control`, decrease `BASE_PIXEL_SIZE`, or increase canvas size in `bubble_gen_illustration.py`

**Problem: "K-S test p-value too low"**
- Solution: Check `filter_size_bins.m` settings or use `direct_pdf` selection method

**Problem: MATLAB .mat file version error in Python**
- Solution: `mat_to_sqlite.py` handles both v7 and v7.3 formats automatically

## Database Tools Reference

| Tool | Language | Function |
|------|----------|----------|
| `mat_to_sqlite.py` | Python | Convert MATLAB .mat files to SQLite database |
| `combine_datasets.m` | MATLAB | Merge multiple dataset folders into single aggregated_results.mat |
| `sync_database.m` | MATLAB | Synchronize .mat file, Cropped/, and Masks/ folders (removes orphans) |
| `check_file_correspondence.m` | MATLAB | Verify 1-to-1 correspondence between .mat, crops, and masks |
| `clean_orphaned_files.m` | MATLAB | Delete files in Cropped/Masks not referenced in .mat |
| `clean_empty_rows.m` | MATLAB | Remove empty rows from aggregated_results.mat |
| `balance_crops.m` | MATLAB | Remove masks without matching crops |
| `balance_masks.m` | MATLAB | Remove crops without matching masks |
| `filter_size_bins.m` | MATLAB | Proportionally reduce bubble counts in specified diameter bins |
| `compute_ellipse_fits.m` | MATLAB | Batch compute ellipse fits for all masks in dataset folders |
| `plot_distribution.m` | MATLAB | Plot bubble diameter distribution histogram (Nature-style) |

## Example output

<div align="center">
  <img src="Generator/assets/SynImg.png" alt="PlumeDEBuG Mode Type" width="600">
  <p><em>Example of PlumeDEBuG's output image and labels</em></p>
</div>

<div align="center">
  <img src="Generator/assets/output_distribution.png" alt="PlumeDEBuG Mode Type" width="600">
  <p><em>Example of PlumeDEBuG's output distribution</em></p>
</div>

<div align="center">
  <img src="Generator/assets/Mode.png" alt="PlumeDEBuG Mode Type" width="600">
  <p><em>Example of PlumeDEBuG's Mode type</em></p>
</div>

## Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push: `git push origin feature/name`
5. Open Pull Request

## Demo


## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Xuchen (Schuetzen) Ying** - xuchen.ying@mail.missouri.edu  
Project: https://github.com/Schuetzen/PlumeDEBuG

**Dr. Binbin Wang** -  wangbinb@umsystem.edu