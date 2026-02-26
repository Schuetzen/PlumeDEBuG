"""
Simple Bubble Visualization - Multiple Styles
==============================================

Minimal visualization script for quick inspection of bubble images.
Supports three visualization modes with different color schemes.

Modes:
    1. multi-color: Vibrant colors, each bubble different (default)
    2. mask-only: Soft macaron colors, no bounding boxes
    3. macaron: Unified soft pink masks + blue borders

Features:
    - Multiple color palettes (vibrant or soft macaron style)
    - Semi-transparent mask fills (50% opacity)
    - Optional bounding boxes
    - Publication-quality output (300 DPI)

Usage:
    # Generate both mask-only and macaron images
    python visualize_simple.py output/run0 0 --both

    # Single mode
    python visualize_simple.py output/run0 0 --mode mask-only
    python visualize_simple.py output/run0 0 --mode macaron

    # Custom output path
    python visualize_simple.py output/run0 0 --mode macaron -o result.png

Author: Claude + Xuchen
Version: 1.3 (Multi-Style Support)
"""

import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Publication quality settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0


def visualize_overlay(run_dir, image_id, save_path=None, show=False, mode='multi-color'):
    """
    Simple overlay visualization: original image + bounding boxes + semi-transparent masks

    Parameters:
        run_dir: Path to run directory (e.g., 'output/run0')
        image_id: Image ID (e.g., 0)
        save_path: Output path (optional)
        show: Show figure interactively (default: False)
        mode: Visualization mode
              'multi-color' - Multi-color masks with matching borders (default)
              'mask-only' - Multi-color masks without borders
              'macaron' - Soft macaron-style: pink masks + blue borders
    """
    run_path = Path(run_dir)

    # Load image
    img_path = run_path / f"synth_{image_id:04d}.png"
    if not img_path.exists():
        print(f"❌ Error: Image not found: {img_path}")
        return None

    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # Load annotation
    json_path = run_path / f"synth_{image_id:04d}.json"
    if not json_path.exists():
        print(f"❌ Error: Annotation not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        annotation = json.load(f)

    # Print info to console only
    num_bubbles = annotation['num_bubbles']
    void_fraction = annotation['void_fraction']
    print(f"Image ID: {image_id}")
    print(f"Bubbles: {num_bubbles}")
    print(f"Void Fraction: {void_fraction:.2%}")

    # Create RGB overlay image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = image_rgb.copy()

    boxes = annotation['boxes']
    masks = annotation.get('masks', None)

    # Color palettes for different modes
    if mode == 'mask-only':
        # Multi-color palette for mask-only mode (enhanced visibility)
        # Brighter macaron colors with better contrast on grayscale
        color_palette = [
            np.array([255, 105, 180], dtype=np.uint8),  # Hot Pink
            np.array([255, 160, 122], dtype=np.uint8),  # Light Salmon
            np.array([186, 85, 211], dtype=np.uint8),   # Medium Orchid
            np.array([135, 206, 250], dtype=np.uint8),  # Light Sky Blue
            np.array([255, 182, 193], dtype=np.uint8),  # Light Pink
            np.array([221, 160, 221], dtype=np.uint8),  # Plum
            np.array([255, 192, 203], dtype=np.uint8),  # Pink
            np.array([255, 215, 0], dtype=np.uint8),    # Gold
            np.array([173, 216, 230], dtype=np.uint8),  # Light Blue
            np.array([240, 128, 128], dtype=np.uint8),  # Light Coral
            np.array([144, 238, 144], dtype=np.uint8),  # Light Green
            np.array([255, 218, 185], dtype=np.uint8),  # Peach Puff
        ]
        draw_bbox = False
    elif mode == 'macaron':
        # Macaron style: soft pink masks with blue borders (all same color)
        mask_color_macaron = np.array([255, 182, 193], dtype=np.uint8)  # Light Pink
        border_color_macaron = (230, 180, 140)  # Soft Blue (BGR: 140, 180, 230)
        draw_bbox = True
    else:
        # 'multi-color' mode: vibrant colors with matching borders
        color_palette = [
            np.array([255, 0, 0], dtype=np.uint8),      # Red
            np.array([0, 255, 0], dtype=np.uint8),      # Green
            np.array([0, 0, 255], dtype=np.uint8),      # Blue
            np.array([255, 255, 0], dtype=np.uint8),    # Yellow
            np.array([255, 0, 255], dtype=np.uint8),    # Magenta
            np.array([0, 255, 255], dtype=np.uint8),    # Cyan
            np.array([255, 128, 0], dtype=np.uint8),    # Orange
            np.array([128, 0, 255], dtype=np.uint8),    # Purple
            np.array([0, 255, 128], dtype=np.uint8),    # Spring Green
            np.array([255, 0, 128], dtype=np.uint8),    # Rose
            np.array([128, 255, 0], dtype=np.uint8),    # Chartreuse
            np.array([0, 128, 255], dtype=np.uint8),    # Sky Blue
        ]
        draw_bbox = True

    # Draw masks if available
    if masks is not None:
        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            x, y, w, h = box
            mask_array = np.array(mask, dtype=np.uint8)

            # Select color based on mode
            if mode == 'macaron':
                mask_color = mask_color_macaron
                border_color = border_color_macaron
            else:
                # Select color from palette (cycle if more bubbles than colors)
                mask_color = color_palette[idx % len(color_palette)]
                # Border uses same color as mask (convert RGB to BGR for OpenCV)
                border_color = (int(mask_color[2]), int(mask_color[1]), int(mask_color[0]))

            # Create colored mask with transparency
            mask_3d = np.stack([mask_array] * 3, axis=-1)
            colored_mask = np.zeros_like(mask_3d, dtype=np.uint8)
            colored_mask[mask_3d > 0] = np.tile(mask_color, (np.sum(mask_3d > 0) // 3,))

            # Blend with overlay (50% transparency for softer look)
            roi = overlay[y:y+h, x:x+w]
            mask_bool = mask_array > 0
            roi[mask_bool] = cv2.addWeighted(roi[mask_bool], 0.5,
                                            colored_mask[mask_bool], 0.5, 0)

            # Draw bounding box (if enabled)
            if draw_bbox:
                cv2.rectangle(overlay, (x, y), (x+w, y+h), border_color, 2)

    # Blend overlay with original image
    result = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    # Create figure without any text
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   pad_inches=0, facecolor='white')
        print(f"✓ Saved: {save_path}")
    elif not show:
        # Default save location
        default_path = run_path / f"vis_overlay_{image_id:04d}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight',
                   pad_inches=0, facecolor='white')
        print(f"✓ Saved: {default_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Simple bubble visualization with multiple style modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both images (mask-only + macaron)
  python visualize_simple.py output/run0 0 --both

  # Single mode visualization
  python visualize_simple.py output/run0 0 --mode mask-only
  python visualize_simple.py output/run0 0 --mode macaron
  python visualize_simple.py output/run0 0 --mode multi-color

  # Custom output
  python visualize_simple.py output/run0 5 --mode macaron -o result.png

  # Interactive display
  python visualize_simple.py output/run0 0 --mode mask-only --show
        """
    )

    parser.add_argument('run_dir', type=str,
                       help='Path to run directory (e.g., output/run0)')
    parser.add_argument('image_id', type=int,
                       help='Image ID (e.g., 0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (optional)')
    parser.add_argument('--show', action='store_true',
                       help='Show figure interactively')
    parser.add_argument('--mode', '-m', type=str, default='multi-color',
                       choices=['multi-color', 'mask-only', 'macaron'],
                       help='Visualization mode: multi-color (default), mask-only, or macaron')
    parser.add_argument('--both', action='store_true',
                       help='Generate both mask-only and macaron images')

    args = parser.parse_args()

    # Validate run directory
    if not Path(args.run_dir).exists():
        print(f"❌ Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Simple Bubble Visualization (Overlay)")
    print("=" * 60)
    print(f"Run: {args.run_dir}")
    print(f"Image ID: {args.image_id}")
    print("=" * 60)

    # Generate both images if --both flag is used
    if args.both:
        run_path = Path(args.run_dir)

        # Image 1: Mask-only (multi-color masks, no borders)
        print("\n[1/2] Generating mask-only visualization...")
        output1 = run_path / f"vis_mask_only_{args.image_id:04d}.png"
        visualize_overlay(args.run_dir, args.image_id, str(output1), False, mode='mask-only')

        # Image 2: Macaron style (pink masks + blue borders)
        print("\n[2/2] Generating macaron-style visualization...")
        output2 = run_path / f"vis_macaron_{args.image_id:04d}.png"
        visualize_overlay(args.run_dir, args.image_id, str(output2), False, mode='macaron')
    else:
        # Single mode visualization
        visualize_overlay(args.run_dir, args.image_id, args.output, args.show, mode=args.mode)

    print("=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
