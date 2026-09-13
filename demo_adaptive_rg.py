"""
SIMPLE DEMO: Run Your Novel Adaptive Region Growing Algorithm
==============================================================
This is a simple script to test your proposed algorithm independently.
"""

import os
import argparse
import cv2
import sys

# Add paths
sys.path.append('/mnt/user-data/uploads')
sys.path.append('/home/claude')

from features import extract_all_features
from fusion import build_feature_matrix, normalize_features, adjust_weights
from adaptive_region_growing import adaptive_region_growing_pipeline
import matplotlib.pyplot as plt
import numpy as np


def demo_adaptive_region_growing(input_path, output_dir, num_regions=5, 
                                 texture_weight=0.5, initial_threshold=0.15):
    """
    Demo function to run and visualize your novel algorithm.
    """
    print("\n" + "="*70)
    print(" ADAPTIVE REGION GROWING DEMO")
    print("   Your Novel Segmentation Algorithm")
    print("="*70)
    
    # Load image
    print(f"\n Loading image: {input_path}")
    
    if not os.path.exists(input_path):
        print(f" ERROR: Image not found")
        return
    
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f" ERROR: Could not read image")
        return
    
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {image_bgr.shape[1]} x {image_bgr.shape[0]} pixels")
    
    # Extract features
    print(f"\n Extracting texture-color features...")
    hsv_features, lbp_features = extract_all_features(image_bgr)
    print(f"✓ Features extracted")
    
    # Build feature matrix
    print(f"\n Building feature matrix...")
    weights = adjust_weights(texture_importance=texture_weight)
    feature_matrix, original_shape = build_feature_matrix(
        lbp_features, hsv_features, weights=weights
    )
    feature_matrix = normalize_features(feature_matrix)
    print(f"✓ Feature matrix ready: {feature_matrix.shape}")
    
    # Run your novel algorithm
    print(f"\n Running Adaptive Region Growing Algorithm...")
    print(f"   Parameters:")
    print(f"   • Number of regions: {num_regions}")
    print(f"   • Initial threshold: {initial_threshold}")
    print(f"   • Texture weight: {texture_weight}")
    
    label_image, colored_seg = adaptive_region_growing_pipeline(
        feature_matrix=feature_matrix,
        original_shape=original_shape,
        num_regions=num_regions,
        initial_threshold=initial_threshold,
        texture_weight=texture_weight
    )
    
    # Create visualization
    print(f"\n Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation result
    axes[1].imshow(colored_seg)
    axes[1].set_title(
        f"Adaptive Region Growing Result\n({len(np.unique(label_image))} regions)",
        fontsize=14,
        fontweight='bold',
        color='darkgreen'
    )
    axes[1].axis('off')
    
    plt.suptitle(
        "Your Novel Adaptive Region Growing Algorithm",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    save_path = os.path.join(output_dir, f"{name_without_ext}_adaptive_rg.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved result to: {save_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print(" DEMO COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: Adaptive Region Growing Algorithm"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/output",
        help="Output directory (default: data/output)"
    )
    
    parser.add_argument(
        "--regions",
        type=int,
        default=5,
        help="Number of regions (default: 5)"
    )
    
    parser.add_argument(
        "--texture_weight",
        type=float,
        default=0.5,
        help="Texture weight 0-1 (default: 0.5)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Initial similarity threshold (default: 0.15)"
    )
    
    args = parser.parse_args()
    
    demo_adaptive_region_growing(
        input_path=args.input,
        output_dir=args.output,
        num_regions=args.regions,
        texture_weight=args.texture_weight,
        initial_threshold=args.threshold
    )