
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.features import extract_all_features
from src.fusion import build_feature_matrix, normalize_features, adjust_weights
from src.segment_kmeans import segment_pipeline


def compare_texture_color_weights(input_path, k=5, weights=[0.0, 0.3, 0.5, 0.7, 1.0], save_path=None):
   
    
    print("\n" + "="*60)
    print(" TEXTURE vs COLOR COMPARISON")
    print("="*60)
    
    # Load image
    print(f"\n Loading image: {input_path}")
    image_bgr = cv2.imread(input_path)
    
    if image_bgr is None:
        print(f" ERROR: Could not load image from {input_path}")
        return
    
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f" Image loaded: {image_bgr.shape[1]} x {image_bgr.shape[0]} pixels")
    
    # Extract features ONCE
    print(f"\nExtracting features...")
    hsv_features, lbp_features = extract_all_features(image_bgr)
    print(f"Features ready!")
    
    # Segment with different weights
    print(f"\n Segmenting with K={k} and different texture/color weights...")
    
    segmentations = []
    for weight in weights:
        print(f"   Processing texture_weight={weight:.1f}...")
        
        # Adjust weights for this iteration
        w = adjust_weights(texture_importance=weight)
        feature_matrix, original_shape = build_feature_matrix(
            lbp_features, hsv_features, weights=w
        )
        feature_matrix = normalize_features(feature_matrix)
        
        # Segment
        _, colored_seg = segment_pipeline(feature_matrix, original_shape, k=k)
        segmentations.append(colored_seg)
    
    print(f"All segmentations complete!")
    
    # Create comparison visualization
    print(f"\nCreating comparison visualization...")
    
    num_results = len(weights) + 1  # +1 for original
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show each segmentation
    for i, (weight, seg) in enumerate(zip(weights, segmentations)):
        idx = i + 1
        axes[idx].imshow(seg)
        
        # Create descriptive title
        if weight == 0.0:
            title = f"Pure COLOR Only\n(texture={weight:.1f}, color={1-weight:.1f})"
        elif weight == 1.0:
            title = f"Pure TEXTURE Only\n(texture={weight:.1f}, color={1-weight:.1f})"
        elif weight == 0.5:
            title = f"BALANCED Fusion \n(texture={weight:.1f}, color={1-weight:.1f})"
        else:
            title = f"Texture={weight:.1f}, Color={1-weight:.1f}"
        
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra subplot if we have one
    if len(axes) > num_results:
        axes[-1].axis('off')
    
    plt.suptitle(
        f"Texture-Color Fusion Comparison (K={k} segments)",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("TEXTURE-COLOR COMPARISON COMPLETE!")
    print("="*60)
    print("\n Observations:")
    print("   • Pure COLOR (0.0): Groups by color only, ignores texture")
    print("   • Pure TEXTURE (1.0): Groups by texture only, ignores color")
    print("   • BALANCED (0.5): Best of both worlds! ")
    print("\n   Notice how the balanced approach often gives the most")
    print("   meaningful segmentation by considering BOTH properties!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare segmentation with different texture/color weights"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of segments (default: 5)"
    )
    
    parser.add_argument(
        "--weights",
        type=float,
        nargs='+',
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help="List of texture weights to compare (default: 0.0 0.3 0.5 0.7 1.0)"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        default="data/output/texture_color_comparison.png",
        help="Where to save comparison (default: data/output/texture_color_comparison.png)"
    )
    
    args = parser.parse_args()
    
    compare_texture_color_weights(
        input_path=args.input,
        k=args.k,
        weights=args.weights,
        save_path=args.save
    )