
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.features import extract_all_features
from src.fusion import build_feature_matrix, normalize_features, adjust_weights
from src.segment_kmeans import segment_pipeline


def compare_k_values(input_path, k_values=[3, 4, 5, 6], texture_weight=0.5, save_path=None):
    
    
    print("\n" + "="*60)
    print(" K-VALUE COMPARISON")
    print("="*60)
    
    # Load image
    print(f"\n Loading image: {input_path}")
    image_bgr = cv2.imread(input_path)
    
    if image_bgr is None:
        print(f" ERROR: Could not load image from {input_path}")
        return
    
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image_bgr.shape[1]} x {image_bgr.shape[0]} pixels")
    
    # Extract features ONCE (same for all K values)
    print(f"\n Extracting features...")
    hsv_features, lbp_features = extract_all_features(image_bgr)
    
    weights = adjust_weights(texture_importance=texture_weight)
    feature_matrix, original_shape = build_feature_matrix(
        lbp_features, hsv_features, weights=weights
    )
    feature_matrix = normalize_features(feature_matrix)
    print(f"Features ready!")
    
    # Segment with different K values
    print(f"\n Segmenting with different K values: {k_values}")
    
    segmentations = []
    for k in k_values:
        print(f"   Processing K={k}...")
        _, colored_seg = segment_pipeline(feature_matrix, original_shape, k=k)
        segmentations.append(colored_seg)
    
    print(f" All segmentations complete!")
    
    # Create comparison visualization
    print(f"\n Creating comparison visualization...")
    
    num_results = len(k_values) + 1  # +1 for original
    fig, axes = plt.subplots(1, num_results, figsize=(5*num_results, 5))
    
    # Show original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Show each segmentation
    for i, (k, seg) in enumerate(zip(k_values, segmentations)):
        axes[i+1].imshow(seg)
        axes[i+1].set_title(f"K={k} regions", fontsize=14, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.suptitle(
        f"Segmentation Comparison: Different K Values\n(Texture weight: {texture_weight})",
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
    print("K-VALUE COMPARISON COMPLETE!")
    print("="*60)
    print("\n Observations:")
    print("   • Lower K = Broader, simpler regions")
    print("   • Higher K = More detailed, finer regions")
    print("   • Find the K that best captures your image structure!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare segmentation with different K values"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--k_values",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6],
        help="List of K values to compare (default: 3 4 5 6)"
    )
    
    parser.add_argument(
        "--texture_weight",
        type=float,
        default=0.5,
        help="Texture weight (default: 0.5)"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        default="data/output/k_comparison.png",
        help="Where to save comparison (default: data/output/k_comparison.png)"
    )
    
    args = parser.parse_args()
    
    compare_k_values(
        input_path=args.input,
        k_values=args.k_values,
        texture_weight=args.texture_weight,
        save_path=args.save
    )