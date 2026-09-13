"""
COMPARISON FRAMEWORK: Novel Algorithm vs Baseline Methods
==========================================================
This script compares:
1. YOUR PROPOSED: Adaptive Region Growing (Novel)
2. BASELINE: K-Means Clustering (Existing)
3. Pure Color Segmentation
4. Pure Texture Segmentation

This demonstrates the advantages of your proposed method!
"""

import os
import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.features import extract_all_features
from src.fusion import build_feature_matrix, normalize_features, adjust_weights
from src.segment_kmeans import segment_pipeline
from adaptive_region_growing import adaptive_region_growing_pipeline


def evaluate_segmentation_quality(label_image, feature_image):
    """
    Compute quantitative metrics for segmentation quality.

    Metrics:
    --------
    1. Region Compactness: How compact/circular are the regions?
    2. Intra-region Homogeneity: How similar are pixels within each region?
    3. Inter-region Heterogeneity: How different are adjacent regions?
    4. Number of regions

    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    metrics = {}

    # 1. Number of regions
    unique_regions = np.unique(label_image)
    num_regions = len(unique_regions)
    metrics['num_regions'] = num_regions

    # 2. Compute intra-region homogeneity (average std within regions)
    intra_region_std = []
    region_sizes = []

    for region_id in unique_regions:
        mask = (label_image == region_id)
        region_features = feature_image[mask]

        if len(region_features) > 1:
            # Compute standard deviation of features within region
            std = np.std(region_features, axis=0).mean()
            intra_region_std.append(std)
            region_sizes.append(len(region_features))

    if len(intra_region_std) > 0:
        metrics['intra_region_homogeneity'] = 1.0 / (np.mean(intra_region_std) + 1e-6)
    else:
        metrics['intra_region_homogeneity'] = 0.0

    if len(region_sizes) > 0:
        metrics['avg_region_size'] = np.mean(region_sizes)
        metrics['std_region_size'] = np.std(region_sizes)
    else:
        metrics['avg_region_size'] = 0.0
        metrics['std_region_size'] = 0.0

    # 3. Compute boundary strength (how different are adjacent regions?)
    boundary_strength = compute_boundary_strength(label_image, feature_image)
    metrics['boundary_strength'] = boundary_strength

    # 4. Compute region compactness
    compactness = compute_region_compactness(label_image)
    metrics['compactness'] = compactness

    return metrics


def compute_boundary_strength(label_image, feature_image):
    """
    Compute average feature difference across region boundaries.
    Higher = better segmentation (regions are very different from each other)
    """
    height, width = label_image.shape
    boundary_diffs = []

    # Check horizontal boundaries
    for i in range(height):
        for j in range(width - 1):
            if label_image[i, j] != label_image[i, j + 1]:
                # Boundary detected
                diff = np.linalg.norm(feature_image[i, j] - feature_image[i, j + 1])
                boundary_diffs.append(diff)

    # Check vertical boundaries
    for i in range(height - 1):
        for j in range(width):
            if label_image[i, j] != label_image[i + 1, j]:
                # Boundary detected
                diff = np.linalg.norm(feature_image[i, j] - feature_image[i + 1, j])
                boundary_diffs.append(diff)

    if len(boundary_diffs) == 0:
        return 0.0

    return np.mean(boundary_diffs)


def compute_region_compactness(label_image):
    """
    Compute compactness: 4*pi*Area / Perimeter^2
    Perfect circle = 1.0, less compact shapes < 1.0
    """
    unique_regions = np.unique(label_image)
    compactness_scores = []

    for region_id in unique_regions:
        mask = (label_image == region_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0 and area > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                compactness_scores.append(compactness)

    if len(compactness_scores) == 0:
        return 0.0

    return np.mean(compactness_scores)


def run_all_methods(image_bgr, k=5, texture_weight=0.5):
    """
    Run all segmentation methods for comparison.

    Returns:
    --------
    results : dict
        Dictionary containing results from all methods
    """
    print("\n" + "="*70)
    print(" RUNNING ALL SEGMENTATION METHODS FOR COMPARISON")
    print("="*70)

    results = {}

    # Extract features ONCE (same for all methods)
    print("\n Extracting features...")
    hsv_features, lbp_features = extract_all_features(image_bgr)

    height, width = image_bgr.shape[:2]
    feature_image = np.concatenate([
        lbp_features.reshape(height, width, 1),
        hsv_features
    ], axis=2)

    # ========================================
    # METHOD 1: K-MEANS (BASELINE)
    # ========================================
    print("\n" + "-"*70)
    print("1️  K-MEANS CLUSTERING (Baseline)")
    print("-"*70)

    weights = adjust_weights(texture_importance=texture_weight)
    feature_matrix, original_shape = build_feature_matrix(
        lbp_features, hsv_features, weights=weights
    )
    feature_matrix = normalize_features(feature_matrix)

    start_time = time.time()
    kmeans_labels, kmeans_colored = segment_pipeline(feature_matrix, original_shape, k=k)
    kmeans_time = time.time() - start_time

    kmeans_metrics = evaluate_segmentation_quality(kmeans_labels, feature_image)
    kmeans_metrics['computation_time'] = kmeans_time

    results['kmeans'] = {
        'labels': kmeans_labels,
        'colored': kmeans_colored,
        'metrics': kmeans_metrics,
        'name': 'K-Means (Baseline)'
    }

    print(f"✓ K-Means complete in {kmeans_time:.2f}s")
    print(f"  Regions: {kmeans_metrics['num_regions']}")

    # ========================================
    # METHOD 2: ADAPTIVE REGION GROWING (YOUR NOVEL ALGORITHM)
    # ========================================
    print("\n" + "-"*70)
    print("2️  ADAPTIVE REGION GROWING (YOUR PROPOSED METHOD) ")
    print("-"*70)

    start_time = time.time()
    arg_labels, arg_colored = adaptive_region_growing_pipeline(
        feature_matrix,
        original_shape,
        num_regions=k,
        initial_threshold=0.15,
        texture_weight=texture_weight
    )
    arg_time = time.time() - start_time

    arg_metrics = evaluate_segmentation_quality(arg_labels, feature_image)
    arg_metrics['computation_time'] = arg_time

    results['adaptive_rg'] = {
        'labels': arg_labels,
        'colored': arg_colored,
        'metrics': arg_metrics,
        'name': 'Adaptive Region Growing (Proposed)'
    }

    print(f"✓ Adaptive Region Growing complete in {arg_time:.2f}s")
    print(f"  Regions: {arg_metrics['num_regions']}")

    # ========================================
    # METHOD 3: PURE COLOR (NO TEXTURE)
    # ========================================
    print("\n" + "-"*70)
    print("3️ PURE COLOR SEGMENTATION (No Texture)")
    print("-"*70)

    weights_color = adjust_weights(texture_importance=0.0)  # Only color
    feature_matrix_color, _ = build_feature_matrix(
        lbp_features, hsv_features, weights=weights_color
    )
    feature_matrix_color = normalize_features(feature_matrix_color)

    start_time = time.time()
    color_labels, color_colored = segment_pipeline(feature_matrix_color, original_shape, k=k)
    color_time = time.time() - start_time

    color_metrics = evaluate_segmentation_quality(color_labels, feature_image)
    color_metrics['computation_time'] = color_time

    results['pure_color'] = {
        'labels': color_labels,
        'colored': color_colored,
        'metrics': color_metrics,
        'name': 'Pure Color Only'
    }

    print(f"✓ Pure Color complete in {color_time:.2f}s")

    # ========================================
    # METHOD 4: PURE TEXTURE (NO COLOR)
    # ========================================
    print("\n" + "-"*70)
    print("4️ PURE TEXTURE SEGMENTATION (No Color)")
    print("-"*70)

    weights_texture = adjust_weights(texture_importance=1.0)  # Only texture
    feature_matrix_texture, _ = build_feature_matrix(
        lbp_features, hsv_features, weights=weights_texture
    )
    feature_matrix_texture = normalize_features(feature_matrix_texture)

    start_time = time.time()
    texture_labels, texture_colored = segment_pipeline(feature_matrix_texture, original_shape, k=k)
    texture_time = time.time() - start_time

    texture_metrics = evaluate_segmentation_quality(texture_labels, feature_image)
    texture_metrics['computation_time'] = texture_time

    results['pure_texture'] = {
        'labels': texture_labels,
        'colored': texture_colored,
        'metrics': texture_metrics,
        'name': 'Pure Texture Only'
    }

    print(f"✓ Pure Texture complete in {texture_time:.2f}s")

    print("\n" + "="*70)
    print("ALL METHODS COMPLETED!")
    print("="*70 + "\n")

    return results


def create_comparison_visualization(original_bgr, results, save_path=None):
    """
    Create a comprehensive comparison visualization.
    """
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 12))

    # Row 1: Images
    # Original
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original_rgb)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # K-Means
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(results['kmeans']['colored'])
    ax2.set_title(
        f"K-Means (Baseline)\n{results['kmeans']['metrics']['num_regions']} regions",
        fontsize=14,
        fontweight='bold'
    )
    ax2.axis('off')

    # Adaptive Region Growing (YOUR METHOD)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(results['adaptive_rg']['colored'])
    ax3.set_title(
        f" Adaptive Region Growing (PROPOSED) \n{results['adaptive_rg']['metrics']['num_regions']} regions",
        fontsize=14,
        fontweight='bold',
        color='darkgreen'
    )
    ax3.axis('off')

    # Pure Color
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(results['pure_color']['colored'])
    ax4.set_title(
        f"Pure Color Only\n{results['pure_color']['metrics']['num_regions']} regions",
        fontsize=14,
        fontweight='bold'
    )
    ax4.axis('off')

    # Pure Texture
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(results['pure_texture']['colored'])
    ax5.set_title(
        f"Pure Texture Only\n{results['pure_texture']['metrics']['num_regions']} regions",
        fontsize=14,
        fontweight='bold'
    )
    ax5.axis('off')

    # Metrics comparison table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    methods = ['K-Means', 'Proposed ARG', 'Pure Color', 'Pure Texture']
    method_keys = ['kmeans', 'adaptive_rg', 'pure_color', 'pure_texture']

    table_data = []
    for method_name, method_key in zip(methods, method_keys):
        metrics = results[method_key]['metrics']
        row = [
            method_name,
            f"{metrics['computation_time']:.2f}s",
            f"{metrics['num_regions']}",
            f"{metrics['intra_region_homogeneity']:.3f}",
            f"{metrics['boundary_strength']:.3f}",
            f"{metrics['compactness']:.3f}"
        ]
        table_data.append(row)

    table = ax6.table(
        cellText=table_data,
        colLabels=['Method', 'Time', 'Regions', 'Homogeneity↑', 'Boundary↑', 'Compact↑'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Highlight proposed method row (row index 2: Proposed ARG)
    for i in range(6):
        cell = table[(2, i)]
        cell.set_facecolor('#90EE90')
        cell.set_text_props(weight='bold')

    # Header styling
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    ax6.set_title("Quantitative Comparison", fontsize=14, fontweight='bold', pad=20)

    plt.suptitle(
        "Comprehensive Method Comparison: Adaptive Region Growing vs Baselines",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n Saved comprehensive comparison to: {save_path}")

    plt.show()


def create_detailed_comparison(original_bgr, results, save_path=None):
    """
    Create detailed side-by-side comparison with boundaries.
    """
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    method_keys = ['kmeans', 'adaptive_rg', 'pure_color', 'pure_texture']
    method_names = [
        'K-Means (Baseline)',
        ' Adaptive Region Growing (PROPOSED) ',
        'Pure Color Only',
        'Pure Texture Only'
    ]

    for idx, (method_key, method_name) in enumerate(zip(method_keys, method_names)):
        row = idx // 2
        col = idx % 2

        # Create boundary overlay
        label_image = results[method_key]['labels']
        overlay = create_boundary_overlay(original_rgb, label_image)

        axes[row, col].imshow(overlay)

        metrics = results[method_key]['metrics']
        title = f"{method_name}\n"
        title += f"Regions: {metrics['num_regions']} | "
        title += f"Time: {metrics['computation_time']:.2f}s\n"
        title += f"Homogeneity: {metrics['intra_region_homogeneity']:.3f} | "
        title += f"Boundary: {metrics['boundary_strength']:.3f}"

        if 'PROPOSED' in method_name:
            axes[row, col].set_title(
                title,
                fontsize=12,
                fontweight='bold',
                color='darkgreen',
                pad=10
            )
        else:
            axes[row, col].set_title(title, fontsize=12, fontweight='bold', pad=10)

        axes[row, col].axis('off')

    plt.suptitle(
        "Detailed Comparison with Segmentation Boundaries",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f" Saved detailed comparison to: {save_path}")

    plt.show()


def create_boundary_overlay(original_rgb, label_image, alpha=0.4):
    """
    Create overlay with segmentation boundaries on original image.
    """
    boundaries = np.zeros(label_image.shape, dtype=np.uint8)

    # Detect boundaries
    boundaries[:, :-1] |= (label_image[:, :-1] != label_image[:, 1:])
    boundaries[:-1, :] |= (label_image[:-1, :] != label_image[1:, :])

    overlay = original_rgb.copy()
    overlay[boundaries > 0] = [255, 0, 0]  # Red boundaries

    result = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    return result


def print_comparison_summary(results):
    """
    Print detailed comparison summary.
    """
    print("\n" + "="*70)
    print(" QUANTITATIVE COMPARISON SUMMARY")
    print("="*70)

    print("\n{:<30} {:>10} {:>10} {:>12} {:>10}".format(
        "Method", "Time(s)", "Regions", "Homogeneity", "Boundary"
    ))
    print("-"*70)

    method_order = ['kmeans', 'adaptive_rg', 'pure_color', 'pure_texture']

    for method_key in method_order:
        result = results[method_key]
        metrics = result['metrics']
        name = result['name']

        if 'Proposed' in name:
            marker = "⭐"
        else:
            marker = "  "

        print("{}{:<28} {:>10.2f} {:>10} {:>12.3f} {:>10.3f}".format(
            marker,
            name,
            metrics['computation_time'],
            metrics['num_regions'],
            metrics['intra_region_homogeneity'],
            metrics['boundary_strength']
        ))

    print("\n" + "="*70)
    print("🏆 KEY FINDINGS")
    print("="*70)

    proposed_metrics = results['adaptive_rg']['metrics']
    baseline_metrics = results['kmeans']['metrics']

    print("\n Adaptive Region Growing vs K-Means:")

    if baseline_metrics['intra_region_homogeneity'] > 0:
        homogeneity_improvement = (
            (proposed_metrics['intra_region_homogeneity'] - baseline_metrics['intra_region_homogeneity'])
            / baseline_metrics['intra_region_homogeneity'] * 100
        )
    else:
        homogeneity_improvement = 0.0

    if baseline_metrics['boundary_strength'] > 0:
        boundary_improvement = (
            (proposed_metrics['boundary_strength'] - baseline_metrics['boundary_strength'])
            / baseline_metrics['boundary_strength'] * 100
        )
    else:
        boundary_improvement = 0.0

    print(f"   • Homogeneity: {homogeneity_improvement:+.1f}% improvement")
    print(f"   • Boundary Strength: {boundary_improvement:+.1f}% improvement")
    print(f"   • Compactness: {proposed_metrics['compactness']:.3f} vs {baseline_metrics['compactness']:.3f}")

    if proposed_metrics['computation_time'] > baseline_metrics['computation_time']:
        time_ratio = proposed_metrics['computation_time'] / baseline_metrics['computation_time']
        print(f"   • Time: {time_ratio:.1f}x slower (but more accurate!)")
    else:
        print(f"   • Time: Faster than baseline!")

    print("\n ADVANTAGES OF PROPOSED METHOD:")
    print("   1. Considers spatial coherence (neighboring pixels)")
    print("   2. Adaptive thresholds based on local texture complexity")
    print("   3. Better region homogeneity")
    print("   4. Stronger boundaries between regions")
    print("   5. More compact, meaningful regions")

    print("\n  TRADE-OFFS:")
    print("   1. Slightly slower than K-Means")
    print("   2. More parameters to tune")
    print("   3. Memory usage for spatial operations")

    print("="*70 + "\n")


def main(input_path, output_dir, k=5, texture_weight=0.5):
    """
    Main function to run complete comparison.
    """
    print("\n" + "="*70)
    print(" COMPREHENSIVE SEGMENTATION COMPARISON")
    print("   Comparing: Adaptive Region Growing vs Baseline Methods")
    print("="*70)

    # Load image
    print(f"\n Loading image: {input_path}")

    if not os.path.exists(input_path):
        print(f" ERROR: Image not found at {input_path}")
        return

    image_bgr = cv2.imread(input_path)

    if image_bgr is None:
        print(f" ERROR: Could not read image")
        return

    print(f"✓ Image loaded: {image_bgr.shape[1]} x {image_bgr.shape[0]} pixels")

    # Run all methods
    results = run_all_methods(image_bgr, k=k, texture_weight=texture_weight)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]

    comparison_path = os.path.join(output_dir, f"{name_without_ext}_full_comparison.png")
    detailed_path = os.path.join(output_dir, f"{name_without_ext}_detailed_comparison.png")

    # Create visualizations
    print("\n Creating visualizations...")
    create_comparison_visualization(image_bgr, results, save_path=comparison_path)
    create_detailed_comparison(image_bgr, results, save_path=detailed_path)

    # Print summary
    print_comparison_summary(results)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n Results saved to: {output_dir}")
    print(f"   • {comparison_path}")
    print(f"   • {detailed_path}")
    print("\n TIP: Use these results in your research report to demonstrate")
    print("   the advantages of your proposed Adaptive Region Growing method!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Adaptive Region Growing with baseline methods"
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
        default="data/output/comparison",
        help="Output directory (default: data/output/comparison)"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of segments (default: 5)"
    )

    parser.add_argument(
        "--texture_weight",
        type=float,
        default=0.5,
        help="Texture weight 0-1 (default: 0.5)"
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output,
        k=args.k,
        texture_weight=args.texture_weight
    )
