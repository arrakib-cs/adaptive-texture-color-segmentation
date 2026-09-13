
import os
import argparse
import cv2

# Import our custom functions from other files
from src.features import extract_all_features
from src.fusion import build_feature_matrix, normalize_features, adjust_weights
from src.segment_kmeans import segment_pipeline
from src.visualize import (
    visualize_segmentation_result,
    save_result,
    visualize_features,
    create_overlay,
    show_images
)


def main(input_path, output_dir, k=4, texture_weight=0.5, show_features=False):
    
    
    print("\n" + "="*60)
    print(" ADAPTIVE IMAGE SEGMENTATION")
    print("   Texture-Color Fusion Method")
    print("="*60)
    
    # ==========================================
    # STEP 1: Load the image
    # ==========================================
    print(f"\n Loading image from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f" ERROR: Image not found at {input_path}")
        print("   Please check the path and try again!")
        return
    
    # Read image with OpenCV (loads as BGR)
    image_bgr = cv2.imread(input_path)
    
    if image_bgr is None:
        print(f" ERROR: Could not read image at {input_path}")
        print("   Make sure it's a valid image file (.jpg, .png, etc.)")
        return
    
    height, width = image_bgr.shape[:2]
    print(f" Image loaded successfully!")
    print(f"   Size: {width} x {height} pixels")
    
    # ==========================================
    # STEP 2: Extract Features
    # ==========================================
    print(f"\n Extracting features...")
    print(f"   - Color features: HSV (Hue, Saturation, Value)")
    print(f"   - Texture features: LBP (Local Binary Pattern)")
    
    hsv_features, lbp_features = extract_all_features(image_bgr)
    
    print(f"Features extracted!")
    print(f"   HSV shape: {hsv_features.shape}")
    print(f"   LBP shape: {lbp_features.shape}")
    
    # Optionally visualize the features
    if show_features:
        print("\n  Showing extracted features...")
        visualize_features(lbp_features, hsv_features)
    
    # ==========================================
    # STEP 3: Fuse Features
    # ==========================================
    print(f"\n Fusing features...")
    print(f"   Texture importance: {texture_weight:.2f}")
    print(f"   Color importance: {1-texture_weight:.2f}")
    
    # Adjust weights based on texture_weight parameter
    weights = adjust_weights(texture_importance=texture_weight)
    print(f"   Weights: {weights}")
    
    # Build combined feature matrix
    feature_matrix, original_shape = build_feature_matrix(
        lbp_features, 
        hsv_features, 
        weights=weights
    )
    
    print(f"Feature matrix created!")
    print(f"   Shape: {feature_matrix.shape}")
    print(f"   (Each of {feature_matrix.shape[0]} pixels has {feature_matrix.shape[1]} features)")
    
    # Normalize features for better clustering
    feature_matrix = normalize_features(feature_matrix)
    print(f" Features normalized to [0, 1] range")
    
    # ==========================================
    # STEP 4: Segment the Image
    # ==========================================
    print(f"\n Segmenting image into K={k} regions...")
    print(f"   Using KMeans clustering algorithm...")
    
    label_image, colored_seg = segment_pipeline(
        feature_matrix,
        original_shape,
        k=k
    )
    
    print(f"Segmentation complete!")
    print(f"   Created {k} distinct regions")
    
    # ==========================================
    # STEP 5: Create Overlay (boundaries)
    # ==========================================
    print(f"\n Creating boundary overlay...")
    overlay = create_overlay(image_bgr, label_image, alpha=0.3)
    print(f" Overlay created!")
    
    # ==========================================
    # STEP 6: Save Results
    # ==========================================
    print(f"\n Saving results to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    
    seg_path = os.path.join(output_dir, f"{name_without_ext}_segmented_k{k}.png")
    overlay_path = os.path.join(output_dir, f"{name_without_ext}_overlay_k{k}.png")
    comparison_path = os.path.join(output_dir, f"{name_without_ext}_comparison_k{k}.png")
    
    # Save segmentation result
    save_result(colored_seg, seg_path)
    
    # Save overlay
    save_result(overlay, overlay_path)
    
    # ==========================================
    # STEP 7: Visualize Results
    # ==========================================
    print(f"\n Displaying results...")
    
    # Show comparison
    visualize_segmentation_result(
        image_bgr, 
        colored_seg, 
        k, 
        save_path=comparison_path
    )
    
    # Show overlay
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    show_images(
        [original_rgb, overlay],
        ["Original Image", "Segmentation Boundaries"],
        figsize=(12, 5)
    )
    
    # ==========================================
    # STEP 8: Summary
    # ==========================================
    print("\n" + "="*60)
    print("SEGMENTATION COMPLETE!")
    print("="*60)
    print(f"Results saved to:")
    print(f"   • {seg_path}")
    print(f"   • {overlay_path}")
    print(f"   • {comparison_path}")
    print("\n Tip: Try changing K or texture_weight to see different results!")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    This part runs when you execute the script from command line.
    
    It handles command-line arguments (the parameters you pass when running).
    """
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Adaptive Image Segmentation using Texture-Color Fusion"
    )
    
    # Define arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image (e.g., data/input/sample.jpg)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/output",
        help="Output directory for results (default: data/output)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of segments/clusters (default: 4)"
    )
    
    parser.add_argument(
        "--texture_weight",
        type=float,
        default=0.5,
        help="Weight for texture (0.0-1.0, default: 0.5). Higher = more texture importance"
    )
    
    parser.add_argument(
        "--show_features",
        action="store_true",
        help="Show extracted features (LBP, H, S, V) before segmentation"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with parsed arguments
    main(
        input_path=args.input,
        output_dir=args.output,
        k=args.k,
        texture_weight=args.texture_weight,
        show_features=args.show_features
    )