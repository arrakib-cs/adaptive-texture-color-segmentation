

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_images(images, titles, figsize=(15, 5)):
    """
    Display multiple images side by side.
    
    Input:
        images: List of images to show
        titles: List of titles for each image
        figsize: Size of the figure (width, height) in inches
    """
    num_images = len(images)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # If only one image, axes is not a list, so make it one
    if num_images == 1:
        axes = [axes]
    
    # Display each image
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.show()


def save_result(image, output_path):
    """
    Save an image to a file.
    
    Input:
        image: The image to save (numpy array)
        output_path: Where to save it (e.g., "data/output/result.png")
    """
    # OpenCV expects BGR for saving, but our colored segmentation is RGB
    # So we need to check and convert if necessary
    
    # If it's a colored segmentation (3 channels), convert RGB to BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image
    
    cv2.imwrite(output_path, image_to_save)
    print(f" Saved result to: {output_path}")


def visualize_segmentation_result(original_bgr, colored_seg, k, save_path=None):
    """
    Create a nice visualization comparing original image and segmentation.
    
    Input:
        original_bgr: Original image in BGR format
        colored_seg: Colored segmentation in RGB format
        k: Number of segments (for title)
        save_path: If provided, save the figure to this path
    """
    # Convert original from BGR to RGB for display
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation result
    axes[1].imshow(colored_seg)
    axes[1].set_title(f"Segmentation (K={k} regions)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved visualization to: {save_path}")
    
    plt.show()


def create_overlay(original_bgr, label_image, alpha=0.5):
    """
    Create an overlay where segmentation boundaries are drawn on original image.
    
    This creates a nice effect where you can see both the original image
    and the segmentation at the same time!
    
    Input:
        original_bgr: Original image in BGR
        label_image: 2D array of cluster labels
        alpha: Transparency (0=invisible, 1=fully visible)
    
    Output:
        overlay: RGB image with boundaries
    """
    # Convert original to RGB
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Find boundaries between different segments
    # We'll use a simple method: check if neighboring pixels have different labels
    boundaries = np.zeros(label_image.shape, dtype=np.uint8)
    
    # Check horizontal boundaries
    boundaries[:, :-1] |= (label_image[:, :-1] != label_image[:, 1:])
    # Check vertical boundaries
    boundaries[:-1, :] |= (label_image[:-1, :] != label_image[1:, :])
    
    # Create overlay
    overlay = original_rgb.copy()
    
    # Draw boundaries in red
    overlay[boundaries > 0] = [255, 0, 0]  # Red color
    
    # Blend with original
    result = cv2.addWeighted(original_rgb, 1-alpha, overlay, alpha, 0)
    
    return result


def visualize_features(lbp_features, hsv_features):
    """
    Visualize the extracted features (LBP and HSV channels).
    
    This helps us understand what the algorithm "sees"!
    
    Input:
        lbp_features: LBP texture features
        hsv_features: HSV color features (H, S, V channels)
    """
    # Extract individual channels
    H = hsv_features[:, :, 0]
    S = hsv_features[:, :, 1]
    V = hsv_features[:, :, 2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # LBP texture
    im0 = axes[0, 0].imshow(lbp_features, cmap='gray')
    axes[0, 0].set_title("LBP Texture", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Hue
    im1 = axes[0, 1].imshow(H, cmap='hsv')
    axes[0, 1].set_title("Hue (Color Type)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Saturation
    im2 = axes[1, 0].imshow(S, cmap='gray')
    axes[1, 0].set_title("Saturation (Color Strength)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Value
    im3 = axes[1, 1].imshow(V, cmap='gray')
    axes[1, 1].set_title("Value (Brightness)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.show()