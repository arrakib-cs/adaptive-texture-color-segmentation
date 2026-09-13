
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def convert_to_hsv(image_bgr):
   
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return image_hsv


def compute_lbp(image_gray, P=8, R=1):
 
    # The method="uniform" looks for common patterns
    # It makes the LBP more stable and useful
    lbp = local_binary_pattern(image_gray, P, R, method="uniform")
    return lbp


def extract_color_features(image_bgr):
   
    # Convert to HSV
    image_hsv = convert_to_hsv(image_bgr)
    
    # Split into separate channels
    H = image_hsv[:, :, 0]  # Hue channel
    S = image_hsv[:, :, 1]  # Saturation channel
    V = image_hsv[:, :, 2]  # Value channel
    
    # Normalize to [0, 1] range so they work well with LBP later
    # Hue is 0-179, so divide by 179
    # S and V are 0-255, so divide by 255
    H_norm = H.astype(np.float32) / 179.0
    S_norm = S.astype(np.float32) / 255.0
    V_norm = V.astype(np.float32) / 255.0
    
    # Stack them back together into one 3D array
    # Shape will be (height, width, 3)
    hsv_features = np.stack([H_norm, S_norm, V_norm], axis=2)
    
    return hsv_features


def extract_texture_features(image_bgr, P=8, R=1):
  
    # LBP works on grayscale images (black and white)
    # So first, convert the color image to grayscale
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute LBP
    lbp = compute_lbp(image_gray, P, R)
    
    # Normalize to [0, 1] range
    # LBP values can be 0 to 58 (for P=8, method="uniform")
    # We'll just divide by the maximum value in the image
    lbp_max = lbp.max()
    if lbp_max > 0:
        lbp_norm = lbp.astype(np.float32) / lbp_max
    else:
        lbp_norm = lbp.astype(np.float32)
    
    return lbp_norm


def extract_all_features(image_bgr, P=8, R=1):
 
    hsv_features = extract_color_features(image_bgr)
    lbp_features = extract_texture_features(image_bgr, P, R)
    
    return hsv_features, lbp_features