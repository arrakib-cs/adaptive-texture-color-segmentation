

import numpy as np


def build_feature_matrix(lbp_features, hsv_features, weights=None):
   
    
    # Get image dimensions
    height, width = lbp_features.shape
    num_pixels = height * width
    
    # Set default weights if none provided
    if weights is None:
        weights = {
            'texture': 0.5,   # 50% importance to texture
            'hue': 0.2,       # 20% importance to hue (color type)
            'sat': 0.15,      # 15% importance to saturation (color strength)
            'val': 0.15       # 15% importance to value (brightness)
        }
    
    # Extract individual color channels
    H = hsv_features[:, :, 0]  # Hue
    S = hsv_features[:, :, 1]  # Saturation
    V = hsv_features[:, :, 2]  # Value
    
    # Flatten everything into 1D arrays (like unrolling a carpet into a line)
    # This converts from 2D (image) to 1D (list of pixels)
    lbp_flat = lbp_features.flatten()
    H_flat = H.flatten()
    S_flat = S.flatten()
    V_flat = V.flatten()
    
    # Apply weights
    lbp_weighted = lbp_flat * weights['texture']
    H_weighted = H_flat * weights['hue']
    S_weighted = S_flat * weights['sat']
    V_weighted = V_flat * weights['val']
    
    # Stack them together into a matrix
    # Each row = one pixel's complete feature vector
    feature_matrix = np.column_stack([
        lbp_weighted,
        H_weighted,
        S_weighted,
        V_weighted
    ])
    
    # feature_matrix shape: (num_pixels, 4)
    # Row 0 = pixel 0's features: [LBP, H, S, V]
    # Row 1 = pixel 1's features: [LBP, H, S, V]
    # ...
    
    return feature_matrix, (height, width)


def compute_pixel_similarity(feature1, feature2):
    
    # Euclidean distance (like measuring with a ruler in 4D space)
    distance = np.linalg.norm(feature1 - feature2)
    return distance


def normalize_features(feature_matrix):
   
    # For each feature column, find min and max
    min_vals = feature_matrix.min(axis=0)
    max_vals = feature_matrix.max(axis=0)
    
    # Avoid division by zero
    # If all values in a column are the same, max_vals = min_vals
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # Prevent division by zero
    
    # Scale: (value - min) / (max - min)
    # This transforms each column to [0, 1] range
    normalized_matrix = (feature_matrix - min_vals) / range_vals
    
    return normalized_matrix


def adjust_weights(texture_importance=0.5):
   
    # Make sure texture_importance is between 0 and 1
    texture_importance = max(0.0, min(1.0, texture_importance))
    
    # Remaining importance goes to color (split equally among H, S, V)
    color_importance = 1.0 - texture_importance
    individual_color_weight = color_importance / 3.0  # Divide by 3 for H, S, V
    
    weights = {
        'texture': texture_importance,
        'hue': individual_color_weight,
        'sat': individual_color_weight,
        'val': individual_color_weight
    }
    
    return weights