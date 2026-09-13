

import numpy as np
from sklearn.cluster import KMeans


def segment_image_kmeans(feature_matrix, k=4, random_state=42):
   
    # Create KMeans model
    # n_clusters = how many groups we want
    # random_state = makes it reproducible
    # n_init = number of times to run with different starting points
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )
    
    # Fit the model and predict cluster labels
    # This is where the magic happens!
    # KMeans looks at all pixels and groups them
    labels = kmeans.fit_predict(feature_matrix)
    
    return labels


def reshape_labels_to_image(labels, original_shape):
   
    height, width = original_shape
    label_image = labels.reshape(height, width)
    return label_image


def create_colored_segmentation(label_image, k):
    
    height, width = label_image.shape
    
    # Create an empty RGB image
    colored_seg = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create random colors for each cluster
    # We use a fixed random seed so colors are consistent
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    
    # Assign colors to each pixel based on its cluster
    for cluster_id in range(k):
        # Find all pixels in this cluster
        mask = (label_image == cluster_id)
        # Color them with the cluster's color
        colored_seg[mask] = colors[cluster_id]
    
    return colored_seg


def segment_pipeline(feature_matrix, original_shape, k=4):
    
    # Step 1: Cluster the pixels
    labels = segment_image_kmeans(feature_matrix, k=k)
    
    # Step 2: Reshape to image
    label_image = reshape_labels_to_image(labels, original_shape)
    
    # Step 3: Create colored visualization
    colored_seg = create_colored_segmentation(label_image, k)
    
    return label_image, colored_seg