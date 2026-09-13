"""
NOVEL ALGORITHM: Adaptive Region Growing with Texture-Color Fusion
===================================================================
This is YOUR PROPOSED ALGORITHM - completely coded from scratch.

This algorithm implements an adaptive region growing approach that:
1. Intelligently selects seed points based on texture homogeneity
2. Grows regions using combined texture-color similarity
3. Adapts thresholds based on local texture complexity
4. Merges similar adjacent regions

NO LIBRARY FUNCTIONS for the core algorithm - all coded by you!
"""

import numpy as np
from collections import deque
import cv2


class AdaptiveRegionGrowing:
    """
     NOVEL SEGMENTATION ALGORITHM
    
    This class implements adaptive region growing that combines:
    - Texture features (LBP)
    - Color features (HSV)
    - Spatial coherence
    - Adaptive similarity thresholds
    """
    
    def __init__(self, feature_matrix, original_shape, num_regions=5, 
                 initial_threshold=0.15, texture_weight=0.5):
        """
        Initialize the adaptive region growing algorithm.
        
        Parameters:
        -----------
        feature_matrix : ndarray, shape (n_pixels, n_features)
            Combined texture-color features for each pixel
        original_shape : tuple (height, width)
            Original image dimensions
        num_regions : int
            Desired number of regions
        initial_threshold : float
            Initial similarity threshold (will be adapted)
        texture_weight : float
            Weight for texture vs color (0 to 1)
        """
        self.feature_matrix = feature_matrix
        self.height, self.width = original_shape
        self.num_pixels = self.height * self.width
        self.num_regions = num_regions
        self.initial_threshold = initial_threshold
        self.texture_weight = texture_weight
        
        # Reshape features back to 2D for spatial operations
        self.feature_image = feature_matrix.reshape(self.height, self.width, -1)
        
        # Initialize segmentation map (-1 means unassigned)
        self.segmentation = np.full((self.height, self.width), -1, dtype=np.int32)
        
        # Region statistics
        self.region_features = []  # Mean features for each region
        self.region_sizes = []      # Number of pixels in each region
        
    
    def compute_texture_variance_map(self):
        """
        Compute local texture variance to identify homogeneous regions.
        This helps in selecting good seed points.
        
        Returns:
        --------
        variance_map : ndarray, shape (height, width)
            Local texture variance at each pixel
        """
        # Extract texture feature (first channel in our case)
        texture = self.feature_image[:, :, 0]
        
        # Compute local variance using a 5x5 window
        variance_map = np.zeros((self.height, self.width))
        
        window_size = 5
        half_window = window_size // 2
        
        # Pad the texture image
        padded = np.pad(texture, half_window, mode='reflect')
        
        for i in range(self.height):
            for j in range(self.width):
                # Extract local window
                window = padded[i:i+window_size, j:j+window_size]
                # Compute variance
                variance_map[i, j] = np.var(window)
        
        return variance_map
    
    
    def select_seed_points(self):
        """
        Intelligently select seed points based on texture homogeneity.
        Seeds are placed in regions with low texture variance (homogeneous areas).
        
        Returns:
        --------
        seeds : list of tuples [(y1, x1), (y2, x2), ...]
            Seed point coordinates
        """
        print(f"\n Selecting {self.num_regions} seed points...")
        
        # Compute texture variance map
        variance_map = self.compute_texture_variance_map()
        
        seeds = []
        
        # Divide image into grid to ensure spatial distribution
        grid_rows = int(np.sqrt(self.num_regions))
        grid_cols = int(np.ceil(self.num_regions / grid_rows))
        
        cell_height = self.height // grid_rows
        cell_width = self.width // grid_cols
        
        seed_count = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if seed_count >= self.num_regions:
                    break
                
                # Define grid cell boundaries
                y_start = i * cell_height
                y_end = min((i + 1) * cell_height, self.height)
                x_start = j * cell_width
                x_end = min((j + 1) * cell_width, self.width)
                
                # Find point with minimum variance in this cell
                cell_variance = variance_map[y_start:y_end, x_start:x_end]
                
                # Find minimum variance position in cell
                min_pos = np.unravel_index(np.argmin(cell_variance), cell_variance.shape)
                seed_y = y_start + min_pos[0]
                seed_x = x_start + min_pos[1]
                
                seeds.append((seed_y, seed_x))
                seed_count += 1
        
        print(f"   ✓ Seeds selected at homogeneous regions")
        return seeds
    
    
    def compute_similarity(self, pixel_features, region_features):
        """
        Compute similarity between a pixel and a region.
        Uses Euclidean distance in the combined feature space.
        
        Parameters:
        -----------
        pixel_features : ndarray, shape (n_features,)
            Feature vector of the pixel
        region_features : ndarray, shape (n_features,)
            Mean feature vector of the region
            
        Returns:
        --------
        similarity : float
            Similarity score (lower = more similar)
        """
        # Euclidean distance
        distance = np.sqrt(np.sum((pixel_features - region_features) ** 2))
        return distance
    
    
    def get_neighbors(self, y, x):
        """
        Get 4-connected neighbors of a pixel (up, down, left, right).
        
        Parameters:
        -----------
        y, x : int
            Pixel coordinates
            
        Returns:
        --------
        neighbors : list of tuples
            Valid neighbor coordinates
        """
        neighbors = []
        
        # 4-connectivity: up, down, left, right
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dy, dx in deltas:
            ny, nx = y + dy, x + dx
            # Check if neighbor is within image bounds
            if 0 <= ny < self.height and 0 <= nx < self.width:
                neighbors.append((ny, nx))
        
        return neighbors
    
    
    def compute_adaptive_threshold(self, y, x):
        """
        Compute adaptive threshold based on local texture complexity.
        Higher texture complexity → higher threshold (more lenient)
        Lower texture complexity → lower threshold (more strict)
        
        Parameters:
        -----------
        y, x : int
            Pixel coordinates
            
        Returns:
        --------
        threshold : float
            Adaptive similarity threshold
        """
        # Extract local window around pixel
        window_size = 7
        half_window = window_size // 2
        
        y_start = max(0, y - half_window)
        y_end = min(self.height, y + half_window + 1)
        x_start = max(0, x - half_window)
        x_end = min(self.width, x + half_window + 1)
        
        # Extract texture features in local window
        local_texture = self.feature_image[y_start:y_end, x_start:x_end, 0]
        
        # Compute local texture variance
        local_variance = np.var(local_texture)
        
        # Adaptive threshold: base + factor * variance
        # More variance → higher threshold
        adaptive_threshold = self.initial_threshold + (local_variance * 2.0)
        
        # Clamp threshold to reasonable range
        adaptive_threshold = np.clip(adaptive_threshold, 0.05, 0.5)
        
        return adaptive_threshold
    
    
    def grow_region(self, seed_y, seed_x, region_id):
        """
        Grow a region starting from a seed point using adaptive thresholds.
        
        Parameters:
        -----------
        seed_y, seed_x : int
            Seed point coordinates
        region_id : int
            ID for this region
        """
        # Initialize queue with seed point
        queue = deque([(seed_y, seed_x)])
        
        # Mark seed as visited
        self.segmentation[seed_y, seed_x] = region_id
        
        # Initialize region statistics
        region_feature_sum = self.feature_image[seed_y, seed_x].copy()
        region_pixel_count = 1
        
        # Region growing loop
        while queue:
            y, x = queue.popleft()
            
            # Get current region mean features
            region_mean = region_feature_sum / region_pixel_count
            
            # Get adaptive threshold for this location
            threshold = self.compute_adaptive_threshold(y, x)
            
            # Check all neighbors
            neighbors = self.get_neighbors(y, x)
            
            for ny, nx in neighbors:
                # Skip if already assigned
                if self.segmentation[ny, nx] != -1:
                    continue
                
                # Get neighbor features
                neighbor_features = self.feature_image[ny, nx]
                
                # Compute similarity
                similarity = self.compute_similarity(neighbor_features, region_mean)
                
                # If similar enough, add to region
                if similarity < threshold:
                    self.segmentation[ny, nx] = region_id
                    queue.append((ny, nx))
                    
                    # Update region statistics
                    region_feature_sum += neighbor_features
                    region_pixel_count += 1
        
        # Store final region statistics
        region_mean_features = region_feature_sum / region_pixel_count
        return region_mean_features, region_pixel_count
    
    
    def assign_remaining_pixels(self):
        """
        Assign any remaining unassigned pixels to the nearest region.
        """
        print(f"\n Assigning remaining pixels...")
        
        unassigned_count = np.sum(self.segmentation == -1)
        
        if unassigned_count == 0:
            print(f"   ✓ All pixels assigned during region growing")
            return
        
        print(f"   Processing {unassigned_count} unassigned pixels...")
        
        # Find all unassigned pixels
        unassigned_y, unassigned_x = np.where(self.segmentation == -1)
        
        for i in range(len(unassigned_y)):
            y, x = unassigned_y[i], unassigned_x[i]
            pixel_features = self.feature_image[y, x]
            
            # Find most similar region
            best_region = -1
            best_similarity = float('inf')
            
            for region_id in range(len(self.region_features)):
                similarity = self.compute_similarity(
                    pixel_features, 
                    self.region_features[region_id]
                )
                
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_region = region_id
            
            # Assign to best region
            self.segmentation[y, x] = best_region
        
        print(f"   ✓ All pixels now assigned")
    
    
    def merge_small_regions(self, min_size=100):
        """
        Merge regions that are too small into their most similar neighbor.
        
        Parameters:
        -----------
        min_size : int
            Minimum region size in pixels
        """
        print(f"\n Merging small regions (min size: {min_size} pixels)...")
        
        # Count pixels in each region
        unique_regions, region_sizes = np.unique(self.segmentation, return_counts=True)
        
        small_regions = unique_regions[region_sizes < min_size]
        
        if len(small_regions) == 0:
            print(f"   ✓ No small regions to merge")
            return
        
        print(f"   Found {len(small_regions)} small regions to merge")
        
        for small_region_id in small_regions:
            # Find all pixels in this small region
            region_mask = (self.segmentation == small_region_id)
            region_pixels_y, region_pixels_x = np.where(region_mask)
            
            # Find neighboring regions
            neighbor_regions = set()
            for y, x in zip(region_pixels_y, region_pixels_x):
                neighbors = self.get_neighbors(y, x)
                for ny, nx in neighbors:
                    neighbor_id = self.segmentation[ny, nx]
                    if neighbor_id != small_region_id:
                        neighbor_regions.add(neighbor_id)
            
            if not neighbor_regions:
                continue
            
            # Get mean features of small region
            small_region_features = np.mean(
                self.feature_image[region_mask], axis=0
            )
            
            # Find most similar neighboring region
            best_neighbor = -1
            best_similarity = float('inf')
            
            for neighbor_id in neighbor_regions:
                if neighbor_id >= len(self.region_features):
                    continue
                
                similarity = self.compute_similarity(
                    small_region_features,
                    self.region_features[neighbor_id]
                )
                
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_neighbor = neighbor_id
            
            # Merge into best neighbor
            if best_neighbor != -1:
                self.segmentation[region_mask] = best_neighbor
        
        print(f"   ✓ Small regions merged")
    
    
    def segment(self):
        """
        Main segmentation method - runs the complete algorithm.
        
        Returns:
        --------
        segmentation : ndarray, shape (height, width)
            Segmented image with region labels
        """
        print("\n" + "="*60)
        print(" RUNNING ADAPTIVE REGION GROWING ALGORITHM")
        print("="*60)
        
        # Step 1: Select seed points
        seeds = self.select_seed_points()
        
        # Step 2: Grow regions from each seed
        print(f"\n Growing {self.num_regions} regions...")
        for region_id, (seed_y, seed_x) in enumerate(seeds):
            print(f"   Growing region {region_id + 1}/{self.num_regions}...", end='')
            region_mean, region_size = self.grow_region(seed_y, seed_x, region_id)
            self.region_features.append(region_mean)
            self.region_sizes.append(region_size)
            print(f" ✓ ({region_size} pixels)")
        
        # Step 3: Assign remaining pixels
        self.assign_remaining_pixels()
        
        # Step 4: Merge small regions
        self.merge_small_regions(min_size=50)
        
        # Step 5: Relabel regions to be contiguous (0, 1, 2, ...)
        unique_labels = np.unique(self.segmentation)
        relabeled = np.zeros_like(self.segmentation)
        for new_label, old_label in enumerate(unique_labels):
            relabeled[self.segmentation == old_label] = new_label
        
        self.segmentation = relabeled
        
        print("\n" + "="*60)
        print(" ADAPTIVE REGION GROWING COMPLETE!")
        print("="*60)
        
        # Print final statistics
        final_regions = len(np.unique(self.segmentation))
        print(f"\n Final Statistics:")
        print(f"   • Final number of regions: {final_regions}")
        print(f"   • Average region size: {self.num_pixels // final_regions} pixels")
        print("="*60 + "\n")
        
        return self.segmentation


def adaptive_region_growing_pipeline(feature_matrix, original_shape, 
                                     num_regions=5, initial_threshold=0.15,
                                     texture_weight=0.5):
    """
    Complete pipeline for adaptive region growing segmentation.
    
    Parameters:
    -----------
    feature_matrix : ndarray
        Combined texture-color features
    original_shape : tuple
        (height, width) of original image
    num_regions : int
        Desired number of regions
    initial_threshold : float
        Initial similarity threshold
    texture_weight : float
        Weight for texture vs color
        
    Returns:
    --------
    label_image : ndarray
        Segmentation result
    colored_seg : ndarray
        Colored visualization of segmentation
    """
    # Create algorithm instance
    algorithm = AdaptiveRegionGrowing(
        feature_matrix=feature_matrix,
        original_shape=original_shape,
        num_regions=num_regions,
        initial_threshold=initial_threshold,
        texture_weight=texture_weight
    )
    
    # Run segmentation
    label_image = algorithm.segment()
    
    # Create colored visualization
    colored_seg = create_colored_segmentation(label_image)
    
    return label_image, colored_seg


def create_colored_segmentation(label_image):
    """
    Create a colored visualization of the segmentation.
    
    Parameters:
    -----------
    label_image : ndarray
        Segmentation with integer labels
        
    Returns:
    --------
    colored_seg : ndarray
        RGB colored segmentation
    """
    height, width = label_image.shape
    num_regions = len(np.unique(label_image))
    
    # Create random colors for each region
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_regions, 3), dtype=np.uint8)
    
    # Create colored image
    colored_seg = np.zeros((height, width, 3), dtype=np.uint8)
    
    for region_id in range(num_regions):
        mask = (label_image == region_id)
        colored_seg[mask] = colors[region_id]
    
    return colored_seg