import numpy as np
import cv2

def preprocess_image(image_path, color_space='rgb', channel=None):
    """
    Preprocess image for K-means clustering with different color spaces
    
    Args:
        image_path (str): Path to the input image
        color_space (str): Color space to use ('rgb', 'hsv', 'lab', 'ycrcb')
        channel (int or None): Specific channel to use for clustering (None means use all)
        
    Returns:
        tuple: (image_rgb, features, original) - processed image for display, 
               flattened features for clustering, and original image
    """
    # Read image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Convert BGR to RGB (for display purposes)
    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Convert to the specified color space
    if color_space.lower() == 'rgb':
        image = image_rgb
    elif color_space.lower() == 'hsv':
        image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    elif color_space.lower() == 'lab':
        image = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    elif color_space.lower() == 'ycrcb':
        image = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    # Extract specific channel if requested
    if channel is not None:
        if channel < 0 or channel >= image.shape[2]:
            raise ValueError(f"Invalid channel {channel} for color space {color_space}")
        
        # Extract the specified channel and reshape for clustering
        height, width = image.shape[:2]
        features = image[:, :, channel].reshape(height * width, 1)
    else:
        # Use all channels
        height, width, channels = image.shape
        features = image.reshape(height * width, channels)
    
    return image_rgb, features, original

def segment_image(kmeans, features, image_shape, channel=None):
    """
    Segment image based on K-means clustering
    
    Args:
        kmeans: Fitted K-means model
        features (numpy.ndarray): Flattened features
        image_shape (tuple): Shape of the original image
        channel (int or None): Specific channel used for clustering
        
    Returns:
        tuple: (labels, segmented_image) - cluster labels and segmented image
    """
    # Predict clusters
    labels = kmeans.predict(features)
    
    # Reshape labels to original image shape (2D if we used a specific channel, 3D otherwise)
    if channel is not None:
        labels = labels.reshape(image_shape[0], image_shape[1])
        
        # Create a grayscale segmented image
        # Sort clusters by size (number of pixels)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Map clusters to grayscale values (largest cluster = black, smallest = white)
        cluster_map = {}
        n_clusters = len(sorted_clusters)
        for i, (cluster, _) in enumerate(sorted_clusters):
            # Map to a grayscale value (0 to 255)
            gray_value = int(255 * (1 - i / (n_clusters - 1))) if n_clusters > 1 else 0
            cluster_map[cluster] = gray_value
        
        # Create segmented image
        segmented_image = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        for cluster, gray_value in cluster_map.items():
            segmented_image[labels == cluster] = gray_value
    else:
        # Handle full color segmentation
        labels = labels.reshape(image_shape[0], image_shape[1])
        
        # Create segmented image by assigning each pixel to its cluster centroid
        segmented_image = np.zeros((image_shape[0], image_shape[1], image_shape[2]))
        
        for i in range(kmeans.k):
            segmented_image[labels == i] = kmeans.centroids[i]
        
        # Convert to uint8
        segmented_image = segmented_image.astype(np.uint8)
    
    return labels, segmented_image

def extract_number(labels, k, invert=False):
    """
    Extract the number from segmented Ishihara image
    
    Args:
        labels (numpy.ndarray): Cluster labels
        k (int): Number of clusters
        invert (bool): Whether to invert the result (for certain color spaces)
        
    Returns:
        numpy.ndarray: Binary mask of the extracted number
    """
    # For Ishihara images, one of the clusters should represent the number
    # We need to find which cluster represents the number (usually distinct from background)
    
    # Convert to binary masks for each cluster
    masks = []
    for i in range(k):
        mask = (labels == i).astype(np.uint8)
        masks.append(mask)
    
    # Find cluster sizes
    cluster_sizes = [np.sum(mask) for mask in masks]
    
    # Sort clusters by size (largest first)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    
    # For Ishihara tests, we typically have:
    # 1. The largest cluster is usually the background
    # 2. The second largest is often the circle of dots
    # 3. The third largest (or smaller) is often the number
    
    # Select the appropriate cluster based on k
    if k == 2:
        number_cluster = sorted_indices[-1]  # Smallest cluster
    else:
        # For k>2, use the second smallest cluster (often the number)
        number_cluster = sorted_indices[-2]
    
    # Get the binary mask for the number cluster
    number_mask = masks[number_cluster].copy()
    
    # Invert if requested (depending on the color space and how clusters are assigned)
    if invert:
        number_mask = 1 - number_mask
    
    # Post-process the mask to clean noise
    kernel = np.ones((5, 5), np.uint8)
    number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_OPEN, kernel)
    number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_CLOSE, kernel)
    
    return number_mask * 255  # Scale to 0-255 for visualization

def analyze_with_multiple_params(image_path):
    """
    Analyze an image with multiple parameter combinations
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        list: List of tuples (parameters, segmented_image, number_mask)
    """
    from src.kmeans import KMeansClustering
    
    # Best parameters for Ishihara tests based on research
    parameter_sets = [
        # color_space, channel, k, invert, title
        ('lab', 1, 2, False, 'Lab (a channel)'),
        ('ycrcb', 1, 2, False, 'YCrCb (Cr channel)'),
        ('hsv', 0, 3, False, 'HSV (Hue channel)'),
        ('rgb', 0, 2, False, 'RGB (Red channel)'),
        ('rgb', 1, 2, True, 'RGB (Green channel)'),
    ]
    
    results = []
    
    # Try different parameter sets
    for color_space, channel, k, invert, title in parameter_sets:
        # Preprocess image
        image_rgb, features, _ = preprocess_image(image_path, color_space, channel)
        
        # Initialize and fit K-means
        kmeans = KMeansClustering(k=k, max_iterations=100)
        kmeans.fit(features)
        
        # Segment image
        labels, segmented_image = segment_image(kmeans, features, image_rgb.shape, channel)
        
        # Extract number
        number_mask = extract_number(labels, k, invert)
        
        # Store results
        params = {
            'color_space': color_space,
            'channel': channel,
            'k': k,
            'invert': invert,
            'title': title
        }
        results.append((params, segmented_image, number_mask))
    
    return image_rgb, results
