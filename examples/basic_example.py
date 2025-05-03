#!/usr/bin/env python3
"""
Basic example of using the Ishihara K-means package

This script demonstrates how to use the Ishihara K-means package
to extract numbers from Ishihara color blindness test images.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kmeans import KMeansClustering
from src.image_processing import preprocess_image, segment_image, extract_number
from src.visualization import plot_results, create_final_visualization

def process_image(image_path, color_space='lab', channel=1, k=2, invert=False):
    """
    Process an Ishihara test image to extract the hidden number
    
    Args:
        image_path (str): Path to the Ishihara test image
        color_space (str): Color space to use ('rgb', 'hsv', 'lab', 'ycrcb')
        channel (int or None): Specific channel to use (None means use all channels)
        k (int): Number of clusters
        invert (bool): Whether to invert the extracted number mask
        
    Returns:
        tuple: (original_image, segmented_image, number_mask)
    """
    print(f"Processing image: {image_path}")
    print(f"Parameters: {color_space.upper()} color space" + 
          (f", channel {channel}" if channel is not None else "") + 
          f", k={k}" + (", inverted" if invert else ""))
    
    # Preprocess image
    image_rgb, features, _ = preprocess_image(image_path, color_space, channel)
    
    # Apply K-means clustering
    print("Fitting K-means clustering...")
    kmeans = KMeansClustering(k=k, max_iterations=100)
    kmeans.fit(features)
    
    # Segment image
    print("Segmenting image...")
    labels, segmented_image = segment_image(kmeans, features, image_rgb.shape, channel)
    
    # Extract number
    print("Extracting number...")
    number_mask = extract_number(labels, k, invert)
    
    return image_rgb, segmented_image, number_mask

def main():
    """
    Main function to demonstrate the Ishihara K-means package
    """
    # Set path to your Ishihara test image
    # You should replace this with the path to your own image
    image_path = "../data/ishihara_74.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path variable to point to a valid image.")
        sys.exit(1)
    
    # Process the image
    image_rgb, segmented_image, number_mask = process_image(
        image_path, 
        color_space='lab',  # Lab color space works well for most Ishihara tests
        channel=1,          # a channel (red-green component)
        k=2,                # 2 clusters is usually sufficient
        invert=False        # Usually not needed for Lab a-channel
    )
    
    # Plot results
    title = "LAB color space, a channel (red-green), k=2"
    plot_results(image_rgb, segmented_image, number_mask, title)
    
    # Save final visualization
    output_path = "ishihara_result.png"
    create_final_visualization(image_rgb, number_mask, output_path)
    
    print(f"Result saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
