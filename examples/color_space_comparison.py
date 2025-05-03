#!/usr/bin/env python3
"""
Color space comparison for Ishihara test images

This script demonstrates how different color spaces affect
the extraction of numbers from Ishihara color blindness test images.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kmeans import KMeansClustering
from src.image_processing import preprocess_image, segment_image, extract_number
from src.color_spaces import visualize_color_channels, recommend_color_space
from src.visualization import plot_color_space_comparison, save_comparison_figure

def compare_color_spaces(image_path):
    """
    Compare different color spaces for extracting numbers from an Ishihara test image
    
    Args:
        image_path (str): Path to the Ishihara test image
        
    Returns:
        tuple: (original_image, results)
    """
    # Parameter sets to try (color_space, channel, k, invert, title)
    parameter_sets = [
        ('lab', 1, 2, False, 'Lab (a channel)'),
        ('ycrcb', 1, 2, False, 'YCrCb (Cr channel)'),
        ('hsv', 0, 3, False, 'HSV (Hue channel)'),
        ('rgb', 0, 2, False, 'RGB (Red channel)'),
        ('rgb', 1, 2, True, 'RGB (Green channel)'),
    ]
    
    # Initialize results list
    results = []
    original_image = None
    
    # Process with each parameter set
    for color_space, channel, k, invert, title in parameter_sets:
        print(f"Processing with {title}...")
        
        # Preprocess image
        image_rgb, features, _ = preprocess_image(image_path, color_space, channel)
        
        # Store original image
        if original_image is None:
            original_image = image_rgb
        
        # Apply K-means clustering
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
    
    return original_image, results

def analyze_color_channels(image_path):
    """
    Visualize the individual channels of an image in different color spaces
    
    Args:
        image_path (str): Path to the Ishihara test image
    """
    import cv2
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Visualize channels in each color space
    color_spaces = ['rgb', 'hsv', 'lab', 'ycrcb']
    
    for cs in color_spaces:
        print(f"Visualizing {cs.upper()} color space channels...")
        fig, _, _, _ = visualize_color_channels(image, cs)
        fig.savefig(f"{os.path.splitext(image_path)[0]}_{cs}_channels.png", dpi=300)
        plt.close(fig)

def main():
    """
    Main function to demonstrate color space comparison
    """
    # Set path to your Ishihara test image
    # You should replace this with the path to your own image
    image_path = "../data/ishihara_74.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path variable to point to a valid image.")
        sys.exit(1)
    
    # Get color space recommendations
    import cv2
    image = cv2.imread(image_path)
    recommendations = recommend_color_space(image)
    
    print("\nRecommended color spaces and channels for this image:")
    print("------------------------------------------------------")
    for i, rec in enumerate(recommendations):
        print(f"Recommendation {i+1}:")
        print(f"  Color space: {rec['color_space'].upper()}")
        print(f"  Channel: {rec['channel']}")
        print(f"  Explanation: {rec['explanation']}")
        print("")
    
    # Analyze color channels
    print("\nAnalyzing color channels...")
    analyze_color_channels(image_path)
    
    # Compare color spaces
    print("\nComparing different color spaces and parameters...")
    original_image, results = compare_color_spaces(image_path)
    
    # Save comparison figure
    output_path = f"{os.path.splitext(image_path)[0]}_comparison.png"
    save_comparison_figure(original_image, results, output_path)
    
    # Display results
    plot_color_space_comparison(original_image, results)
    plt.show()
    
    print(f"\nResults saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
